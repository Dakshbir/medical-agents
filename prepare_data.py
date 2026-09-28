"""
Fetches small, real samples of several medical QA/VQA benchmarks used by the
MDAgents pipeline and writes them to ./data/<dataset>/{test,train}.jsonl (plus
./data/<dataset>/images/ for the two image datasets).

Every loader here hits a real, publicly-accessible source over HTTP (mostly the
Hugging Face `datasets-server` REST API, which needs no auth and no `datasets`
package -- same approach as the original MedQA loader below) and was verified
by hand before being wired in here. No synthetic/placeholder data is ever
written.

Datasets covered (pass --dataset <name>):
  medqa      GBaker/MedQA-USMLE-4-options (HF datasets-server)
  pubmedqa   qiaojin/PubMedQA, config pqa_labeled (HF datasets-server)
  ddxplus    aai530-group6/ddxplus (HF datasets-server + release_evidences.json)
  symcat     teliov/symcat-to-synthea raw CSV scrape of symcat.com (GitHub raw)
  medbullets HanjieChen/ChallengeClinicalQA medbullets_op4.csv (GitHub raw)
  pmcvqa     RadGenome/PMC-VQA test.csv + images.zip (partial download via HTTP
             range requests, see load_pmcvqa)
  pathvqa    flaviagiammarino/path-vqa (HF datasets-server, images resolved to
             real per-row asset URLs and downloaded)

NOT covered: JAMA Clinical Challenge. The paper that introduced a JAMA
Clinical Challenge QA dataset (Chen et al., "Benchmarking LLMs on Answering
and Explaining Challenging Medical Questions", github.com/HanjieChen/
ChallengeClinicalQA) explicitly does NOT redistribute the JAMA data because of
JAMA's license -- the repo only ships article URLs plus a scraper that
requires the user to have a paid JAMA subscription. No legitimate public
mirror of this data was found, so it is intentionally skipped rather than
faked. See the final report for details.
"""
import argparse
import ast
import csv
import io
import json
import os
import random
import string

import requests

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch_hf_rows(dataset, config, split, offset, n):
    resp = requests.get(
        HF_ROWS_URL,
        params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": n},
        timeout=30,
    )
    resp.raise_for_status()
    return [r["row"] for r in resp.json()["rows"]]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# MedQA (original loader, unchanged)
# --------------------------------------------------------------------------
MEDQA_DATASET = "GBaker/MedQA-USMLE-4-options"


def load_medqa(n_test, n_train):
    def convert(split, n):
        rows = fetch_hf_rows(MEDQA_DATASET, "default", split, 0, n)
        return [
            {
                "question": r["question"],
                "options": r["options"],
                "answer_idx": r["answer_idx"],
                "answer": r["answer"],
            }
            for r in rows
        ]

    return convert("test", n_test), convert("train", n_train)


# --------------------------------------------------------------------------
# PubMedQA -- {"question", "context", "answer": "yes"|"no"|"maybe"}
# --------------------------------------------------------------------------
PUBMEDQA_DATASET = "qiaojin/PubMedQA"


def load_pubmedqa(n_test, n_train):
    # The `pqa_labeled` config (the 1k expert-annotated subset) only exposes a
    # single "train" split on the hub -- there is no official test/train
    # division. We take a contiguous slice for "test" and the next disjoint
    # slice for "train" (few-shot exemplars), offset so the two never overlap.
    rows = fetch_hf_rows(PUBMEDQA_DATASET, "pqa_labeled", "train", 0, n_test + n_train)
    out = []
    for r in rows:
        # `context.contexts` is the list of structured abstract sections
        # (BACKGROUND/METHODS/RESULTS/...); PubMedQA never includes a
        # CONCLUSIONS section in this field -- the conclusion sentence used to
        # derive the yes/no/maybe label lives separately in `long_answer`,
        # which we deliberately do NOT include, per the "no conclusion" spec.
        contexts = r["context"]["contexts"]
        out.append({
            "question": r["question"],
            "context": " ".join(contexts),
            "answer": r["final_decision"],
        })
    return out[:n_test], out[n_test:n_test + n_train]


# --------------------------------------------------------------------------
# DDXPlus -- {"question", "options", "answer_idx", "answer"}
# --------------------------------------------------------------------------
DDXPLUS_DATASET = "aai530-group6/ddxplus"
DDXPLUS_EVIDENCES_URL = "https://huggingface.co/datasets/aai530-group6/ddxplus/resolve/main/release_evidences.json"

_ddxplus_evidence_cache = None


def _ddxplus_evidence_map():
    global _ddxplus_evidence_cache
    if _ddxplus_evidence_cache is None:
        resp = requests.get(DDXPLUS_EVIDENCES_URL, timeout=30)
        resp.raise_for_status()
        _ddxplus_evidence_cache = resp.json()
    return _ddxplus_evidence_cache


def _render_ddxplus_evidence(code, evidence_map):
    # Evidence codes look like "E_91" (binary finding, present) or
    # "E_54_@_V_71" (categorical finding "E_54" with chosen value "V_71").
    # We render each as a short "Q: ... A: ..." line using the official
    # bilingual question bank shipped in release_evidences.json, so the
    # question text stays human-readable instead of showing raw codes.
    if "_@_" in code:
        ev_code, val_code = code.split("_@_", 1)
    else:
        ev_code, val_code = code, None
    ev = evidence_map.get(ev_code)
    if ev is None:
        return None
    q = ev.get("question_en", ev_code)
    if val_code:
        value_meaning = ev.get("value_meaning") or {}
        if val_code in value_meaning:
            meaning = value_meaning[val_code].get("en", val_code)
        else:
            # Numeric-scale evidences (data_type "C" with possible-values like
            # 0-10, e.g. pain intensity) have no value_meaning text -- the
            # code itself is the value, just strip the "V_" prefix.
            meaning = val_code[2:] if val_code.startswith("V_") else val_code
        return f"Q: {q} A: {meaning}."
    return f"Q: {q} A: Yes."


def load_ddxplus(n_test, n_train):
    evidence_map = _ddxplus_evidence_map()
    letters = string.ascii_uppercase

    def convert(split, n):
        rows = fetch_hf_rows(DDXPLUS_DATASET, "default", split, 0, n)
        out = []
        for r in rows:
            # EVIDENCES / DIFFERENTIAL_DIAGNOSIS come back from datasets-server
            # as Python-repr strings (single-quoted), not JSON -- ast.literal_eval
            # is required, json.loads will fail on them.
            evidences = ast.literal_eval(r["EVIDENCES"])
            differential = ast.literal_eval(r["DIFFERENTIAL_DIAGNOSIS"])  # [[name, prob], ...]

            initial = _render_ddxplus_evidence(r["INITIAL_EVIDENCE"], evidence_map)
            findings = [
                _render_ddxplus_evidence(code, evidence_map)
                for code in evidences
                if code != r["INITIAL_EVIDENCE"]
            ]
            findings = [f for f in findings if f]

            sex = "male" if r["SEX"] == "M" else "female"
            question = (
                f"A {r['AGE']}-year-old {sex} patient presents to the clinic. "
                f"Reason for consultation -- {initial} "
                f"Findings from the intake questionnaire: {' '.join(findings)} "
                "Given this presentation, which of the following is the correct diagnosis?"
            )

            options = {letters[i]: name for i, (name, _prob) in enumerate(differential[: len(letters)])}
            answer = r["PATHOLOGY"]
            answer_idx = next((k for k, v in options.items() if v == answer), None)
            if answer_idx is None:
                # Ground-truth pathology is occasionally absent from the
                # differential list; append it as an extra option so the
                # question always has a correct, selectable answer.
                answer_idx = letters[len(options)]
                options[answer_idx] = answer

            out.append({
                "question": question,
                "options": options,
                "answer_idx": answer_idx,
                "answer": answer,
            })
        return out

    return convert("test", n_test), convert("train", n_train)


# --------------------------------------------------------------------------
# SymCat -- {"question", "options", "answer_idx", "answer"}
# --------------------------------------------------------------------------
SYMCAT_DISEASES_URL = "https://raw.githubusercontent.com/teliov/symcat-to-synthea/master/symcat/symcat-801-diseases.csv"


def _load_symcat_diseases():
    # This CSV (mirrored on GitHub by teliov/symcat-to-synthea from the
    # symcat.com symptom checker) is itself a raw export from a recursive web
    # scraper: it has deeply nested "selectionN_selectionM_..._name" columns
    # left over from the scraper's pagination, and only ONE flat block of
    # columns (the "selection1_*" ones, ignoring anything with a nested
    # "selectionN_selectionM" prefix) is reliably populated, for ~50 of the
    # 801 conditions -- the remaining conditions are buried in the nested
    # duplicate columns and were not parsed here (out of scope for a 40-row
    # sample). We only use these clean "selection1_*" rows: each contiguous
    # block of rows sharing the same selection1_name is one condition, and
    # each row in that block is one (symptom, probability%) pair for it.
    resp = requests.get(SYMCAT_DISEASES_URL, timeout=60)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    diseases = {}
    for row in reader:
        name = (row.get("selection1_name") or "").strip()
        symptom = (row.get("selection1_symptoms_name") or "").strip()
        prob = (row.get("selection1_symptoms_probability") or "").strip()
        if not name or not symptom or not prob:
            continue
        try:
            prob_val = float(prob)
        except ValueError:
            continue
        diseases.setdefault(name, []).append((symptom, prob_val))
    return diseases


def load_symcat(n_test, n_train, seed=42):
    diseases = _load_symcat_diseases()
    names = list(diseases.keys())
    rng = random.Random(seed)
    rng.shuffle(names)

    if n_test + n_train > len(names):
        raise ValueError(
            f"symcat: only {len(names)} clean conditions available, cannot build "
            f"{n_test} test + {n_train} train non-overlapping questions"
        )

    letters = string.ascii_uppercase

    def build(name):
        top_symptoms = sorted(diseases[name], key=lambda x: -x[1])[:5]
        symptom_text = "; ".join(f"{s} (~{int(p)}% of patients)" for s, p in top_symptoms)
        question = (
            f"A patient presents with the following symptoms: {symptom_text}. "
            "Which of the following conditions is the most likely diagnosis?"
        )
        distractor_pool = [n2 for n2 in names if n2 != name]
        distractors = rng.sample(distractor_pool, k=min(3, len(distractor_pool)))
        choices = distractors + [name]
        rng.shuffle(choices)
        options = {letters[i]: c for i, c in enumerate(choices)}
        answer_idx = next(k for k, v in options.items() if v == name)
        return {"question": question, "options": options, "answer_idx": answer_idx, "answer": name}

    test_names = names[:n_test]
    train_names = names[n_test:n_test + n_train]
    return [build(n) for n in test_names], [build(n) for n in train_names]


# --------------------------------------------------------------------------
# MedBullets -- {"question", "options", "answer_idx", "answer"}
# --------------------------------------------------------------------------
MEDBULLETS_URL = "https://raw.githubusercontent.com/HanjieChen/ChallengeClinicalQA/main/medbullets/medbullets_op4.csv"


def load_medbullets(n_test, n_train, seed=42):
    # Using the 4-option version (medbullets_op4.csv) so the schema matches
    # MedQA's A-D options exactly; the repo also ships a 5-option variant
    # (medbullets_op5.csv) which we don't use here.
    resp = requests.get(MEDBULLETS_URL, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = [r for r in reader if r.get("question")]
    rng = random.Random(seed)
    rng.shuffle(rows)

    def convert(batch):
        out = []
        for r in batch:
            out.append({
                "question": r["question"],
                "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]},
                "answer_idx": r["answer_idx"],
                "answer": r["answer"],
            })
        return out

    return convert(rows[:n_test]), convert(rows[n_test:n_test + n_train])


# --------------------------------------------------------------------------
# PMC-VQA (image+text) -- {"question", "options", "answer_idx", "answer", "image_path"}
# --------------------------------------------------------------------------
PMCVQA_TEST_CSV_URL = "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/test.csv"
PMCVQA_IMAGES_ZIP_URL = "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/images.zip"


def load_pmcvqa(n_test, n_train, images_dir, seed=42):
    # RadGenome/PMC-VQA's HF datasets-server auto-conversion is broken
    # (schema mismatch between train/train_2/test/test_2 csvs -> 500 error),
    # so we read test.csv directly (10MB metadata file, safe to fetch whole).
    # images.zip itself is ~19GB, so we do NOT download it -- instead we use
    # `remotezip` to open it over HTTP range requests and pull out only the
    # ~40 specific member files (by name, from Figure_path) we need for this
    # sample, verified against real JPEG bytes before being used here.
    from remotezip import RemoteZip

    resp = requests.get(PMCVQA_TEST_CSV_URL, timeout=60)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = [r for r in reader if r.get("Figure_path")]
    rng = random.Random(seed)
    rng.shuffle(rows)

    os.makedirs(images_dir, exist_ok=True)
    rz = RemoteZip(PMCVQA_IMAGES_ZIP_URL)

    out = []
    i = 0
    while len(out) < n_test + n_train and i < len(rows):
        r = rows[i]
        i += 1
        fname = r["Figure_path"].strip()
        try:
            data = rz.read(f"images/{fname}")
        except KeyError:
            continue
        local_name = f"{len(out):03d}_{fname}"
        with open(os.path.join(images_dir, local_name), "wb") as f:
            f.write(data)
        # Raw "Choice A"/"Choice B"/... columns already embed their own
        # "A: "/"B: " prefix (e.g. " A: Chromatin condensation "); strip that
        # off so it isn't duplicated against our own A/B/C/D option keys.
        def _clean_choice(text, letter):
            text = text.strip()
            prefix = f"{letter}:"
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
            return text

        options = {
            "A": _clean_choice(r["Choice A"], "A"),
            "B": _clean_choice(r["Choice B"], "B"),
            "C": _clean_choice(r["Choice C"], "C"),
            "D": _clean_choice(r["Choice D"], "D"),
        }
        out.append({
            "question": r["Question"].strip(),
            "options": options,
            "answer_idx": r["Answer_label"].strip(),
            "answer": r["Answer"].strip(),
            "image_path": f"images/{local_name}",
        })

    return out[:n_test], out[n_test:n_test + n_train]


# --------------------------------------------------------------------------
# Path-VQA (image+text) -- {"question", "options": null, "answer", "image_path"}
# --------------------------------------------------------------------------
PATHVQA_DATASET = "flaviagiammarino/path-vqa"


def load_pathvqa(n_test, n_train, images_dir):
    # The paper (Appendix A) explicitly restricts Path-VQA to yes/no
    # questions only, "for the purpose of maintaining a standardized
    # accuracy evaluation" -- Path-VQA's source data is otherwise a mix of
    # yes/no and free-form short-answer questions with no multiple-choice
    # candidates at all. Only ~45% of rows are yes/no, so we fetch in
    # growing batches and filter until n is reached, rather than a single
    # fixed-size pull (an earlier version of this loader kept everything,
    # which doesn't match the paper's methodology and can't be graded by
    # this project's yes/no/maybe answer extraction).
    # Images are exposed by datasets-server as short-lived signed URLs
    # (`row["image"]["src"]`), so we download the real bytes immediately
    # after fetching each row rather than storing the URL (it expires).
    os.makedirs(images_dir, exist_ok=True)

    def fetch(split, n, prefix):
        out = []
        offset = 0
        batch = max(n * 3, 30)
        idx = 0
        while len(out) < n:
            rows = fetch_hf_rows(PATHVQA_DATASET, "default", split, offset, batch)
            if not rows:
                break
            offset += batch
            for r in rows:
                if r["answer"].strip().lower() not in ("yes", "no"):
                    continue
                src = r["image"]["src"]
                img_resp = requests.get(src, timeout=30)
                img_resp.raise_for_status()
                fname = f"{prefix}_{idx:03d}.jpg"
                with open(os.path.join(images_dir, fname), "wb") as f:
                    f.write(img_resp.content)
                out.append({
                    "question": r["question"],
                    "options": None,
                    "answer": r["answer"].strip().lower(),
                    "image_path": f"images/{fname}",
                })
                idx += 1
                if len(out) == n:
                    break
        return out

    return fetch("test", n_test, "test"), fetch("train", n_train, "train")


# --------------------------------------------------------------------------
DATASET_LOADERS = {
    "medqa": "medqa",
    "pubmedqa": "pubmedqa",
    "ddxplus": "ddxplus",
    "symcat": "symcat",
    "medbullets": "medbullets",
    "pmcvqa": "pmcvqa",
    "pathvqa": "pathvqa",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default="medqa")
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--n_train", type=int, default=None)
    args = parser.parse_args()

    # medqa keeps its original historical defaults (20/10); every other
    # dataset added here defaults to the 30/10 sample size requested for
    # this batch of loaders.
    default_n_test, default_n_train = (20, 10) if args.dataset == "medqa" else (30, 10)
    n_test = args.n_test if args.n_test is not None else default_n_test
    n_train = args.n_train if args.n_train is not None else default_n_train

    out_dir = os.path.join(DATA_DIR, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "medqa":
        test_rows, train_rows = load_medqa(n_test, n_train)
    elif args.dataset == "pubmedqa":
        test_rows, train_rows = load_pubmedqa(n_test, n_train)
    elif args.dataset == "ddxplus":
        test_rows, train_rows = load_ddxplus(n_test, n_train)
    elif args.dataset == "symcat":
        test_rows, train_rows = load_symcat(n_test, n_train)
    elif args.dataset == "medbullets":
        test_rows, train_rows = load_medbullets(n_test, n_train)
    elif args.dataset == "pmcvqa":
        test_rows, train_rows = load_pmcvqa(n_test, n_train, os.path.join(out_dir, "images"))
    elif args.dataset == "pathvqa":
        test_rows, train_rows = load_pathvqa(n_test, n_train, os.path.join(out_dir, "images"))
    else:
        raise ValueError(args.dataset)

    write_jsonl(os.path.join(out_dir, "test.jsonl"), test_rows)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows)

    print(f"Wrote {len(test_rows)} test and {len(train_rows)} train examples to {out_dir}")
    for row in test_rows[:3]:
        print(json.dumps(row, ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()
