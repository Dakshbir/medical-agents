# Architecture

This document explains exactly what runs at each stage of the pipeline, what
is and isn't faithful to the original paper (Kim et al., *MDAgents: An
Adaptive Collaboration of LLMs for Medical Decision-Making*, NeurIPS 2024),
and the concrete design decisions made while implementing it in this repo.

## There is no separate trained model anywhere in this pipeline

Every stage below is the *same* general-purpose LLM (whichever `--model` you
pass — a Groq-hosted model by default, or a Gemini model for multimodal
questions), called repeatedly with a different role-playing system prompt.
There is no fine-tuning, no separate classifier, no separate retriever model,
no separate voting model. "Different experts" are different `Agent` objects
— each with its own independent conversation history and system prompt — but
all backed by the identical underlying model weights. Specialization is
achieved entirely through prompt engineering, not through different models.

## Pipeline stages

### 1. Complexity check (`determine_difficulty` in `utils.py`)

One LLM call, persona *"a medical expert who conducts initial assessment."*
Given the question and a text description of the three tiers (basic /
intermediate / advanced, corresponding to the paper's low / moderate /
high), it's asked to classify and briefly explain why. The label is parsed
out of the free-text response.

### 2. Expert recruitment

- **Basic (low)**: no recruitment — a single agent answers directly.
- **Intermediate (moderate)**: one LLM call, persona *"an experienced
  medical expert who recruits a group of experts."* Asked to invent N
  (default 5) plausible specialist roles, descriptions, and a reporting
  hierarchy. The response is parsed into role/description pairs, each of
  which spins up a new independent `Agent`.
- **Advanced (high)**: one LLM call asks for 3 Multidisciplinary Teams
  (MDTs) of 3 clinicians each, following the paper's Initial Assessment Team
  (IAT) → other MDTs → Final Review and Decision Team (FRDT) structure.

Recruiter output is free text; different models format lists very
differently (markdown tables, en-dashes instead of hyphens, bold markers).
`_normalize_llm_list_text()` and defensive parsing in
`process_intermediate_query`/`process_advanced_query` handle this — see
"Known fragility" below.

### 3. Initial assessment

Each recruited expert (its own `Agent`, own conversation) independently
answers the question with **no visibility into the other experts' answers
yet** — a parallel, independent first pass. If `--use_rag` is on, a shared
block of retrieved evidence (see "RAG" below) is included in every expert's
prompt at this stage.

### 4. Collaborative discussion (intermediate path only)

For up to `num_rounds` (default 5) rounds, each with up to `num_turns`
(default 5) turns:

- Every agent is shown the current transcript and asked (another LLM call)
  whether it wants to message a specific peer; if so, it generates that
  message, logged as `(round, from, to, message)`.
- At the end of each round, every agent gives an updated "final answer for
  this round." **Consensus is checked for real** (`_extract_answer_letter`
  on every agent's answer — if they all land on the same letter, the round
  loop stops early).
- If consensus wasn't reached and rounds remain, a **moderator reviews the
  round and gives the panel feedback**, which is fed back to every agent
  before the next round (Algorithm 1, lines 13–17 in the paper). This loop
  was entirely absent from the initial implementation of this repo — every
  round just repeated the same debate mechanic with no guidance injected
  between rounds.

The advanced path has a related but structurally different version of this:
teams don't debate turn-by-turn, but each team now sees the *prior* teams'
reports before producing its own (`prior_reports` param on `Group.interact`)
— matching the paper's "each team builds upon the foundation laid by the
previous."

### 5. Review and final decision

- **Basic**: the single agent's own answer.
- **Intermediate**: a moderator persona reviews the **entire accumulated
  interaction history** (all rounds, including any moderator feedback given
  between rounds) — not just a snapshot of the last round, which is what the
  original code in this repo did.
- **Advanced**: a decision-maker persona reviews **all three team reports**
  (IAT + other MDTs + FRDT) — the original code in this repo only ever
  passed the IAT's report into this prompt, silently discarding the other
  two team's work.

All three paths can sample the final decision at multiple temperatures and
take a real majority vote over the extracted answer letters
(`majority_vote()`), matching the paper's "temperature ensemble." This
defaults to a single sample (same cost as before); set
`ENSEMBLE_TEMPERATURES="0.0,0.5,1.0"` to turn on real ensembling.

## Confidence and structured explainability

Added on top of the base pipeline (not part of the original paper) to
support a downstream project's requirement for calibrated uncertainty and
machine-readable explanations (see `HANDOFF.md` for that project's context).

**`consensus_confidence(texts)`** (`utils.py`) takes the raw response texts
from multiple independent opinions on the same question - the intermediate
path's recruited experts, the advanced path's per-team reports, or the basic
path's ensemble-temperature samples - extracts each one's answer letter, and
returns:
- `confidence`: vote share, i.e. the fraction that agree with the majority
  answer (1.0 = everyone agreed). This is what gets calibrated.
- `entropy`: normalized Shannon entropy over the answer distribution (the
  same underlying signal the paper itself computes for its own consensus-
  dynamics analysis, Appendix B - lower entropy = higher confidence).
- `n`, `votes`, `winner`: the raw counts behind the score.

Returns `None` fields (not zeros) when nothing could be extracted, so
"no data" is distinguishable from "zero confidence."

**Important caveat**: for the basic (single-agent) path, confidence is only
meaningful if `ENSEMBLE_TEMPERATURES` has more than one value - with the
default single sample, "agreement" is trivially 1.0 (an opinion always
"agrees" with itself). Set e.g. `ENSEMBLE_TEMPERATURES="0.0,0.7"` for a real
signal there; the intermediate/advanced paths get real variance for free
from having multiple independent agents/teams regardless of this setting.

**`build_explanation(...)`** (`utils.py`) packages the confidence score
alongside reasoning text, RAG evidence (with citation metadata), and
per-specialist/per-team attribution (including whether each one agreed with
the final answer) into one structured dict, attached as
`final_decision['explanation']`:
```python
{
  "answer": "C",
  "confidence": 0.8,
  "confidence_detail": {"confidence": 0.8, "entropy": 0.32, "n": 5, "votes": {"C": 4, "A": 1}, "winner": "C"},
  "reasoning": "...",              # the moderator/decision-maker's own text
  "evidence": [{"source": "pubmed", "id": "12345678", "title": "...", "url": "...", "snippet": "..."}],
  "specialist_attribution": [{"role": "Cardiologist", "opinion": "...", "agreed_with_final": true}],
  "complexity": "intermediate",
  "consensus_reached": true,
}
```
`rag.py`'s `retrieve_structured()` is the source of the `evidence` list -
`retrieve()` (the original plain-text interface, still used for prompt
injection) is now just a thin wrapper that extracts the `display` string
from each structured item, so getting citation metadata doesn't cost a
second network round-trip.

**Calibration evaluation** (`evaluate_calibration.py`): reads one or more
`output/*.json` files (each result row now carries a `confidence` field
alongside `correct`), buckets predictions by confidence, and reports actual
accuracy per bucket plus Expected Calibration Error (ECE) - i.e. does "80%
confident" really mean "correct ~80% of the time"? Also saves a reliability
diagram (`calibration_reliability.png`) if matplotlib is installed.
```bash
python evaluate_calibration.py output/*.json
```

## RAG ("MedRAG")

The paper's own MedRAG uses large curated corpora (PubMed abstracts,
StatPearls, textbooks) with a pre-built vector index. That's out of scope
here for cost/complexity reasons. Instead, `rag.py` implements **lightweight
live retrieval**: at inference time, it queries NCBI E-utilities (PubMed
search + abstract fetch) and Wikipedia's public API for a few short,
topically-relevant snippets, with no local corpus, no vector DB, no API key,
and no cost. It never raises — any failure degrades to fewer/zero results,
logged as a warning. Toggle with `--use_rag` / `--no-use_rag` (default on).

## Multimodal (image) support

Groq's hosted models are text-only. Image questions (PMC-VQA, Path-VQA) are
routed to whatever Gemini model you pass via `--model` — the `Agent` class
loads the image with Pillow and passes it alongside the text to the Gemini
API. Passing an image path to a non-Gemini model raises a clear
`ValueError` rather than silently ignoring the image.

## Known fragility / honest limitations

- **List-parsing is inherently brittle.** The recruiter/team-structure
  prompts ask the model to return a specific plain-text format, but
  different models format free text differently. This repo has accumulated
  defensive normalization and fallback parsing for the formatting quirks
  observed in testing (markdown bold, en-dashes, markdown tables, narrow
  spaces), but it is not a guaranteed-robust parser for arbitrary future
  model output — a sufficiently different format could still degrade
  gracefully to fewer/garbled recruited experts rather than an exact match.
- **RAG is a lightweight stand-in, not the paper's MedRAG.** See above.
- **SymCat's usable condition pool is ~50, not the full 801** — the public
  GitHub mirror used to source this dataset (`teliov/symcat-to-synthea`) is
  a messy recursive-scraper export where only ~50 conditions are exposed in
  flat, parseable columns; the rest are buried in nested pagination
  artifacts. Multiple-choice distractors for this dataset are synthesized
  from that 50-condition pool, not sourced from the original data (unlike
  DDXPlus, whose distractors come from the dataset's own differential
  diagnosis candidates).
- **JAMA Clinical Challenge is not implemented.** The paper that defines
  this dataset (Chen et al., *Benchmarking LLMs on Answering and Explaining
  Challenging Medical Questions*) explicitly does not redistribute the data
  due to JAMA's license — no legitimate public mirror exists. Rather than
  fabricate or scrape paywalled content, this dataset is skipped.
- **MedVidQA and MIMIC-CXR are out of scope for now** (by agreement — video
  processing and MIMIC-CXR's PhysioNet credentialing requirement are
  significant additional scope beyond the current phase).
- **Gemini vision calls have shown intermittent hangs in testing** on this
  network environment, where the underlying SDK's own
  `request_options={'timeout': ...}` did not reliably bound a stalled
  streaming call. A hard, application-level timeout
  (`ThreadPoolExecutor.result(timeout=...)`) now guarantees the pipeline
  fails with a clear `RuntimeError` after a bounded number of retries
  instead of hanging indefinitely — but the intermittent root cause itself
  (network/proxy/environment specific) has not been fully diagnosed.
- **Groq's free-tier daily token quota (TPD) is easily exhausted** by real
  usage, and this is the single biggest operational constraint on this
  project. Concretely observed: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`,
  and `qwen/qwen3.8-27b` are each capped at **200,000 tokens/day**, tracked
  **per Groq organization, not per API key** (a new key from the same
  account shares the same exhausted pool - only a key from a genuinely
  different account gets fresh quota). A single "intermediate" complexity
  question can burn several thousand tokens across the recruiter + N experts
  + multiple debate rounds, so a handful of benchmark runs can exhaust a
  day's quota. `FAST_MODEL` (used for the difficulty-check call on *every*
  question regardless of dataset/model) is a special risk: if it hits its
  daily cap, every question fails immediately, on every dataset, regardless
  of which `--model` you chose - always check `FAST_MODEL`'s quota isn't
  already exhausted before assuming a `--model` swap will fix cascading
  failures. Untested/less-loaded fallback models on this account:
  `allam-2-7b` (small general model) and `openai/gpt-oss-safeguard-20b` (a
  safety/moderation-tuned variant - usable but not built for open-ended
  reasoning, expect lower quality). Gemini's free tier is far stricter still
  (observed: **20 requests/day** for `gemini-2.5-flash`, same
  per-organization-not-per-key caveat), which matters because it's also the
  Groq-failure fallback path - if Gemini's quota is already exhausted when
  Groq rate-limits, the fallback itself fails too (this is now a fast,
  clearly-labeled failure showing *both* the original and fallback errors,
  not a silent hang - see `_gemini_fallback_generate`'s docstring).
- **Batch runs are resilient to individual question failures.** `main.py`
  wraps each question in try/except and writes the accumulated results file
  after *every* question, not just at the end. An earlier version only wrote
  once at the very end - a single uncaught exception (e.g. the quota
  cascade above) on question 8 of 10 silently discarded all 7
  already-computed results, not just the failed one. A failed question is
  recorded as `{"question": ..., "error": "...", "difficulty": null}` in the
  results list and graded as not-correct; `num_failed` is tracked separately
  from `num_graded` in the output JSON.
- **Debate volume, not just round/turn caps, drives cost.** Agents will
  often keep "wanting to talk" purely to affirm agreement, not only to
  resolve disagreement - so 5 agents × up to `num_turns` × up to
  `num_rounds` can spiral into 100+ LLM calls for a single question even
  when everyone already agrees. `--num_rounds`/`--num_turns` (both default
  5, matching the original paper) are exposed as CLI flags specifically so a
  cost-bounded run can lower them (e.g. 2/3) without editing code.
