# Team Handoff: MDAgents Component → Digital Twin Integration

**Purpose of this document.** This repo implements and extends the
multi-agent clinical reasoning core of a larger B.Tech final year project
(see `BTP_Proposal_Dakshbir (1) (1).pdf` in this repo for the full proposal).
This document is written to be handed to a teammate — or to an AI coding
agent working on their behalf — who is picking up the **digital twin**
module and needs to integrate it with what's already built here, without
needing the full conversation history that produced it. Read this file
first; it links to `ARCHITECTURE.md` for pipeline internals where relevant,
but everything needed to *start integrating* is here.

---

## 1. Project context (why this repo exists)

The B.Tech proposal ("Adaptive Multi-Agent Digital Twin Framework for
Speech-Enabled Explainable Clinical Decision Support") frames the project as
an **AI-systems research project using clinical decision support as its test
domain**, not a healthcare chatbot. Its explicit claim to novelty is not "a
new multi-agent architecture" — the proposal itself says adaptive routing
and multi-agent collaboration are "active research directions... not unique
to this proposal." The claimed contribution is **integration**: adaptive
routing + consensus + speech accessibility + a digital twin + calibrated
uncertainty + structured explainability, evaluated together as one system.

The proposal's system architecture (see the PDF, page 2) is:

```
PATIENT (speech input)
  -> ASR (speech -> text)
  -> Clinical Information Extraction (NLP): symptoms, vitals, history -> structured data
  -> Patient Digital Twin (simulated-data prototype): evolving patient state + what-if simulation
  -> Adaptive Multi-Agent Router: selects which specialists, how many, when to stop
  -> [General Physician, Cardiology, Endocrinology, Radiology, Pharmacy, ...]
  -> Consensus Engine: aggregate / weight / reconcile disagreement
  -> Explainable Recommendation: reason + evidence + guideline + confidence/uncertainty
  -> Speech Synthesis (TTS) -> patient hears the response
```

**This repo (MDAgents) implements the "Adaptive Multi-Agent Router" +
"Consensus Engine" + "Explainable Recommendation" boxes** — i.e. everything
from where structured patient data enters the router through to the
explained recommendation. It does **not** implement ASR/TTS or the digital
twin itself. The proposal's four research questions and how this repo
relates to each:

1. *Does adaptive multi-specialist routing beat a single LLM at comparable
   cost?* — Partially answerable now: the adaptive complexity-based routing
   (basic/intermediate/advanced) exists and is instrumented with confidence
   scores; a rigorous baseline-vs-adaptive comparison run has not yet been
   completed (see §5, "What's not done yet").
2. *How should conflicting specialist recommendations be reconciled?* —
   **Implemented.** See §3 below (moderator-feedback loop + majority vote).
3. *Does speech-first interaction improve usability?* — **Not implemented
   in this repo.** Out of scope here; belongs to a separate ASR/TTS module.
4. *Can uncertainty be estimated in a way that's calibrated?* —
   **Implemented and partially validated.** See §3 ("Confidence") and
   `evaluate_calibration.py`. Full calibration numbers are blocked on API
   quota right now (§6).

---

## 2. Repo orientation

```
main.py                CLI batch-evaluation harness (runs N questions from a dataset, grades, saves results)
utils.py                Core pipeline: Agent class, Group class, process_basic/intermediate/advanced_query,
                         complexity classification, RAG integration, confidence scoring, explanation building
rag.py                  Lightweight live retrieval (PubMed + Wikipedia), no local corpus/vector DB
demo_app.py              Streamlit interactive single-case demo (complexity check -> recruit -> collaborate -> decide)
prepare_data.py         Fetches small real samples of 7 benchmark datasets from public sources
evaluate_calibration.py  Reads output/*.json, reports whether confidence scores are actually calibrated
run_benchmark.sh        Orchestrates main.py across all datasets in one command
ARCHITECTURE.md         Pipeline internals, stage-by-stage, what differs from the original paper
README.md               Setup + usage
data/                   (gitignored) fetched dataset samples, created by prepare_data.py
output/                 (gitignored) benchmark results as JSON
.env / .env.example     API keys (GROQ_API_KEY, GENAI_API_KEY, FAST_MODEL, etc.) - .env is gitignored
```

**Setup** (full detail in `README.md`): `python -m venv .venv` → activate →
`pip install -r requirements.txt` → `cp .env.example .env` and fill in keys
→ `python prepare_data.py --dataset medqa` to get sample data →
`python main.py --dataset medqa --model openai/gpt-oss-120b` to run.

---

## 3. What this repo actually does (condensed — see `ARCHITECTURE.md` for full detail)

**There is no trained/fine-tuned model anywhere in this pipeline.** Every
stage is the same general-purpose LLM (a Groq-hosted model, or Gemini for
image questions), called repeatedly with different role-playing system
prompts. "Different experts" = different `Agent` objects (own conversation
history, own system prompt) backed by the same underlying model weights.
This matters for the digital twin integration: **there is no model to
retrain or fine-tune to make it "patient-aware" — patient context has to be
injected as text into the prompts the existing pipeline already builds.**

Pipeline, given a question string:
1. **Complexity check** — one LLM call classifies the question as
   basic/intermediate/advanced (maps to the proposal's low/moderate/high).
2. **Recruitment** — basic: no recruitment (single agent). intermediate: an
   LLM invents ~5 specialist roles. advanced: an LLM organizes 3 teams of 3
   (Initial Assessment Team → other MDTs → Final Review and Decision Team).
3. **Initial assessment** — each recruited agent answers independently, no
   cross-visibility yet. RAG evidence (if enabled) is injected here.
4. **Collaborative discussion** (intermediate/advanced only) — agents debate
   over rounds/turns; real consensus detection (do all agents' extracted
   answers agree?); a moderator gives the panel feedback between rounds if
   they haven't converged (this is Algorithm 1 from the paper).
5. **Final decision** — a moderator/decision-maker persona reviews the full
   discussion (not just a snapshot) and produces the answer.

**Confidence and structured explanation** (built for this project, not part
of the original MDAgents paper): every `final_decision` dict returned by
`process_basic_query` / `process_intermediate_query` / `process_advanced_query`
now looks like:
```python
{
  "majority": {0.0: "raw response text", ...},   # per-temperature raw samples
  "answer": "Answer: (C) ...",                    # the majority-voted response text
  "explanation": {
    "answer": "C",                                 # extracted option letter
    "confidence": 0.8,                              # vote share across independent opinions, 0-1
    "confidence_detail": {"confidence": 0.8, "entropy": 0.32, "n": 5, "votes": {"C": 4, "A": 1}, "winner": "C"},
    "reasoning": "...",                             # the decision-maker's own text
    "evidence": [{"source": "pubmed", "id": "...", "title": "...", "url": "...", "snippet": "..."}],
    "specialist_attribution": [{"role": "Cardiologist", "opinion": "...", "agreed_with_final": true}],
    "complexity": "intermediate",                   # "basic" | "intermediate" | "advanced"
    "consensus_reached": true,
  }
}
```
**This `explanation` dict is the primary integration surface.** It's
already shaped close to the proposal's Explainability Layer requirement
(reason + evidence + attribution + confidence) and is what a speech-
synthesis (TTS) stage or a UI would consume field-by-field.

Functions relevant to integration (all in `utils.py`, all stable/tested):
- `create_question(sample, dataset) -> (question_text, img_path)` — builds
  the prompt-ready question from a raw data row. **This is the main input
  seam** — see §4.
- `process_basic_query(question, examplers, model, args, img_path=None)`
- `process_intermediate_query(question, examplers, model, args, img_path=None)`
- `process_advanced_query(question, model, args, img_path=None)`
- `determine_difficulty(question, difficulty)` — call this first; its return
  value (`'basic'|'intermediate'|'advanced'`) tells you which of the three
  `process_*` functions to call.
- `consensus_confidence(texts)` / `build_explanation(...)` — reusable for
  any new comparison logic (e.g. scoring confidence across what-if
  scenarios — see §7).
- `get_rag_evidence_and_citations(query, use_rag, k=3)` — returns
  `(evidence_text_for_prompt, structured_citations_list)`.

All three `process_*` functions take an `args` object (in practice,
whatever `argparse.Namespace` `main.py` builds) and read flags off it via
`getattr(args, 'flag_name', default)` — so **you can pass any object with
the right attributes**, not necessarily real CLI args. Relevant attributes:
`use_rag` (bool), `use_moderator_review` (bool), `num_rounds` (int, default
5), `num_turns` (int, default 5), `dataset` (str, only used to decide
whether to inject MedQA-style few-shot examplers).

---

## 4. How to feed new input into this pipeline (the main integration seam)

`main.py`'s batch loop does, per question:
```python
question, img_path = create_question(sample, args.dataset)
difficulty = determine_difficulty(question, args.difficulty)
if difficulty == 'basic':
    final_decision = process_basic_query(question, examplers, args.model, args, img_path=img_path)
elif difficulty == 'intermediate':
    final_decision = process_intermediate_query(question, examplers, args.model, args, img_path=img_path)
elif difficulty == 'advanced':
    final_decision = process_advanced_query(question, args.model, args, img_path=img_path)
```
`question` is **just a string**. `create_question()` builds it from a
`sample` dict shaped like `{"question": str, "options": {...}|None,
"context": str (optional, PubMedQA-style), "image_path": str (optional)}` —
but nothing downstream cares how that string was constructed. **The digital
twin's job, from this pipeline's point of view, is to produce a richer
question string** (patient history + current presentation + whatever
context is relevant) and hand it to `determine_difficulty()` /
`process_*_query()` exactly like a benchmark question would be. You do not
need to go through `create_question()` or the `sample` dict schema at all if
you're not using the benchmark dataset loaders — just build your own
context string and call the same three functions `main.py` calls.

---

## 5. Digital twin integration guide

### 5.1 What the proposal means by "digital twin"

Per the proposal (page 2): *"A lightweight, continuously-updated patient
state model used for illustrative what-if comparisons across treatment
options... explicitly scoped as simulated-data, not clinically validated."*
This is **not** a physiological simulation engine — it's a patient state
object that evolves over a session/interaction, plus the ability to ask
"what would the recommendation be if X were different?"

### 5.2 Where it plugs in

Per the proposal's architecture diagram (§1), the digital twin sits
**between structured input extraction and the multi-agent router**. In this
repo's terms: **it plugs in before `determine_difficulty()`/`process_*`
functions are called, by constructing the question/context string they
receive.** It does not need to touch, wrap, or modify `utils.py`'s internals
— treat `create_question`, `determine_difficulty`, and the three
`process_*_query` functions as a stable API you call into, not code to
change.

### 5.3 Suggested design (not prescriptive — the team should own this)

**A. A `PatientState` object**, e.g. in a new `digital_twin.py`:
```python
@dataclass
class PatientState:
    demographics: dict          # age, sex, ...
    history: list[str]          # prior conditions, surgeries, allergies
    medications: list[str]
    vitals_timeline: list[dict] # [{"timestamp": ..., "bp": ..., "hr": ..., ...}, ...]
    symptoms_timeline: list[dict]  # evolving complaints over the session
    notes: list[str]            # free-text clinician/patient notes accumulated so far
```
This is intentionally simple (proposal explicitly scopes it as
simulated-data, not clinically validated) — a dict or dataclass that's easy
to serialize to text is enough. Whether it's persisted in memory (e.g.
Streamlit `st.session_state`, mirroring how `demo_app.py` already holds
per-session UI state), a local file, or a small database is an
implementation choice for the team; nothing in this repo assumes any
particular persistence mechanism.

**B. A rendering function** that turns the current state into the context
string the pipeline consumes:
```python
def render_state_as_context(state: PatientState, new_complaint: str) -> str:
    # produce a text block: demographics + relevant history + vitals trend +
    # current complaint, formatted the same way create_question() formats
    # a benchmark question (see utils.py for that shape) - this becomes the
    # `question` string passed to determine_difficulty()/process_*_query().
    ...
```
This mirrors exactly how this repo already injects RAG evidence as a text
block prepended to prompts (`get_rag_evidence_and_citations` in `utils.py`)
— same pattern, different source of context.

**C. What-if simulation**, using the pipeline's existing confidence/
explanation machinery rather than inventing new comparison logic:
```python
def compare_interventions(state: PatientState, interventions: list[str], model, args) -> list[dict]:
    results = []
    for intervention in interventions:
        hypothetical_state = simulate(state, intervention)  # simulated projection, not clinically validated
        question = render_state_as_context(hypothetical_state, "...")
        difficulty = determine_difficulty(question, args.difficulty)
        decision = process_intermediate_query(question, [], model, args)  # or whichever path
        results.append({"intervention": intervention, "decision": decision})
    return results
```
Each `decision['explanation']['confidence']` is directly comparable across
scenarios — "the system is more confident recommending X under intervention
A (0.8) than intervention B (0.5)" is a real, already-implemented signal,
not something that needs to be built from scratch. `simulate(...)` (the
projection from current state + a hypothetical intervention to a
hypothetical future state) is the part that's genuinely new work for this
team — it can be as simple as an LLM call asked to project plausible
short-term effects, explicitly labeled as an illustrative simulation.

**D. Session-based demo.** `demo_app.py` is a good template for a
digital-twin-aware interactive demo: it's a single-session Streamlit app
today (no persistence across questions). Extending it with
`st.session_state` to hold a `PatientState` that updates turn-by-turn, and
re-rendering context each turn via 5.3.B, is a natural next artifact —
follow its existing pattern (see `run_low`/`run_moderate`/`run_high` in
`demo_app.py` for how it currently calls into the `Agent` class directly)
rather than routing through `main.py`'s batch-benchmark path, which has no
concept of a multi-turn session.

### 5.4 What NOT to do

- Don't modify `process_basic_query`/`process_intermediate_query`/
  `process_advanced_query` internals to "know about" patient state. Keep
  the digital twin's output as a plain string/context block fed in at the
  same seam benchmark questions use. This keeps the validated pipeline
  logic untouched and the integration testable in isolation.
- Don't try to make patient state "flow through" the multi-agent debate
  mechanically (e.g. giving each recruited agent direct access to a
  `PatientState` object) — the existing agents only consume text prompts.
  If specific agents need different views of the state, render different
  text blocks per agent rather than changing the `Agent` class.
- Don't assume dataset-loader (`prepare_data.py`/`create_question`) schema
  applies to digital-twin-originated questions — those are for static
  benchmark rows, not evolving sessions. Build context strings directly.

---

## 6. Current operational status (as of this handoff)

- **Implementation status**: complete and validated via smoke tests across
  all three complexity paths, all 7 datasets' schemas, and the confidence/
  explanation pipeline. See `ARCHITECTURE.md` for the full list of bugs
  found and fixed during validation (worth reading before extending this
  code — several were subtle, e.g. an answer-extraction regex that silently
  scored correct answers as wrong).
- **Benchmark run: incomplete, paused on API quota.** A partial run exists:
  MedQA got 4/10 questions graded (2 correct) before the Groq organization's
  daily token quota was exhausted across all three text models tried
  (`gpt-oss-20b`, `gpt-oss-120b`, `qwen/qwen3.8-27b` — each capped at
  200,000 tokens/day, **shared across the whole org, not per API key**).
  `pubmedqa`, `ddxplus`, `symcat`, `medbullets` have not been benchmarked
  yet. The two image datasets (`pmcvqa`, `pathvqa`) are additionally blocked
  on Gemini's free tier (observed cap: 20 requests/day for
  `gemini-2.5-flash`, same per-org caveat). **Resuming this requires a Groq
  API key from a different account** (not just a new key from the same
  account) and ideally a Gemini key from a different account too, or
  waiting for the daily quota windows to roll over.
- **No calibration numbers yet** (`evaluate_calibration.py` is ready to run,
  but needs a completed benchmark batch with real accuracy/confidence pairs
  to be meaningful — 4 data points isn't enough for a reliability diagram).
- Run `bash run_benchmark.sh` (env vars `N`, `NUM_ROUNDS`, `NUM_TURNS`,
  `TEXT_MODEL`, `VISION_MODEL` override defaults) once quota is available
  to pick this back up; it's resilient to individual question failures and
  writes results incrementally, so it's safe to just re-run.

---

## 7. Suggested next steps (for whoever picks this up)

Roughly in order of dependency, not necessarily priority:

1. **Get fresh API quota** (Groq from a different account is the immediate
   blocker) and complete the 5-dataset text benchmark, then run
   `evaluate_calibration.py` to get real calibration numbers — this is the
   proposal's research question 4, and the data pipeline for it is already
   built and waiting.
2. **Baseline comparison** (research question 1): the proposal explicitly
   wants adaptive multi-agent vs. a single-LLM baseline. `process_basic_query`
   already *is* a single-agent baseline in spirit, but a clean head-to-head
   (same questions, same model, adaptive routing on vs. forced-basic-only)
   hasn't been run as a dedicated comparison yet — `main.py --difficulty basic`
   vs `--difficulty adaptive` on the same dataset/N gets you this directly.
3. **Digital twin module** (this handoff's main subject) — §5 above.
4. **Speech I/O** (ASR/TTS) — entirely separate from this repo; the natural
   integration point once built is wrapping whatever produces a question
   string (§4) and whatever consumes `explanation['reasoning']` (§3) for
   TTS output.
5. **Guideline-specific citations** — current RAG (`rag.py`) cites PubMed/
   Wikipedia, not curated clinical guidelines (NICE/ACC-AHA/WHO), which are
   mostly paywalled. If the explainability layer needs formal guideline
   references specifically (not just literature evidence), that's a
   separate retrieval source to add to `rag.py`, following its existing
   `retrieve_structured()` pattern.

---

## 8. Quick reference

```bash
# Setup
python -m venv .venv && source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY, GENAI_API_KEY, FAST_MODEL

# Get data
python prepare_data.py --dataset medqa   # or pubmedqa/ddxplus/symcat/medbullets/pmcvqa/pathvqa

# Run one dataset
python main.py --dataset medqa --model openai/gpt-oss-120b --difficulty adaptive --num_samples 20

# Run everything
N=10 NUM_ROUNDS=2 NUM_TURNS=3 bash run_benchmark.sh

# Check calibration
python evaluate_calibration.py output/*.json

# Interactive single-case demo
streamlit run demo_app.py
```

Questions this document doesn't answer should be answerable from
`ARCHITECTURE.md` (pipeline internals) or `README.md` (setup/usage); if
neither covers it, it's genuinely undecided and worth raising with the rest
of the team rather than guessing.
