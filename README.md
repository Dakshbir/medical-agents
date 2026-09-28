# [NeurIPS 2024 Oral] MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making

<p align="center">
   📖 <a href="https://arxiv.org/abs/2404.15155" target="_blank">Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;🤖 <a href="https://mdagents2024.github.io/" target="_blank">Project Page</a>
</p>



Foundation models are becoming invaluable tools in medicine. Despite their promise, the strategic deployment of Large Language Models (LLMs) for effective utility in complex medical tasks remains an open question. We introduce a novel framework, <ins><b>M</b></ins>edical <ins><b>D</b></ins>ecision-making <ins><b>Agents</b></ins> (**MDAgents**) which aims to address this gap by automatically assigning a collaboration structure for a team of LLMs. The assigned solo or group collaboration structure is tailored to the medical task at hand, a simple emulation of how real-world medical decision-making processes adapt to tasks of different complexities. We evaluate our framework and baseline methods with state-of-the-art LLMs across a suite of medical benchmarks containing real-world medical knowledge and challenging clinical diagnosis. MDAgents achieved the best performance in **seven out of ten** benchmarks on the tasks that require an understanding of medical knowledge and multi-modal reasoning, showing a significant improvement of up to **11.8\%** compared to previous multi-agent setting (p < 0.05). Ablation studies reveal that our MDAgents effectively determines medical complexity to optimize for *efficiency* and *accuracy* across diverse medical tasks. We also explore the dynamics of group consensus, offering insights into how collaborative agents could behave in complex clinical team dynamics.

<p align="center">
   <img width="800" alt="image" src="imgs/animation.gif">
   <img width="800" alt="image" src="imgs/case_study.png">
</p>

<br>
<br>

> **This fork.** This repository is a local, from-scratch replication and
> extension of the original MDAgents implementation, built to run against
> Groq (free tier) and Gemini instead of OpenAI, with real bug fixes, a
> lightweight RAG stage, the paper's moderator-feedback loop, support for 7
> of the paper's 10 datasets, and two additions beyond the paper: a
> calibrated confidence score and a structured, evidence-cited explanation
> object attached to every answer. See [ARCHITECTURE.md](ARCHITECTURE.md)
> for exactly what runs at each pipeline stage, what differs from the paper,
> and known limitations. **If you're picking this project up to integrate a
> new component (e.g. a digital twin module), start with
> [HANDOFF.md](HANDOFF.md) instead of this file.**

## Quick Start

Create a virtual environment:

```bash
~$ python -m venv .venv
~$ source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
```

Install the required packages:
```bash
~$ pip install -r requirements.txt
```

Set up API keys — copy `.env.example` to `.env` and fill in the keys you
have:
```bash
~$ cp .env.example .env
```
```
GROQ_API_KEY=...      # used for every --model that isn't a gemini model
GENAI_API_KEY=...     # used when --model contains "gemini" (required for image/video questions)
FAST_MODEL=openai/gpt-oss-20b   # cheap/fast model for recruiter & difficulty-check calls
```

Fetch a small real sample of a dataset (no manual downloads needed — this
hits public APIs directly):
```bash
~$ python prepare_data.py --dataset medqa       # or pubmedqa, ddxplus, symcat, medbullets, pmcvqa, pathvqa
```
This writes `data/<dataset>/{test,train}.jsonl` (plus `data/<dataset>/images/`
for the two image datasets).

<br>

## Dataset

<p align="center">
  <img width="900" src="imgs/datasets.png">
</p>

<br>

Of the paper's 10 datasets, **7 are implemented** here (fetchable via
`prepare_data.py`); 3 are out of scope for the reasons below.

1) MedQA — `GBaker/MedQA-USMLE-4-options` on Hugging Face
2) PubMedQA — `qiaojin/PubMedQA` (`pqa_labeled` config)
3) DDXPlus — `aai530-group6/ddxplus` on Hugging Face
4) SymCat — public GitHub mirror of a symcat.com scrape ([details/caveats in ARCHITECTURE.md](ARCHITECTURE.md))
5) Medbullets — `HanjieChen/ChallengeClinicalQA` (official repo)
6) PMC-VQA — `RadGenome/PMC-VQA` on Hugging Face (image + text)
7) Path-VQA — `flaviagiammarino/path-vqa` on Hugging Face (image + text, yes/no subset only, matching the paper)

**Not implemented:**
- **JAMA Clinical Challenge** — the dataset's own source repo explicitly
  does not redistribute this data due to JAMA's license; no legitimate
  public mirror exists.
- **MIMIC-CXR** — requires a credentialed PhysioNet account (signed data-use
  agreement); out of scope without a user who already holds that access.
- **MedVidQA** — video processing is significant additional scope, deferred
  for now.

Original dataset sources, for reference:
[MedQA](https://github.com/jind11/MedQA) ·
[MedMCQA](https://github.com/medmcqa/medmcqa) ·
[PubMedQA](https://github.com/pubmedqa/pubmedqa) ·
[DDXPlus](https://github.com/mila-iqia/ddxplus) ·
[SymCat](https://github.com/teliov/SymCat-to-synthea) ·
[PMC-VQA](https://github.com/xiaoman-zhang/PMC-VQA) ·
[Path-VQA](https://github.com/UCSD-AI4H/PathVQA) ·
[MIMIC-CXR](https://github.com/baeseongsu/mimic-cxr-vqa) ·
[MedVidQA](https://github.com/deepaknlp/MedVidQACL)

<br>

## Comparison to Previous Single Agent/Multi-Agent Methods

<p align="center">
  <img width="900" src="imgs/comparison.png">
</p>

<br>

## Inference

```bash
~$ python main.py --model openai/gpt-oss-120b --dataset medqa --difficulty adaptive --num_samples 20
```

`--model` is any Groq model id (e.g. `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`qwen/qwen3.8-27b`) or any Gemini model id containing `gemini` (required for
`pmcvqa`/`pathvqa`, which need image input). `--dataset` is one of
`medqa, pubmedqa, ddxplus, symcat, medbullets, pmcvqa, pathvqa`.
`--difficulty` is `adaptive` (the paper's default — an LLM classifies each
question's complexity) or a fixed `basic`/`intermediate`/`advanced` for
ablations. Add `--no-use_rag` or `--no-use_moderator_review` to turn off
those stages for ablation comparisons (both default on). `--num_rounds`/
`--num_turns` (default 5/5, matching the paper) cap the moderate-path debate
— lower them (e.g. `2`/`3`) for cost-bounded runs, since agents will keep
"wanting to talk" to affirm agreement, not just to resolve disagreement.
Set `ENSEMBLE_TEMPERATURES="0.0,0.5,1.0"` (env var) to sample the final
decision at multiple temperatures and take a real majority vote, matching
the paper's temperature-ensemble technique (default is a single sample).

Every result now also carries a `confidence` score (agreement across
independent opinions) and a structured `explanation` (reasoning, cited
evidence, per-specialist attribution) — see
[ARCHITECTURE.md](ARCHITECTURE.md#confidence-and-structured-explainability).
Results — including per-question predictions, confidence, and the full
explanation — are written incrementally (after every question, not just at
the end) to `output/<model>_<dataset>_<difficulty>.json`, so a crash never
loses already-completed work.

To run the same evaluation across several datasets in one go:
```bash
~$ N=10 NUM_ROUNDS=2 NUM_TURNS=3 bash run_benchmark.sh
```
Then check how well-calibrated the confidence scores actually are (does "80%
confident" mean "correct ~80% of the time"?):
```bash
~$ python evaluate_calibration.py output/*.json
```

<br>

## Interactive Demo

For a single-case, step-by-step walkthrough (complexity check → recruitment
→ collaboration → final decision) instead of a batch benchmark run:

```bash
~$ streamlit run demo_app.py
```

Opens at `http://localhost:8501`. Pick one of the built-in cases or write
your own, choose a model, and click **Run Analysis**. Each step of the
pipeline is rendered live, ending with a confidence badge, cited evidence
(with links), and a per-specialist/per-team agree/disagree breakdown.

<br>

## Main Results

<p align="center">
  <img width="800" alt="image" src="imgs/main_table.png">
  <img width="500" alt="image" src="imgs/radar.png">
</p>
<br>

## Ablation 1: Impact of Complexity Selection

<p align="center">
  <img width="800" alt="image" src="imgs/ablation1.png">
</p>
<br>

## Ablation 2: Impact of Number of Agents and Temperatures in Group Setting

<p align="center">
  <img width="800" alt="image" src="imgs/ablation2.png">
</p>
<br>

## Ablation 3: Impact of Moderator’s Review and RAG

<p align="center">
  <img width="600" alt="image" src="imgs/ablation4.png">
</p>
<br>



## Cite Us
If you find this repository useful in your research, please cite our works:

```bibtex
@misc{kim2024mdagentsadaptivecollaborationllms,
      title={MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making}, 
      author={Yubin Kim and Chanwoo Park and Hyewon Jeong and Yik Siu Chan and Xuhai Xu and Daniel McDuff and Hyeonhoon Lee and Marzyeh Ghassemi and Cynthia Breazeal and Hae Won Park},
      year={2024},
      eprint={2404.15155},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.15155}, 
}
```

```bibtex
@misc{kim2024demonstrationadaptivecollaborationlarge,
      title={A Demonstration of Adaptive Collaboration of Large Language Models for Medical Decision-Making}, 
      author={Yubin Kim and Chanwoo Park and Hyewon Jeong and Cristina Grau-Vilchez and Yik Siu Chan and Xuhai Xu and Daniel McDuff and Hyeonhoon Lee and Marzyeh Ghassemi and Cynthia Breazeal and Hae Won Park},
      year={2024},
      eprint={2411.00248},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.00248}, 
}
```

## Contact
Yubin Kim (ybkim95@mit.edu)
