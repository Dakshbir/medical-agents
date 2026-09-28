"""
Interactive single-case MDAgents demo (mirrors the mdagents2024.github.io/demo.html
flow: complexity check -> recruitment -> collaboration -> final decision) built on
top of the same Agent class / prompts used by the batch pipeline in utils.py.

Run with: streamlit run demo_app.py
"""
import os
import re
import random

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from utils import (
    Agent, _normalize_llm_list_text, FAST_MODEL, MAX_RESPONSE_TOKENS,
    _extract_answer_letter, consensus_confidence, get_rag_evidence_and_citations,
)

if os.environ.get('GENAI_API_KEY'):
    genai.configure(api_key=os.environ['GENAI_API_KEY'])

st.set_page_config(page_title="MDAgents Demo", layout="wide")

MODEL_OPTIONS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

PRESET_CASES = {
    "Case 1: GERD medication (likely Low)": {
        "question": (
            "A 54-year-old patient with worsening GERD symptoms despite ranitidine "
            "therapy is switched to a new medication. She reports epigastric burning "
            "pain after eating that has improved significantly since starting the new "
            "drug. What is the most likely mechanism of action of this new medication?"
        ),
        "options": {
            "A": "Irreversible inhibition of the H+/K+-ATPase pump",
            "B": "Histamine H2 receptor antagonism",
            "C": "Muscarinic receptor antagonism",
            "D": "Neutralization of gastric acid",
            "E": "Enhancement of lower esophageal sphincter tone",
        },
    },
    "Case 2: Headache & brain lesion (likely Moderate)": {
        "question": (
            "A 52-year-old woman presents with a complaint of headache for the past "
            "10 days. Her headache is diffuse, dull in character, moderate in "
            "intensity, and is worse in the morning. It is not associated with fever "
            "and sensitivity to light or sound. She experiences occasional nausea but "
            "no vomiting. She did not have similar headaches in the past. Her blood "
            "pressure is 140/90 mm Hg; pulse, 60/min, and body mass index is 33.5 "
            "kg/m2. The neurological examination reveals normal extraocular "
            "movements. Mild bilateral papilledema is present. A magnetic resonance "
            "imaging of the brain reveals a solitary lesion in the left temporal "
            "region with predominant hemorrhage. Which of the following types of "
            "cancer has the highest tendency to cause this brain lesion?"
        ),
        "options": {
            "A": "Breast cancer",
            "B": "Lung cancer",
            "C": "Melanoma",
            "D": "Multiple myeloma",
            "E": "Thyroid cancer",
        },
    },
    "Case 3: Muscle fatigue & vision problems (likely High)": {
        "question": (
            "A 20-year-old male presents with progressive muscle fatigue and "
            "double vision that worsens throughout the day and improves with rest. "
            "Examination reveals bilateral ptosis, diplopia on lateral gaze, mild "
            "dysarthria, and dysphagia. He denies fever, trauma, or recent travel. "
            "There is no family history of neurological disease. Which of the "
            "following is the most likely diagnosis?"
        ),
        "options": {
            "A": "Myasthenia gravis",
            "B": "Guillain-Barre syndrome",
            "C": "Pulmonary embolism",
            "D": "Myocarditis",
            "E": "Acute dystonic reaction",
            "F": "Pulmonary edema / fluid overload",
            "G": "Atrial fibrillation / flutter",
            "H": "Anemia",
            "I": "Anaphylaxis",
            "J": "Chagas disease",
        },
    },
    "Custom case": {"question": "", "options": {}},
}

COMPLEXITY_LABEL = {"basic": "Low", "intermediate": "Moderate", "advanced": "High"}
COMPLEXITY_COLOR = {"basic": "#2563eb", "intermediate": "#d97706", "advanced": "#dc2626"}

EXPERT_EMOJI = ['\U0001F468‍⚕️', '\U0001F468\U0001F3FB‍⚕️',
                '\U0001F469\U0001F3FC‍⚕️', '\U0001F469\U0001F3FB‍⚕️',
                '\U0001f9d1‍⚕️', '\U0001f9d1\U0001f3ff‍⚕️']


def format_question(question, options):
    if not options:
        return question
    opts = " ".join(f"({k}) {v}" for k, v in options.items())
    return f"{question} Options: {opts}"


def classify_complexity(question, model):
    instruction = ("You are a medical expert who conducts initial assessment and "
                   "your job is to decide the difficulty/complexity of the medical query.")
    agent = Agent(instruction=instruction, role='medical expert', model_info=model)
    agent.chat(instruction)
    prompt = (f"Now, given the medical query as below, you need to decide the "
              f"difficulty/complexity of it:\n{question}.\n\n"
              "Please indicate the difficulty/complexity of the medical query among "
              "below options, and briefly explain your reasoning in 2-4 sentences "
              "before giving your answer:\n"
              "1) basic: a single medical agent can output an answer.\n"
              "2) intermediate: a number of medical experts with different "
              "expertise should discuss and make the final decision.\n"
              "3) advanced: multiple teams of clinicians from different departments "
              "need to collaborate with each other to make the final decision.\n\n"
              "End your response with a line in exactly this format (no extra "
              "words on that line):\nFINAL COMPLEXITY: basic|intermediate|advanced")
    response = agent.chat(prompt)
    # A plain substring search over the whole response is unreliable: the
    # reasoning text itself restates the option list ("1) basic: ..."), so
    # naively checking for 'basic' anywhere matches before the model ever
    # states its actual verdict. Look for the explicit marker instead, and
    # only fall back to scanning the tail of the response (bias toward the
    # conclusion, not the restated menu) if the model didn't follow it.
    match = re.search(r'final complexity:\s*(basic|intermediate|advanced)', response, re.IGNORECASE)
    if match:
        level = match.group(1).lower()
    else:
        tail = response[-200:].lower()
        if 'advanced' in tail:
            level = 'advanced'
        elif 'intermediate' in tail:
            level = 'intermediate'
        else:
            level = 'basic'
    return level, response


def run_low(question, model, use_rag=True):
    evidence_text, citations = get_rag_evidence_and_citations(question, use_rag, k=3)
    evidence_block = f"\n{evidence_text}\n" if evidence_text else ""
    agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.',
                  role='Primary Care Clinician', model_info=model)
    agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
    answer = agent.chat(f"Let's think step by step.\n{evidence_block}\n**Question:** {question}\nAnswer: ")
    return answer, citations


def recruit_experts(question, model, n=3):
    recruit_prompt = ("You are an experienced medical expert who recruits a group "
                       "of experts with diverse identity to discuss and solve a "
                       "given medical query.")
    agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=FAST_MODEL)
    agent.chat(recruit_prompt)
    response = agent.chat(
        f"Question: {question}\nYou can recruit {n} experts in different medical "
        f"expertise to answer this question. Return your answer as exactly {n} "
        "lines, one per expert, in this plain-text format (no markdown, no bullet "
        "points, no bold):\nRole - one sentence description of their expertise\n\n"
        "Do not include any other text."
    )
    response = _normalize_llm_list_text(response)
    experts = []
    for line in response.split('\n'):
        line = line.strip().lstrip('0123456789.) ').strip()
        # Split on ' - ' (hyphen surrounded by spaces), not bare '-': plenty
        # of real specialties are themselves hyphenated compounds (e.g.
        # "Neuro-oncologist"), and splitting on the first bare '-' mangles
        # those into "Neuro" + "oncologist - ...".
        if not line or ' - ' not in line:
            continue
        role, desc = line.split(' - ', 1)
        experts.append({"role": role.strip(), "description": desc.strip()})
        if len(experts) == n:
            break
    if not experts:
        experts = [{"role": "General Physician", "description": "Provides a general medical assessment."}]
    return experts


def run_moderate(question, model, experts, rounds=2, use_rag=True):
    evidence_text, citations = get_rag_evidence_and_citations(question, use_rag, k=3)
    evidence_block = f"\n{evidence_text}\n" if evidence_text else ""

    agents = []
    for e in experts:
        inst = f"You are a {e['role']} who {e['description']}. Your job is to collaborate with other medical experts in a team."
        a = Agent(instruction=inst, role=e['role'], model_info=model)
        a.chat(inst)
        agents.append(a)

    opinions = {}
    for a in agents:
        opinions[a.role] = a.chat(
            f"Please return your initial assessment and answer to the medical "
            f"query among the options provided.\n{evidence_block}\nQuestion: {question}\n\nAnswer: "
        )

    log = []
    roles = [a.role for a in agents]
    for r in range(1, rounds + 1):
        for i, a in enumerate(agents):
            others = [x for x in roles if x != a.role]
            target = random.choice(others) if others else a.role
            transcript = "\n".join(f"[{k}]: {v}" for k, v in opinions.items())
            msg = a.chat(
                f"Here is the discussion so far:\n{transcript}\n\n"
                f"Remind your expertise as a {a.role}, then send a short (2-3 "
                f"sentence) message to {target} either agreeing, disagreeing, or "
                "asking a clarifying question about the case. Address them by name."
            )
            log.append({"Round": r, "From": a.role, "To": target, "Message": msg})
            opinions[a.role] = msg

    moderator = Agent("You are a final medical decision maker who reviews all "
                       "opinions from different medical experts and makes the "
                       "final decision.", "Moderator", model_info=model)
    moderator.chat("You are a final medical decision maker who reviews all "
                    "opinions from different medical experts and makes the "
                    "final decision.")
    transcript = "\n".join(f"[{row['From']} -> {row['To']}]: {row['Message']}" for row in log)
    final = moderator.chat(
        f"Here is the full discussion among the recruited experts:\n{transcript}\n\n"
        f"Question: {question}\n\nGiven the discussion, provide the final answer "
        "and a short justification."
    )

    confidence_info = consensus_confidence(list(opinions.values()))
    winner_letter = confidence_info.get("winner")
    attribution = [
        {
            "role": role,
            "opinion": opinion,
            "agreed_with_final": (_extract_answer_letter(opinion) == winner_letter) if winner_letter else None,
        }
        for role, opinion in opinions.items()
    ]
    return agents, log, final, confidence_info, attribution, citations


def recruit_teams(question, model, n_teams=3, n_members=2):
    recruit_prompt = ("You are an experienced medical expert. Given the complex "
                       "medical query, you organize Multidisciplinary Teams (MDTs) "
                       "to make an accurate and robust decision.")
    agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=FAST_MODEL)
    agent.chat(recruit_prompt)
    response = agent.chat(
        f"Question: {question}\n\nOrganize {n_teams} teams with {n_members} "
        "clinicians each, including an Initial Assessment Team and a Final Review "
        "and Decision Team. Return plain text only (no markdown tables, no bold), "
        "in this exact format:\nTeam: <team name>\nMember: <role> - <one sentence "
        "description>\nMember: <role> - <one sentence description>\n\n"
        "(blank line between teams, no other text)"
    )
    response = _normalize_llm_list_text(response)
    teams = []
    current = None
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith('team'):
            if current:
                teams.append(current)
            current = {"name": line.split(':', 1)[-1].strip(), "members": []}
        elif line.lower().startswith('member') and current is not None:
            body = line.split(':', 1)[-1].strip()
            if ' - ' in body:
                role, desc = body.split(' - ', 1)
                current["members"].append({"role": role.strip(), "description": desc.strip()})
    if current:
        teams.append(current)
    return teams[:n_teams] if teams else [{"name": "Initial Assessment Team", "members": [{"role": "General Physician", "description": "Provides a general medical assessment."}]}]


def run_high(question, model, teams, use_rag=True):
    evidence_text, citations = get_rag_evidence_and_citations(question, use_rag, k=3)
    evidence_block = f"\nRelevant medical literature:\n{evidence_text}\n" if evidence_text else ""

    reports = []
    for team in teams:
        member_agents = []
        for m in team['members']:
            inst = f"You are a {m['role']} who {m['description']}. You work on a multidisciplinary team."
            a = Agent(instruction=inst, role=m['role'], model_info=model)
            a.chat(inst)
            member_agents.append(a)

        findings = []
        for a in member_agents:
            resp = a.chat(f"As part of the '{team['name']}', review this case and give your findings in 2-3 sentences.{evidence_block}\n\nQuestion: {question}")
            findings.append(f"[{a.role}]: {resp}")

        lead = member_agents[0]
        report = lead.chat(
            "Here are your team's findings:\n" + "\n".join(findings) +
            f"\n\nSynthesize these into a short team report (3-4 sentences) for the '{team['name']}'."
        )
        reports.append({"team": team['name'], "report": report})

    decision_maker = Agent("You are an experienced medical expert who reviews "
                            "investigations from multiple teams and makes the "
                            "final decision.", "decision maker", model_info=model)
    decision_maker.chat("You are an experienced medical expert who reviews "
                         "investigations from multiple teams and makes the "
                         "final decision.")
    transcript = "\n\n".join(f"{r['team']}:\n{r['report']}" for r in reports)
    final = decision_maker.chat(
        f"Team reports:\n{transcript}\n\nQuestion: {question}\n\n"
        "Given these reports, provide the final answer and a short justification."
    )

    confidence_info = consensus_confidence([r["report"] for r in reports])
    winner_letter = confidence_info.get("winner")
    attribution = [
        {
            "role": r["team"],
            "opinion": r["report"],
            "agreed_with_final": (_extract_answer_letter(r["report"]) == winner_letter) if winner_letter else None,
        }
        for r in reports
    ]
    return reports, final, confidence_info, attribution, citations


# ---------------------------------------------------------------- UI ------

st.title("MDAgents — Interactive Case Demo")
st.caption("Adaptive collaboration of LLM agents for medical decision-making")

def render_confidence(confidence_info):
    conf = confidence_info.get("confidence")
    if conf is None:
        st.caption("Confidence: not available (no extractable answer to compare)")
        return
    pct = round(conf * 100)
    color = "#16a34a" if conf >= 0.8 else "#d97706" if conf >= 0.5 else "#dc2626"
    votes = confidence_info.get("votes", {})
    votes_str = ", ".join(f"{k}: {v}" for k, v in sorted(votes.items()))
    st.markdown(
        f"<div style='margin:8px 0'><span style='background:{color};color:white;"
        f"padding:4px 10px;border-radius:6px;font-weight:600'>Confidence: {pct}%</span> "
        f"<span style='color:#666'>({confidence_info.get('n')} opinions — {votes_str})</span></div>",
        unsafe_allow_html=True,
    )


def render_evidence(citations):
    if not citations:
        return
    with st.expander(f"Evidence used ({len(citations)} source(s))", expanded=False):
        for c in citations:
            label = c.get("title") or c.get("id") or c.get("source")
            if c.get("url"):
                st.markdown(f"- [{label}]({c['url']}) — *{c.get('source')}*")
            else:
                st.markdown(f"- {label} — *{c.get('source')}*")
            if c.get("snippet"):
                st.caption(c["snippet"][:300])


def render_attribution(attribution):
    if not attribution:
        return
    for a in attribution:
        agree = a.get("agreed_with_final")
        mark = "✅" if agree else ("❌" if agree is False else "•")
        with st.expander(f"{mark} {a['role']}", expanded=False):
            st.write(a["opinion"])


with st.sidebar:
    st.header("Setup")
    model = st.selectbox("Model", MODEL_OPTIONS, index=0)
    use_rag = st.checkbox("Use RAG evidence (PubMed/Wikipedia)", value=True)
    case_name = st.selectbox("Case", list(PRESET_CASES.keys()))
    preset = PRESET_CASES[case_name]

    question_text = st.text_area("Question", value=preset["question"], height=160)
    st.caption("Options (edit as `KEY: text`, one per line; leave blank for none)")
    options_text = st.text_area(
        "Options",
        value="\n".join(f"{k}: {v}" for k, v in preset["options"].items()),
        height=140,
    )
    run_clicked = st.button("Run Analysis", type="primary")

if run_clicked:
    options = {}
    for line in options_text.split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            options[k.strip()] = v.strip()

    full_question = format_question(question_text.strip(), options)

    if not question_text.strip():
        st.error("Please enter a question.")
        st.stop()

    st.markdown("## Step 1: Medical Complexity Classification")
    with st.spinner("Classifying complexity..."):
        level, reasoning = classify_complexity(full_question, model)
    color = COMPLEXITY_COLOR[level]
    st.markdown(
        f"<span style='background:{color};color:white;padding:4px 10px;"
        f"border-radius:6px;font-weight:600'>{COMPLEXITY_LABEL[level]} Complexity</span>",
        unsafe_allow_html=True,
    )
    st.info(reasoning)

    if level == 'basic':
        st.markdown("## Step 2: Primary Care Clinician")
        with st.spinner("Consulting Primary Care Clinician..."):
            answer, citations = run_low(full_question, model, use_rag=use_rag)
        st.write(answer)
        render_evidence(citations)
        st.markdown("## Step 3: Final Decision")
        st.success(answer)
        st.caption("Confidence: not computed for the single-agent path (no independent second opinion to compare against).")

    elif level == 'intermediate':
        st.markdown("## Step 2: Recruitment")
        with st.spinner("Recruiting experts..."):
            experts = recruit_experts(full_question, model, n=3)
        cols = st.columns(len(experts))
        for c, e, emoji in zip(cols, experts, EXPERT_EMOJI):
            with c:
                st.markdown(f"### {emoji} {e['role']}")
                st.caption(e['description'])

        st.markdown("## Step 3: Collaboration")
        with st.spinner("Experts are discussing the case..."):
            agents, log, final, confidence_info, attribution, citations = run_moderate(
                full_question, model, experts, rounds=2, use_rag=use_rag)
        st.table(log)

        st.markdown("## Step 4: Final Decision")
        st.success(final)
        render_confidence(confidence_info)
        render_evidence(citations)
        st.markdown("#### Specialist attribution")
        render_attribution(attribution)

    else:
        st.markdown("## Step 2: Recruitment")
        with st.spinner("Organizing multidisciplinary teams..."):
            teams = recruit_teams(full_question, model, n_teams=3, n_members=2)
        for team in teams:
            with st.expander(f"{team['name']} ({len(team['members'])} members)", expanded=True):
                for m in team['members']:
                    st.markdown(f"**{m['role']}** — {m['description']}")

        st.markdown("## Step 3: Team Reports")
        with st.spinner("Teams are investigating..."):
            reports, final, confidence_info, attribution, citations = run_high(
                full_question, model, teams, use_rag=use_rag)
        for r in reports:
            with st.expander(r['team'], expanded=True):
                st.write(r['report'])

        st.markdown("## Step 4: Final Decision")
        st.success(final)
        render_confidence(confidence_info)
        render_evidence(citations)
        st.markdown("#### Team attribution")
        render_attribution(attribution)
else:
    st.info("Pick a case (or write your own) in the sidebar, then click **Run Analysis**.")
