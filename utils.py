import os
import re
import json
import math
import time
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from tqdm import tqdm
from prettytable import PrettyTable
from termcolor import cprint
from pptree import Node
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from pptree import *
from dotenv import load_dotenv

load_dotenv()

# Used internally for cheap/fast meta-decisions (difficulty check, recruiter,
# report summarization) so those calls don't have to burn the main --model.
FAST_MODEL = os.environ.get('FAST_MODEL', 'openai/gpt-oss-20b')

# The original code never capped response length. That's harmless against
# OpenAI's much higher-throughput tiers, but reasoning models like gpt-oss
# can produce responses of several thousand tokens, which - combined with
# accumulated conversation history - blows straight through Groq's free-tier
# 8000 TPM ceiling within 2-3 chained calls to the same agent. Capping this
# bounds worst-case request size deterministically.
MAX_RESPONSE_TOKENS = int(os.environ.get('MAX_RESPONSE_TOKENS', 700))

# Neither the original code nor the Gemini SDK's own defaults put a ceiling
# on how long a single call can block. Without this, a stalled network call
# (large image upload, transient backend hang) blocks forever - the retry
# loop below never even gets a chance to retry because the first attempt
# never returns.
GEMINI_TIMEOUT_SECONDS = int(os.environ.get('GEMINI_TIMEOUT_SECONDS', 60))

# If Groq is rate-limited/unreachable, fall back to Gemini for that call
# rather than crashing the whole run (best-effort; stateless per call).
FALLBACK_GEMINI_MODEL = os.environ.get('FALLBACK_GEMINI_MODEL', 'gemini-2.5-flash')

# The paper's decision-making stage uses a "temperature ensemble" (sample
# the final decision at several temperatures, take the majority answer).
# The original code called this method `temp_responses` and even labeled
# its output {'majority': ...}, but only ever sampled temperature=0.0 - no
# actual ensembling or voting happened anywhere. Default stays a single
# sample (same cost as before); set ENSEMBLE_TEMPERATURES="0.0,0.5,1.0" to
# turn on real ensembling for paper-faithful runs.
ENSEMBLE_TEMPERATURES = [float(t) for t in os.environ.get('ENSEMBLE_TEMPERATURES', '0.0').split(',') if t.strip()]


def _gemini_fallback_generate(messages, original_error=None):
    """original_error is the Groq exception that triggered this fallback.
    If Gemini ALSO fails, that original error must not be lost: the earlier
    version of this function let the fallback's own exception propagate
    alone, which completely masked WHY the primary call failed in the first
    place. Debugging a batch of "ResourceExhausted: gemini-2.5-flash quota"
    errors was actively misleading when the real root cause was a Groq-side
    429/413 that never got logged anywhere."""
    try:
        genai.configure(api_key=os.environ['GENAI_API_KEY'])
        model = genai.GenerativeModel(FALLBACK_GEMINI_MODEL)
        prompt = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
        response = model.generate_content(prompt)
        return response.text
    except Exception as fallback_error:
        if original_error is not None:
            raise RuntimeError(
                f"Both the primary call and the Gemini fallback failed. "
                f"Primary error: {type(original_error).__name__}: {original_error}. "
                f"Fallback error: {type(fallback_error).__name__}: {fallback_error}"
            ) from fallback_error
        raise


def _is_gemini(model_info):
    return 'gemini' in model_info.lower()


def _load_image(img_path):
    from PIL import Image
    return Image.open(img_path)


# Errors that will never succeed on retry (bad/deprecated model name,
# malformed request, bad API key) - the original code retried these 10
# times identically to a transient rate limit, then swallowed the real
# reason into a generic "Error: Failed to get response from Gemini."
# string. That string would then flow downstream as if it were a real
# model answer (silently parsed as "no answer found" instead of surfacing
# what actually broke) - exactly the kind of hidden failure worth failing
# loudly on instead.
_PERMANENT_GEMINI_ERRORS = (
    google_exceptions.NotFound,
    google_exceptions.InvalidArgument,
    google_exceptions.PermissionDenied,
    google_exceptions.Unauthenticated,
    # A per-minute rate limit would be worth retrying, but Gemini's free
    # tier also enforces a per-DAY quota (observed: 20 requests/day for
    # gemini-2.5-flash) via the same ResourceExhausted exception. Retrying
    # that 5x with exponential backoff (~5+ minutes) is pointless - it
    # cannot possibly succeed again for hours. Treat it as permanent so it
    # fails in seconds with a clear message instead of slowly wasting time
    # per call.
    google_exceptions.ResourceExhausted,
)


# A single, process-wide pool: Gemini calls are infrequent enough relative
# to their duration that reusing a small pool is fine, and it lets us
# enforce a hard wall-clock ceiling per call independent of whatever the
# SDK's own `request_options={'timeout': ...}` does internally. In practice
# that SDK-level timeout was observed NOT reliably bounding streaming
# ChatSession calls in this environment - a call that should time out at
# 60s instead hung for 8+ minutes with no error, no output, nothing to
# retry. Wrapping every attempt in a future with its own .result(timeout=)
# guarantees this code moves on regardless of what the underlying transport
# does; a truly stuck thread is abandoned (leaked, but harmless - it's a
# daemon-pool thread, not a blocker of process exit) rather than hanging
# the whole pipeline.
_GEMINI_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _call_gemini_with_retry(fn, model_info, max_retries=5, base_delay=3):
    last_error = None
    hard_timeout = GEMINI_TIMEOUT_SECONDS + 10
    for attempt in range(max_retries):
        try:
            future = _GEMINI_EXECUTOR.submit(fn)
            return future.result(timeout=hard_timeout)
        except _PERMANENT_GEMINI_ERRORS as e:
            raise RuntimeError(f"Gemini call failed (model={model_info}): {e}") from e
        except FutureTimeoutError:
            last_error = f"call did not return within {hard_timeout}s (abandoned a hung attempt)"
        except Exception as e:
            last_error = e
        if attempt < max_retries - 1:
            time.sleep(min(base_delay * (2 ** attempt), 30))
    raise RuntimeError(f"Gemini call failed after {max_retries} retries (model={model_info}): {last_error}")


# Best-effort import: degrade gracefully (no evidence injected) if rag.py
# isn't there or a lookup fails, since retrieval must never be able to take
# down the whole pipeline.
try:
    from rag import retrieve_structured as _rag_retrieve_structured, format_evidence as _rag_format_evidence
except ImportError:
    def _rag_retrieve_structured(query, k=3):
        return []

    def _rag_format_evidence(snippets):
        return ""


def get_rag_evidence_and_citations(query, use_rag, k=3):
    """Returns (evidence_text_for_prompt_injection, structured_citations).
    Both come from a single retrieval call (rag.py caches by (query, k)), so
    getting citations alongside the prompt text doesn't cost a second
    network round-trip. get_rag_evidence() below is a thin wrapper over this
    for the many call sites that only need the text."""
    if not use_rag:
        return "", []
    # `query` here is usually the fully-formatted "<vignette> Options: (A) ...
    # (B) ..." string built by create_question(). Retrieving on that whole
    # blob is noisy (the options add unrelated keywords, and its length can
    # trip up search APIs entirely) - strip back to just the clinical
    # question for retrieval purposes.
    query = re.split(r'\bOptions:\s*', query)[0]
    try:
        structured = _rag_retrieve_structured(query, k=k)
        text = _rag_format_evidence([item['display'] for item in structured])
        return text, structured
    except Exception:
        return "", []


def get_rag_evidence(query, use_rag, k=3):
    return get_rag_evidence_and_citations(query, use_rag, k)[0]


def _groq_client():
    return OpenAI(
        api_key=os.environ['GROQ_API_KEY'],
        base_url="https://api.groq.com/openai/v1",
    )


def _with_retry(fn, max_retries=6, base_delay=4):
    """Retries transient Groq/OpenAI-compatible errors (rate limits,
    connection blips, 5xx) with exponential backoff. The original code had
    no retry logic at all for this backend, so a single 429 (easy to hit
    given how many sequential agent calls MDAgents makes per question)
    would crash the whole run."""
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(base_delay * (2 ** attempt), 60))
        except (APIConnectionError, APIStatusError):
            if attempt == max_retries - 1:
                raise
            time.sleep(min(base_delay * (2 ** attempt), 30))


def _extract_answer_letter(text):
    """Pulls the option letter an agent settled on. Several prompts in this
    codebase (including the paper's own example: 'Answer: C) 2th pharyngeal
    arch') ask for 'X)' rather than '(X)' - a fully-parenthesized-only regex
    silently misses that format entirely. The opening paren is optional here
    to catch both.

    Anchored on the word 'Answer:' first (which is what every prompt in this
    codebase explicitly asks for), taking the LAST such anchored match - not
    because later mentions are more trustworthy in general, but because
    reasoning models often restate a "Final Answer: X" after their chain of
    thought, and that restatement should win over an earlier exploratory
    mention. Critically, this is anchored to "Answer:" specifically, NOT to
    any parenthesized letter anywhere in the text: an earlier version of this
    function scanned every '(X)' in the whole response, which broke exactly
    the case reasoning models produce most often - state "Answer: (A) ..."
    up front, then explain why (B), (C), (D) are each wrong, ending the
    response on a distractor's letter. That version returned 'D' for a
    response whose actual, explicitly-labeled answer was 'A'. Falls back to
    scanning any bare parenthetical only if no "Answer:" marker exists at
    all (e.g. a response that never used the requested format)."""
    answer_anchored = re.findall(r'Answer:?\s*\(?([A-Za-z])\)', text, re.IGNORECASE)
    if answer_anchored:
        return answer_anchored[-1].upper()
    matches = re.findall(r'\(?([A-Za-z])\)', text)
    return matches[-1].upper() if matches else None


def _extract_yes_no_maybe(text):
    """For PubMedQA-style questions, which are scored yes/no/maybe rather
    than by option letter. Takes the LAST match for the same reason
    _extract_answer_letter does: reasoning text often mentions the other
    possibilities before reaching its conclusion."""
    matches = re.findall(r'\b(yes|no|maybe)\b', text.lower())
    return matches[-1] if matches else None


def majority_vote(responses):
    """Given {temperature: response_text} from Agent.temp_responses, tallies
    the extracted answer letters and returns the response text matching the
    most common one (ties broken by lowest temperature). With a single
    sample (the default), this is just that sample's text - real majority
    voting only kicks in when ENSEMBLE_TEMPERATURES has multiple values."""
    letters = {temp: _extract_answer_letter(text) for temp, text in responses.items()}
    votes = [l for l in letters.values() if l]
    if not votes:
        return next(iter(responses.values()))
    winner = Counter(votes).most_common(1)[0][0]
    for temp in sorted(responses.keys()):
        if letters[temp] == winner:
            return responses[temp]
    return next(iter(responses.values()))


def consensus_confidence(texts):
    """Given raw response texts from multiple independent agents/samples
    answering the same question, extracts each one's answer letter and
    returns an agreement-based confidence estimate.

    'confidence' here is vote share: the fraction of extractable answers
    that match the majority answer (1.0 = everyone agreed). This is the
    simplest, most directly interpretable proxy - "4 of 5 experts agreed" -
    and is what gets calibrated against actual correctness in
    evaluate_calibration.py. 'entropy' (Shannon entropy over the answer
    distribution, normalized to [0,1] by the max possible entropy for that
    many samples) is also returned since it's the metric the original paper
    uses for its own consensus-dynamics analysis (Appendix B) - lower
    entropy is the same underlying signal as higher confidence, just on a
    different, less immediately-interpretable scale.

    Returns None fields (not zeros) when no answer could be extracted at
    all, so callers/evaluation code can distinguish "no data" from "zero
    confidence" rather than silently treating them the same."""
    letters = [_extract_answer_letter(t) for t in texts if t]
    letters = [l for l in letters if l]
    n = len(letters)
    if n == 0:
        return {"confidence": None, "entropy": None, "n": 0, "votes": {}, "winner": None}

    counts = Counter(letters)
    winner, winner_count = counts.most_common(1)[0]
    vote_share = winner_count / n

    if n > 1:
        probs = [c / n for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(n)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        normalized_entropy = 0.0

    return {
        "confidence": vote_share,
        "entropy": normalized_entropy,
        "n": n,
        "votes": dict(counts),
        "winner": winner,
    }


def build_explanation(answer, confidence_info, reasoning, evidence, attribution, complexity, consensus_reached=None):
    """Packages what the pipeline already computes into the structured
    schema the project's explainability layer needs downstream (a UI, or a
    future speech-synthesis stage) - reason + evidence + specialist
    attribution + confidence, as actual fields, not a wall of prose that has
    to be re-parsed. 'evidence' entries come from rag.py's structured
    citations (source/id/title/url/snippet), not just injected text, so
    citations can be rendered/spoken as "according to PubMed, PMID ...".

    Note on 'guideline reference': the retrieval backend here (PubMed +
    Wikipedia, see rag.py) surfaces medical literature, not curated
    clinical-guideline documents (NICE/ACC-AHA/WHO), which are mostly
    paywalled or not freely scrapable. This should be read as an evidence
    citation layer, not a claim of formal guideline linkage."""
    return {
        "answer": answer,
        "confidence": confidence_info.get("confidence"),
        "confidence_detail": confidence_info,
        "reasoning": reasoning,
        "evidence": evidence or [],
        "specialist_attribution": attribution or [],
        "complexity": complexity,
        "consensus_reached": consensus_reached,
    }


def _normalize_llm_list_text(text):
    """Normalizes formatting quirks (markdown bold, en/em dashes, narrow
    spaces) that different models use when asked to return a numbered list
    with '-'-separated fields, so the downstream naive string parsing
    (which expects literal ASCII '-' and '. ') doesn't break."""
    text = text.replace('**', '')
    text = re.sub(r'[–—−]', '-', text)
    text = text.replace(' ', ' ').replace('\xa0', ' ')
    return text


class Agent:
    def __init__(self, instruction, role, examplers=None, model_info=None, img_path=None):
        model_info = model_info or FAST_MODEL
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if _is_gemini(self.model_info):
            self.model = genai.GenerativeModel(self.model_info)
            self._chat = self.model.start_chat(history=[])
        else:
            self.client = _groq_client()
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message, img_path=None, chat_mode=True):
        if _is_gemini(self.model_info):
            content = [message, _load_image(img_path)] if img_path else message

            def _send():
                response = self._chat.send_message(content, stream=True, request_options={'timeout': GEMINI_TIMEOUT_SECONDS})
                responses = ""
                for chunk in response:
                    responses += chunk.text + "\n"
                return responses

            return _call_gemini_with_retry(_send, self.model_info)

        else:
            if img_path:
                raise ValueError(
                    f"Model '{self.model_info}' does not support image input. "
                    "Use a Gemini model (e.g. gemini-2.5-flash) for multimodal questions."
                )
            self.messages.append({"role": "user", "content": message})

            try:
                response = _with_retry(lambda: self.client.chat.completions.create(
                    model=self.model_info,
                    messages=self.messages,
                    max_tokens=MAX_RESPONSE_TOKENS,
                ))
                content = response.choices[0].message.content
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                content = _gemini_fallback_generate(self.messages, original_error=e)

            self.messages.append({"role": "assistant", "content": content})
            return content

    def temp_responses(self, message, img_path=None):
        if not _is_gemini(self.model_info):
            if img_path:
                raise ValueError(
                    f"Model '{self.model_info}' does not support image input. "
                    "Use a Gemini model (e.g. gemini-2.5-flash) for multimodal questions."
                )
            self.messages.append({"role": "user", "content": message})

            temperatures = ENSEMBLE_TEMPERATURES

            responses = {}
            for temperature in temperatures:
                try:
                    response = _with_retry(lambda: self.client.chat.completions.create(
                        model=self.model_info,
                        messages=self.messages,
                        temperature=temperature,
                        max_tokens=MAX_RESPONSE_TOKENS,
                    ))
                    responses[temperature] = response.choices[0].message.content
                except (RateLimitError, APIConnectionError, APIStatusError) as e:
                    responses[temperature] = _gemini_fallback_generate(self.messages, original_error=e)

            return responses

        else:
            content = [message, _load_image(img_path)] if img_path else message

            def _send():
                response = self._chat.send_message(content, stream=True, request_options={'timeout': GEMINI_TIMEOUT_SECONDS})
                responses = ""
                for chunk in response:
                    responses += chunk.text + "\n"
                return responses

            return _call_gemini_with_retry(_send, self.model_info)

class Group:
    def __init__(self, goal, members, question, examplers=None, model_info=None, evidence=""):
        self.goal = goal
        self.members = []
        model_info = model_info or FAST_MODEL
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=model_info)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers
        self.evidence = evidence

    def interact(self, comm_type, message=None, img_path=None, prior_reports=""):
        if comm_type == 'internal':
            if not self.members:
                return f"[No members could be parsed for the '{self.goal}' team from the recruiter's output; skipping this group's assessment.]"

            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]

            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)

            # The paper describes teams building on each other's work ("each
            # team builds upon the foundation laid by the previous"); the
            # original code ran every team fully independently, so e.g. the
            # Final Review and Decision Team never actually saw what earlier
            # teams found before producing its own report.
            prior_block = f"\n\nReports from earlier teams:\n{prior_reports}" if prior_reports else ""
            delivery_prompt += f"{prior_block}\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {self.question}"
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery), img_path=img_path)
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            evidence_block = f"\nRelevant medical literature:\n{self.evidence}\n" if self.evidence else ""
            exampler_block = f"\n\nExample cases for reference:\n{self.examplers}" if self.examplers is not None else ""

            # Paper's "Report Generation" prompt template (Appendix C.2): each
            # team produces a synthesized REPORT, not a direct answer to the
            # question. The final decision-maker (in process_advanced_query)
            # is the one who turns these team reports into an actual answer.
            # The original code had the lead directly "return your answer to
            # the medical query" here, which blurred team-report generation
            # with final decision-making a stage too early.
            investigation_prompt = (
                f"Here are your team's investigation findings:\n{gathered_investigation}"
                f"{evidence_block}{exampler_block}\n\nQuestion: {self.question}\n\n"
                "Please complete the following steps:\n"
                "1. Take careful and comprehensive consideration of the provided findings.\n"
                "2. Extract key knowledge from the findings.\n"
                "3. Derive a comprehensive and summarized analysis based on the extracted knowledge.\n"
                "4. Generate a refined and synthesized report for this team based on your analysis."
            )

            response = lead_member.chat(investigation_prompt, img_path=img_path)

            return response

        elif comm_type == 'external':
            return

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count % len(emojis)]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count % len(emojis)]), moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def setup_model(model_name):
    if _is_gemini(model_name):
        genai.configure(api_key=os.environ['GENAI_API_KEY'])
        return genai, None
    else:
        client = _groq_client()
        return None, client

def load_data(dataset):
    test_qa = []
    examplers = []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base_dir, 'data', dataset, 'test.jsonl')
    with open(test_path, 'r', encoding='utf-8') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = os.path.join(base_dir, 'data', dataset, 'train.jsonl')
    with open(train_path, 'r', encoding='utf-8') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset):
    """Builds the prompt-ready question text (and image path, if any) from a
    dataset row. Driven by which fields are present on the sample rather than
    by dataset name, so it works for any dataset whose loader follows the
    documented schemas in prepare_data.py without needing a new branch here
    per dataset. The original code only handled 'medqa' by name; every other
    dataset silently got the raw, unformatted question with no options."""
    img_path = sample.get('image_path')
    if img_path and not os.path.isabs(img_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_dir, 'data', dataset, img_path)

    if sample.get('options'):
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, img_path

    if 'context' in sample:  # PubMedQA-style: yes/no/maybe, scored via context
        question = f"Context: {sample['context']}\n\nQuestion: {sample['question']} Answer yes, no, or maybe."
        return question, img_path

    return sample['question'], img_path

def determine_difficulty(question, difficulty):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=FAST_MODEL)
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'

def process_basic_query(question, examplers, model, args, img_path=None):
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
    new_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            tmp_exampler = {}
            exampler_question = exampler['question']
            choices = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(choices)
            exampler_question += " " + ' '.join(choices)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            exampler_reason = medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            new_examplers.append(tmp_exampler)
    
    single_agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.', role='medical expert', examplers=new_examplers, model_info=model)
    single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')

    evidence_text, citations = get_rag_evidence_and_citations(question, getattr(args, 'use_rag', False))
    evidence_block = f"\n{evidence_text}\n" if evidence_text else ""
    _decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n{evidence_block}\n**Question:** {question}\nAnswer: ''', img_path=img_path)

    answer_text = majority_vote(_decision)
    confidence_info = consensus_confidence(list(_decision.values()))
    explanation = build_explanation(
        answer=_extract_answer_letter(answer_text),
        confidence_info=confidence_info,
        reasoning=answer_text,
        evidence=citations,
        attribution=[{"role": "Primary Care Clinician", "opinion": answer_text}],
        complexity="basic",
        consensus_reached=True,
    )
    return {'majority': _decision, 'answer': answer_text, 'explanation': explanation}

def process_intermediate_query(question, examplers, model, args, img_path=None):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=FAST_MODEL)
    tmp_agent.chat(recruit_prompt)
    
    num_agents = 5  # You can adjust this number as needed
    recruited = tmp_agent.chat(f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")
    recruited = _normalize_llm_list_text(recruited)

    numbered_lines = [l for l in recruited.split('\n') if re.match(r'^\s*\d+[.)]', l)]
    if not numbered_lines:
        numbered_lines = [l for l in recruited.split('\n') if l.strip()]
    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in numbered_lines[:num_agents]]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
            agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
        except:
            agent_list += f"Agent {i+1}: {agent[0].strip().lower()}\n"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except:
            continue
        
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
        
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    fewshot_examplers = ""
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            exampler_question = f"[Example {ie+1}]\n" + exampler['question']
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)
            exampler_question += " " + " ".join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = tmp_agent.chat(f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")
            
            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    # These were hardcoded to 5/5 with no way to override. That's expensive
    # in practice: agents will happily keep "wanting to talk" purely to
    # affirm agreement (not just to resolve disagreement), so 5 agents x up
    # to 5 turns x up to 5 rounds can spiral into 100+ LLM calls for a
    # single question even when everyone already agrees on round 1. Made
    # configurable so a benchmark run can bound cost/time without editing
    # code; defaults kept at the original 5/5 for anyone not overriding.
    num_rounds = getattr(args, 'num_rounds', 5)
    num_turns = getattr(args, 'num_turns', 5)
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}

    evidence_text, citations = get_rag_evidence_and_citations(question, getattr(args, 'use_rag', False))
    evidence_block = f"\n{evidence_text}\n" if evidence_text else ""

    initial_report = ""
    for k, v in agent_dict.items():
        opinion = v.chat(f'''Given the examplers, please return your answer to the medical query among the option provided.\n\n{fewshot_examplers}\n{evidence_block}\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=img_path)
        initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[1][k.lower()] = opinion

    consensus = False
    interaction_history = f"Initial opinions:\n{initial_report}\n"
    use_moderator_review = getattr(args, 'use_moderator_review', True)

    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"

        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
                participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))
                
                if 'yes' in participate.lower().strip():                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for ce in chosen_experts:
                        specific_question = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.")
                        
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break

        # NOTE: no early-exit here when num_yes==0. The original code treated
        # "nobody wants to keep debating" as a reason to skip collecting each
        # agent's current answer and jump straight to the final decision with
        # whatever was in `final_answer` from a PRIOR round (or nothing, on
        # round 1) - silence was treated as "done" without ever checking
        # whether the panel actually agrees. Since agents very often say no
        # more discussion is needed, this path was hit constantly. Instead,
        # always collect this round's answers and run the real consensus
        # check below; a quiet round just means fewer messages were logged
        # on the way to that check, not that the round is skipped outright.
        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        round_summary = "".join(f"({k}): {v}\n" for k, v in tmp_final_answer.items())
        interaction_history += f"\n{round_name} answers:\n{round_summary}"

        # Consensus check (Algorithm 1): if every agent lands on the same
        # option letter, stop early instead of burning the remaining rounds.
        # The original code had no such check - it always ran either until
        # nobody wanted to keep debating or until num_rounds was exhausted.
        extracted = [_extract_answer_letter(v) for v in tmp_final_answer.values()]
        extracted = [e for e in extracted if e]
        consensus = bool(extracted) and len(set(extracted)) == 1 and len(extracted) == len(tmp_final_answer)
        if consensus:
            print(f"[INFO] Consensus reached in {round_name} ({extracted[0]}).")
            break

        # Moderator-feedback loop (Algorithm 1, lines 13-17): when the panel
        # hasn't converged and rounds remain, a moderator reviews the round
        # and gives feedback each agent carries into the next round. The
        # original code had no such mechanism at all - every round repeated
        # the same debate protocol with no guidance injected between rounds.
        if use_moderator_review and n < num_rounds:
            feedback_agent = Agent(
                instruction="You are a moderator who reviews a medical expert panel's discussion and gives constructive feedback to help them reach a well-reasoned consensus.",
                role="Moderator", model_info=model,
            )
            feedback_agent.chat("You are a moderator who reviews a medical expert panel's discussion and gives constructive feedback to help them reach a well-reasoned consensus.")
            feedback = feedback_agent.chat(
                f"Here is the panel's current round of answers:\n{round_summary}\n\nQuestion: {question}\n\n"
                "The panel has not yet reached consensus. Point out where they disagree, any gaps in "
                "reasoning, and what each should reconsider before the next round. Be specific and "
                "concise (4-6 sentences total)."
            )
            interaction_history += f"\nModerator feedback after {round_name}:\n{feedback}\n"
            for a in medical_agents:
                a.chat(f"Moderator feedback on the panel's discussion so far:\n{feedback}\n\nPlease reconsider your position in light of this feedback before the next round.")

        if n < num_rounds:
            round_opinions[n + 1] = tmp_final_answer

    print('\nInteraction Log')
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')

    # Algorithm 1: ans <- Agent(Q, Interaction) - the moderator reviews the
    # FULL record across all rounds (including any feedback given between
    # rounds), not just a snapshot of the last round's answers. The original
    # code only ever passed the last round's tmp_final_answer, discarding
    # every earlier round and any moderator feedback the instant a new round
    # started.
    consensus_note = "The panel reached consensus." if consensus else "The panel did not fully converge within the round limit; use your judgment to weigh the strongest reasoning."
    _decision = moderator.temp_responses(
        f"Here is the full record of the panel's discussion across all rounds "
        f"(including any moderator feedback given between rounds):\n{interaction_history}\n\n"
        f"{consensus_note}\n\nGiven this discussion, review each agent's opinion and make the final "
        f"answer to the question by majority vote. Your answer should be like below format:\n"
        f"Answer: C) 2th pharyngeal arch\n\nQuestion: {question}", img_path=None)
    answer_text = majority_vote(_decision)

    # Confidence comes from agreement among the recruited experts' own final
    # answers (not the moderator's self-consistency across ensemble
    # temperatures, which is a single entity's repeated sampling, not
    # independent specialist agreement).
    confidence_info = consensus_confidence(list(tmp_final_answer.values()))
    winner_letter = confidence_info.get("winner")
    attribution = [
        {
            "role": role,
            "opinion": opinion,
            "agreed_with_final": (_extract_answer_letter(opinion) == winner_letter) if winner_letter else None,
        }
        for role, opinion in tmp_final_answer.items()
    ]
    explanation = build_explanation(
        answer=_extract_answer_letter(answer_text),
        confidence_info=confidence_info,
        reasoning=answer_text,
        evidence=citations,
        attribution=attribution,
        complexity="intermediate",
        consensus_reached=consensus,
    )
    final_decision = {'majority': _decision, 'answer': answer_text, 'explanation': explanation}

    moderator_emoji = '\U0001F468\u200D\u2696\uFE0F'
    print(f"{moderator_emoji} moderator's final decision (by majority vote):", _decision)
    print()

    return final_decision

def process_advanced_query(question, model, args, img_path=None):
    print("[STEP 1] Recruitment")
    group_instances = []

    recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=FAST_MODEL)
    tmp_agent.chat(recruit_prompt)

    num_teams = 3  # You can adjust this number as needed
    num_agents = 3  # You can adjust this number as needed

    recruited = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format: one 'Group N - Team Name' line per team followed by one 'Member N: Role - description' line per member, plain text only, with no markdown tables, bullet points, or bold formatting.")

    recruited = _normalize_llm_list_text(recruited)
    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]

    evidence_text, citations = get_rag_evidence_and_citations(question, getattr(args, 'use_rag', False))

    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
        print()

        group_instance = Group(res_gs['group_goal'], res_gs['members'], question, model_info=model, evidence=evidence_text)
        group_instances.append(group_instance)

    # STEP 2. Sequential team processing: IAT -> other MDTs -> FRDT, each
    # given the prior teams' reports as context. The original code ran every
    # team fully independently and in isolation (no team ever saw another
    # team's findings), which contradicts the paper's own description of
    # this stage ("each team builds upon the foundation laid by the
    # previous"). It also only ever fed the IAT's report into the final
    # decision prompt below - the other MDTs' and the FRDT's work was
    # computed and then silently discarded.
    all_reports = ""
    team_reports = []  # [{"team": goal, "report": text}, ...] - kept per-team
    # (not just concatenated) so confidence/attribution can be computed per
    # team below, the same way the intermediate path does per-agent.

    # STEP 2.1. IAT (Initial Assessment Team) Process
    initial_assessment_report = ""
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower() or 'iat' in group_instance.goal.lower():
            init_assessment = group_instance.interact(comm_type='internal', prior_reports=all_reports, img_path=img_path)
            initial_assessment_report += f"{group_instance.goal}\n{init_assessment}\n\n"
            team_reports.append({"team": group_instance.goal, "report": init_assessment})
    all_reports += initial_assessment_report

    # STEP 2.2. Other MDTs Process
    assessment_report = ""
    for group_instance in group_instances:
        goal_lower = group_instance.goal.lower()
        is_iat = 'initial' in goal_lower or 'iap' in goal_lower or 'iat' in goal_lower
        is_frdt = 'review' in goal_lower or 'decision' in goal_lower or 'frdt' in goal_lower
        if not is_iat and not is_frdt:
            assessment = group_instance.interact(comm_type='internal', prior_reports=all_reports, img_path=img_path)
            assessment_report += f"{group_instance.goal}\n{assessment}\n\n"
            team_reports.append({"team": group_instance.goal, "report": assessment})
    all_reports += assessment_report

    # STEP 2.3. FRDT (Final Review and Decision Team) Process
    compiled_report = ""
    for group_instance in group_instances:
        goal_lower = group_instance.goal.lower()
        if 'review' in goal_lower or 'decision' in goal_lower or 'frdt' in goal_lower:
            decision = group_instance.interact(comm_type='internal', prior_reports=all_reports, img_path=img_path)
            compiled_report += f"{group_instance.goal}\n{decision}\n\n"
            team_reports.append({"team": group_instance.goal, "report": decision})

    # STEP 3. Final Decision - the moderator reviews EVERY team's report,
    # not just the IAT's.
    combined_reports = f"{initial_assessment_report}{assessment_report}{compiled_report}".strip()
    decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model)
    tmp_agent.chat(decision_prompt)

    _decision = tmp_agent.temp_responses(f"""Investigation:\n{combined_reports}\n\nQuestion: {question}""", img_path=None)
    answer_text = majority_vote(_decision)

    # Confidence from agreement across teams' own reports (each team's
    # report implicitly leans toward an answer letter, extractable the same
    # way as any other agent response).
    confidence_info = consensus_confidence([tr["report"] for tr in team_reports])
    winner_letter = confidence_info.get("winner")
    attribution = [
        {
            "role": tr["team"],
            "opinion": tr["report"],
            "agreed_with_final": (_extract_answer_letter(tr["report"]) == winner_letter) if winner_letter else None,
        }
        for tr in team_reports
    ]
    explanation = build_explanation(
        answer=_extract_answer_letter(answer_text),
        confidence_info=confidence_info,
        reasoning=answer_text,
        evidence=citations,
        attribution=attribution,
        complexity="advanced",
        consensus_reached=(confidence_info.get("confidence") == 1.0) if confidence_info.get("confidence") is not None else None,
    )
    return {'majority': _decision, 'answer': answer_text, 'explanation': explanation}