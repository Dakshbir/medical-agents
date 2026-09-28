import os
import sys
import json
import random
import argparse

# Windows consoles default to a codepage (e.g. cp1252) that can't encode the
# doctor emojis this script prints; force UTF-8 output regardless of platform.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from utils import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    load_data, create_question, determine_difficulty, _is_gemini,
    _extract_answer_letter, _extract_yes_no_maybe, process_basic_query,
    process_intermediate_query, process_advanced_query
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medqa')
parser.add_argument('--model', type=str, default='openai/gpt-oss-120b')
parser.add_argument('--difficulty', type=str, default='adaptive')
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--use_rag', action=argparse.BooleanOptionalAction, default=True,
                     help='Inject live-retrieved medical evidence (rag.py) into agent prompts (paper\'s "MedRAG" ablation).')
parser.add_argument('--use_moderator_review', action=argparse.BooleanOptionalAction, default=True,
                     help="In the moderate/MDT path, have a moderator give the panel feedback between rounds that haven't reached consensus (Algorithm 1, lines 13-17).")
parser.add_argument('--num_rounds', type=int, default=5,
                     help="Max debate rounds in the moderate/MDT path. Agents will keep 'wanting to talk' to affirm agreement, not just to resolve disagreement, so this can get expensive - lower it (e.g. 2-3) for cost-bounded runs.")
parser.add_argument('--num_turns', type=int, default=5,
                     help="Max turns per round in the moderate/MDT path. See --num_rounds.")
args = parser.parse_args()

# NOTE: these must match the actual data/<dataset>/ directory names (no
# hyphens) - an earlier version of this check used 'pmc-vqa'/'path-vqa',
# which never matched anything real and made this validation dead code.
IMAGE_DATASETS = {'pmcvqa', 'pathvqa', 'mimiccxr', 'medvidqa'}
if args.dataset in IMAGE_DATASETS and not _is_gemini(args.model):
    parser.error(
        f"--dataset {args.dataset} requires image input, which only Gemini models "
        f"support in this project (got --model {args.model}). Pass e.g. "
        f"--model gemini-2.5-flash."
    )

model, client = setup_model(args.model)
test_qa, examplers = load_data(args.dataset)

agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
random.shuffle(agent_emoji)

def grade(sample, final_decision):
    """Scores a prediction against ground truth. Multiple-choice datasets are
    scored by option letter; PubMedQA-style datasets (no 'options', a plain
    yes/no/maybe 'answer') are scored by that word. Returns None if the
    prediction couldn't be parsed out of the response (counted as wrong, but
    kept distinguishable in the saved output for debugging prompts/parsing)."""
    predicted_text = final_decision.get('answer', '')
    if sample.get('options'):
        predicted = _extract_answer_letter(predicted_text)
        expected = sample.get('answer_idx', '').strip().upper() or None
    else:
        predicted = _extract_yes_no_maybe(predicted_text)
        expected = (sample.get('answer') or '').strip().lower() or None
    correct = predicted is not None and expected is not None and predicted.lower() == expected.lower()
    return predicted, expected, correct

# Set up the output path up front and write incrementally (after every
# question) rather than once at the very end. The original code kept
# `results` purely in memory and only wrote the file after the whole loop
# finished - one uncaught exception on, say, question 8 of 10 (e.g. Groq
# rate-limited AND the Gemini fallback also failed, which happens for real
# when Gemini's daily quota is exhausted) silently threw away every
# already-computed result for the whole dataset, not just the failed one.
path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(path):
    os.makedirs(path)
safe_model_name = args.model.replace('/', '-')
output_path = os.path.join(path, f'{safe_model_name}_{args.dataset}_{args.difficulty}.json')


def save(results, num_correct, num_graded, num_failed):
    accuracy = num_correct / num_graded if num_graded else 0.0
    output = {
        'dataset': args.dataset,
        'model': args.model,
        'difficulty': args.difficulty,
        'use_rag': args.use_rag,
        'use_moderator_review': args.use_moderator_review,
        'accuracy': accuracy,
        'num_correct': num_correct,
        'num_graded': num_graded,
        'num_failed': num_failed,
        'results': results,
    }
    with open(output_path, 'w') as file:
        json.dump(output, file, indent=4)
    return accuracy


results = []
num_correct = 0
num_graded = 0
num_failed = 0
for no, sample in enumerate(tqdm(test_qa)):
    if no == args.num_samples:
        break

    print(f"\n[INFO] no: {no}")

    try:
        question, img_path = create_question(sample, args.dataset)
        difficulty = determine_difficulty(question, args.difficulty)

        print(f"difficulty: {difficulty}")

        if difficulty == 'basic':
            final_decision = process_basic_query(question, examplers, args.model, args, img_path=img_path)
        elif difficulty == 'intermediate':
            final_decision = process_intermediate_query(question, examplers, args.model, args, img_path=img_path)
        elif difficulty == 'advanced':
            final_decision = process_advanced_query(question, args.model, args, img_path=img_path)

        predicted, expected, correct = grade(sample, final_decision)
        num_graded += 1
        num_correct += int(correct)
        confidence = (final_decision.get('explanation') or {}).get('confidence')
        print(f"predicted={predicted!r} expected={expected!r} correct={correct} confidence={confidence!r}")

        results.append({
            'question': question,
            'expected': expected,
            'predicted': predicted,
            'correct': correct,
            'confidence': confidence,
            'options': sample.get('options'),
            'response': final_decision,
            'difficulty': difficulty,
        })
    except Exception as e:
        # Keep going: one bad question (rate limit exhaustion, a parsing
        # edge case, a transient network failure) shouldn't cost every other
        # already-computed result in this dataset.
        num_failed += 1
        print(f"[ERROR] question {no} failed: {type(e).__name__}: {e}")
        results.append({
            'question': sample.get('question', ''),
            'error': f"{type(e).__name__}: {e}",
            'difficulty': None,
        })

    save(results, num_correct, num_graded, num_failed)

accuracy = save(results, num_correct, num_graded, num_failed)
print(f"\n[RESULT] {args.dataset}/{args.difficulty} accuracy: {num_correct}/{num_graded} = {accuracy:.1%}"
      f" ({num_failed} failed)")