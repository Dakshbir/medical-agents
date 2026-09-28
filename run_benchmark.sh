#!/bin/bash
# Runs the adaptive MDAgents pipeline across all 7 implemented datasets and
# saves results to output/*.json (accuracy + per-question confidence, ready
# for evaluate_calibration.py). Continues past a single dataset's failure
# rather than aborting the whole batch, so one exhausted quota / one bad run
# doesn't lose progress on the others.
set -u

N=${N:-10}
NUM_ROUNDS=${NUM_ROUNDS:-2}
NUM_TURNS=${NUM_TURNS:-3}
export ENSEMBLE_TEMPERATURES="${ENSEMBLE_TEMPERATURES:-0.0,0.7}"
TEXT_MODEL="${TEXT_MODEL:-qwen/qwen3.8-27b}"
VISION_MODEL="${VISION_MODEL:-gemini-2.5-flash}"

PY="./.venv/Scripts/python.exe"
mkdir -p benchmark_logs

run_one() {
    dataset=$1
    model=$2
    echo "=================================================================="
    echo "=== $dataset  (model=$model, N=$N)  $(date)"
    echo "=================================================================="
    $PY main.py --dataset "$dataset" --model "$model" --difficulty adaptive \
        --num_samples "$N" --use_rag --num_rounds "$NUM_ROUNDS" --num_turns "$NUM_TURNS" \
        > "benchmark_logs/${dataset}.log" 2>&1
    status=$?
    echo "--- $dataset finished, exit=$status ---"
    tail -n 5 "benchmark_logs/${dataset}.log"
    echo
}

for ds in medqa pubmedqa ddxplus symcat medbullets; do
    run_one "$ds" "$TEXT_MODEL"
done

# Skipped for now: Gemini's free-tier daily quota (observed: 20 requests/day
# for gemini-2.5-flash) is exhausted as of this run. Re-enable once it resets
# or a key from a different Google account is available - see ARCHITECTURE.md.
# for ds in pmcvqa pathvqa; do
#     run_one "$ds" "$VISION_MODEL"
# done
echo "Skipping pmcvqa/pathvqa - Gemini quota exhausted for today. Run manually later:"
echo "  ./.venv/Scripts/python.exe main.py --dataset pmcvqa --model $VISION_MODEL --difficulty adaptive --num_samples $N --use_rag"
echo "  ./.venv/Scripts/python.exe main.py --dataset pathvqa --model $VISION_MODEL --difficulty adaptive --num_samples $N --use_rag"

echo "=================================================================="
echo "BENCHMARK BATCH COMPLETE $(date)"
echo "=================================================================="
