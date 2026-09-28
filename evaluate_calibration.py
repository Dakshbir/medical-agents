"""
Reads one or more output/*.json result files (produced by main.py, each
result row now carries a 'confidence' field alongside 'correct') and
evaluates how well-calibrated that confidence actually is: does "80%
confident" really mean "correct about 80% of the time"?

Usage:
  python evaluate_calibration.py output/*.json
  python evaluate_calibration.py output/openai-gpt-oss-120b_medqa_adaptive.json

Produces:
  - a printed per-bucket table (bucket range, n, avg confidence, accuracy)
  - Expected Calibration Error (ECE)
  - calibration_reliability.png (a reliability diagram), if matplotlib is available
"""
import argparse
import glob
import json
import sys


def load_confidence_correct_pairs(paths):
    pairs = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        rows = data.get('results', data if isinstance(data, list) else [])
        for row in rows:
            conf = row.get('confidence')
            correct = row.get('correct')
            if conf is None or correct is None:
                continue
            pairs.append((conf, bool(correct), row.get('difficulty'), path))
    return pairs


def bucket_stats(pairs, n_bins=5):
    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets = [[] for _ in range(n_bins)]
    for conf, correct, *_ in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        buckets[idx].append((conf, correct))

    rows = []
    total = len(pairs)
    ece = 0.0
    for i, bucket in enumerate(buckets):
        lo, hi = edges[i], edges[i + 1]
        n = len(bucket)
        if n == 0:
            rows.append({"range": f"{lo:.1f}-{hi:.1f}", "n": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = sum(c for c, _ in bucket) / n
        acc = sum(1 for _, correct in bucket if correct) / n
        ece += (n / total) * abs(acc - avg_conf)
        rows.append({"range": f"{lo:.1f}-{hi:.1f}", "n": n, "avg_confidence": avg_conf, "accuracy": acc})
    return rows, ece


def print_report(pairs, rows, ece):
    overall_acc = sum(1 for _, correct, *_ in pairs if correct) / len(pairs) if pairs else 0.0
    print(f"\nTotal graded questions with a confidence score: {len(pairs)}")
    print(f"Overall accuracy: {overall_acc:.1%}\n")
    print(f"{'Confidence bucket':<20}{'n':>6}{'avg confidence':>18}{'actual accuracy':>18}")
    print("-" * 62)
    for r in rows:
        if r["n"] == 0:
            print(f"{r['range']:<20}{0:>6}{'--':>18}{'--':>18}")
        else:
            print(f"{r['range']:<20}{r['n']:>6}{r['avg_confidence']*100:>17.1f}%{r['accuracy']*100:>17.1f}%")
    print("-" * 62)
    print(f"Expected Calibration Error (ECE): {ece:.4f}  (0 = perfectly calibrated, lower is better)\n")


def plot_reliability_diagram(rows, out_path="calibration_reliability.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping reliability diagram (pip install matplotlib to enable).")
        return

    xs = [r["avg_confidence"] for r in rows if r["n"] > 0]
    ys = [r["accuracy"] for r in rows if r["n"] > 0]
    ns = [r["n"] for r in rows if r["n"] > 0]
    if not xs:
        print("No non-empty buckets - skipping reliability diagram.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    sizes = [max(50, n * 8) for n in ns]
    ax.scatter(xs, ys, s=sizes, color="#2563eb", alpha=0.8, label="Observed", zorder=3)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Average stated confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_title("Reliability diagram: MDAgents consensus confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved reliability diagram to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='output/*.json result files (globs allowed)')
    parser.add_argument('--bins', type=int, default=5)
    parser.add_argument('--by-difficulty', action='store_true',
                         help='Also print a separate calibration table per difficulty level')
    args = parser.parse_args()

    expanded = []
    for p in args.paths:
        matched = glob.glob(p)
        expanded.extend(matched if matched else [p])

    pairs = load_confidence_correct_pairs(expanded)
    if not pairs:
        print("No (confidence, correct) pairs found in the given file(s). "
              "Make sure you're pointing at output/*.json files produced after "
              "the confidence/explanation feature was added.", file=sys.stderr)
        sys.exit(1)

    rows, ece = bucket_stats(pairs, n_bins=args.bins)
    print_report(pairs, rows, ece)
    plot_reliability_diagram(rows)

    if args.by_difficulty:
        by_diff = {}
        for conf, correct, difficulty, _ in pairs:
            by_diff.setdefault(difficulty, []).append((conf, correct))
        for difficulty, sub_pairs in by_diff.items():
            print(f"\n=== Difficulty: {difficulty} ===")
            sub_rows, sub_ece = bucket_stats([(c, r) for c, r in sub_pairs], n_bins=args.bins)
            print_report([(c, r, difficulty, '') for c, r in sub_pairs], sub_rows, sub_ece)


if __name__ == "__main__":
    main()
