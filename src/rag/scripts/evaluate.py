import argparse
from pathlib import Path

from rag.metrics import compute_avg_f1, exact_match_score


def run(args):
    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    total, avg_f1 = compute_avg_f1(str(path))
    em = exact_match_score(str(path))
    print(f"Rows: {total}")
    print(f"Average F1: {avg_f1:.2%}")
    print(f"Exact Match: {em:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CSV predictions.")
    parser.add_argument("--csv", required=True, help="Path to CSV with expected_answer and answer columns.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
