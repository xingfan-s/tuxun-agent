"""Evaluate isolated JSONL predictions and emit a baseline report."""

import argparse
import json
from pathlib import Path

from app.evaluation import evaluate_ablations, evaluate_by_slice, evaluate_predictions, validate_dataset_isolation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--index-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {
        "overall": evaluate_predictions(rows),
        "by_slice": evaluate_by_slice(rows),
        "ablations": evaluate_ablations(rows),
    }
    if args.index_metadata:
        index_rows = json.loads(args.index_metadata.read_text(encoding="utf-8"))
        if isinstance(index_rows, dict):
            index_rows = index_rows.get("entries", [])
        report["isolation_violations"] = validate_dataset_isolation(rows, index_rows)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
