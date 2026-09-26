"""Generate pairwise training/inference features or run the synthetic smoke demo."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features import build_feature_matrix, synthetic_smoke_check


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-demo", action="store_true", help="run synthetic feature assertions")
    parser.add_argument("--pairs")
    parser.add_argument("--source1")
    parser.add_argument("--source2")
    parser.add_argument("--source3")
    parser.add_argument("--ground-truth")
    parser.add_argument("--output", default="output/features.tsv")
    args = parser.parse_args()
    if args.synthetic_demo:
        demo = synthetic_smoke_check()
        print(demo.head(5).to_string(index=False))
        print(f"Synthetic checks passed: {len(demo)} examples")
        return
    required = (args.pairs, args.source1, args.source2, args.source3)
    if not all(required):
        parser.error("provide --pairs, --source1, --source2, and --source3 (or --synthetic-demo)")
    truth = read_tsv(args.ground_truth) if args.ground_truth else None
    frame = build_feature_matrix(read_tsv(args.pairs), read_tsv(args.source1),
                                 read_tsv(args.source2), read_tsv(args.source3), truth)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index=False)
    print(f"Wrote {len(frame)} pair rows and {len(frame.columns) - 2} columns to {output}")


if __name__ == "__main__":
    main()
