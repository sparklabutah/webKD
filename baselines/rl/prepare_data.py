"""Multi-turn policy-gradient baseline for web agents."""

import argparse
import os

from verifiers.utils.timewarp_data import build_timewarp_dataset


def main():
    ap = argparse.ArgumentParser(description="Materialise TimeWarp datasets")
    ap.add_argument("--output_dir", default="./data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ("train", "test"):
        ds = build_timewarp_dataset(split=split, seed=args.seed)
        path = os.path.join(args.output_dir, f"{split}.parquet")
        ds.to_parquet(path)
        print(f"Wrote {path}  ({len(ds)} tasks)")
        if len(ds):
            print(f"  example task_id={ds[0]['task_id']} answer={ds[0]['answer']!r}")


if __name__ == "__main__":
    main()
