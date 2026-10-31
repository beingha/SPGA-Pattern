"""Convert topology CSV files to the msgpack format used by the loaders."""

from __future__ import annotations

import argparse
from pathlib import Path

import msgpack
import pandas as pd


def convert(csv_path: str | Path, train_out: str | Path, test_out: str | Path,
            train_count: int = 8001) -> None:
    """Convert a CSV file and split valid records into train and test sets."""
    if train_count < 0:
        raise ValueError("train_count must be non-negative")

    frame = pd.read_csv(csv_path)
    records = []
    skipped = 0
    for row in frame.itertuples(index=False):
        record = row._asdict()
        signature = str(record["topoSig"])
        width, height = int(record["cX"]), int(record["cY"])
        if len(signature) != width * height or any(bit not in "01" for bit in signature):
            skipped += 1
            continue
        records.append({"cX": width, "cY": height, "topoSig": signature})

    train_records = records[:train_count]
    test_records = records[train_count:]
    for output_path, subset in ((train_out, train_records), (test_out, test_records)):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            msgpack.pack(subset, handle, use_bin_type=True)
        print(f"Wrote {len(subset)} samples to {output_path}")

    print(f"Valid samples: {len(records)}; skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input topology CSV")
    parser.add_argument("train_output", type=Path, help="Training msgpack output")
    parser.add_argument("test_output", type=Path, help="Test msgpack output")
    parser.add_argument("--train-count", type=int, default=8001)
    args = parser.parse_args()
    convert(args.csv, args.train_output, args.test_output, args.train_count)


if __name__ == "__main__":
    main()
