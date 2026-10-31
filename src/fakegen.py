"""Generate small synthetic topology datasets for smoke tests and examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd


class FakePatternGenerator:
    """Generate square binary layouts with alternating empty rows."""

    def __init__(self, min_size: int, max_size: int, count: int, seed: int | None = None):
        if min_size <= 0 or max_size < min_size:
            raise ValueError("min_size must be positive and no larger than max_size")
        if count < 0:
            raise ValueError("count must be non-negative")

        self.sizes = np.arange(min_size, max_size + 1)
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.records: list[dict[str, int | str]] = []

    def generate(self) -> list[dict[str, int | str]]:
        """Generate records with cX, cY, and flattened topology signatures."""
        records = []
        for sample_id in range(self.count):
            size = int(self.rng.choice(self.sizes))
            topology = self.rng.integers(0, 2, size=(size, size), dtype=np.uint8)
            topology[1::2] = 0
            records.append(
                {
                    "id": sample_id,
                    "cX": size,
                    "cY": size,
                    "topoSig": "".join(topology.ravel().astype(str)),
                }
            )
        self.records = records
        return records

    def save(self, path: str | Path) -> None:
        """Write generated records as a msgpack list."""
        if not self.records:
            self.generate()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            msgpack.pack(self.records, handle, use_bin_type=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output msgpack file")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    generator = FakePatternGenerator(args.min_size, args.max_size, args.count, args.seed)
    generator.generate()
    generator.save(args.output)
    print(f"Wrote {len(generator.records)} samples to {args.output}")


if __name__ == "__main__":
    main()
