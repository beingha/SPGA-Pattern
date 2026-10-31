"""Generate x/y spacing constraints from binary topology records."""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd


def _closed_intervals(values: np.ndarray, active_value: int) -> np.ndarray:
    """Return inclusive intervals of active runs separated by one empty cell."""
    intervals: list[list[int]] = []
    start: int | None = None
    gap_seen = False

    for index, value in enumerate(values):
        if value == active_value:
            if start is None:
                start = index
            elif gap_seen:
                intervals.append([start, index])
                start = index
            gap_seen = False
        elif start is not None:
            gap_seen = True

    return np.asarray(intervals, dtype=int).reshape(-1, 2)


def check_floating_wire(values: np.ndarray) -> np.ndarray:
    """Find intervals spanning a wire separated by one empty cell."""
    return _closed_intervals(np.asarray(values), active_value=0)


def check_spacing(values: np.ndarray) -> np.ndarray:
    """Find intervals spanning two active regions separated by one empty cell."""
    return _closed_intervals(np.asarray(values), active_value=1)


def constraints_for_row(row: pd.Series) -> list[str]:
    """Build textual constraints and x/y delta values for one topology row."""
    width, height = int(row["cX"]), int(row["cY"])
    topology = np.asarray(list(str(row["topoSig"])), dtype=np.uint8).reshape(height, width)
    x_delta = np.zeros(width, dtype=int).astype(str)
    y_delta = np.zeros(height, dtype=int).astype(str)
    constraints: list[str] = []

    for row_index in range(height):
        y_index = height - 1 - row_index
        if row_index in (0, height - 1):
            y_delta[y_index] = "1-16" if topology[row_index].sum() else "1-96"
        else:
            y_delta[y_index] = "16"

        for start, end in np.vstack((
            check_spacing(topology[row_index]),
            check_floating_wire(topology[row_index]),
        )):
            if end - start == 2:
                x_delta[end] = "16-96"
            else:
                constraints.append(f"x{end}-x{start + 1} >= 16")

    constraints.extend((f"y{height}-y0 == 192", f"x{width}-x0 == 192"))
    x_delta[x_delta == "0"] = "1-96"
    return [", ".join(constraints), " ".join(x_delta), " ".join(y_delta)]


def add_constraints(input_path: str | Path, output_path: str | Path,
                    workers: int = 1) -> None:
    """Read msgpack records, add constraints, and write a new msgpack file."""
    with Path(input_path).open("rb") as handle:
        records = msgpack.unpack(handle, raw=False)
    frame = pd.DataFrame(records)
    rows = [frame.iloc[index] for index in range(len(frame))]

    if workers > 1:
        with Pool(processes=workers) as pool:
            generated = pool.map(constraints_for_row, rows)
    else:
        generated = [constraints_for_row(row) for row in rows]

    additions = pd.DataFrame(generated, columns=["constraints", "xDelta", "yDelta"])
    result = pd.concat([frame.reset_index(drop=True), additions], axis=1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        msgpack.pack(result.to_dict(orient="records"), handle, use_bin_type=True)
    print(f"Wrote {len(result)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input msgpack file")
    parser.add_argument("output", type=Path, help="Output msgpack file")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    add_constraints(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()
