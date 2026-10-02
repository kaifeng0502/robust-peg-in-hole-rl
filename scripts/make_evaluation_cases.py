"""Freeze reproducible insertion reset inputs before evaluating any controller."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_insertion"))

from evaluation_cases import make_case_set, write_case_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        case_set = make_case_set(args.trials, args.seed)
        write_case_set(args.output, case_set)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {len(case_set['cases'])} cases to {args.output}")
    print(f"case_set_id: {case_set['case_set_id']}")
    print("Equal reset seeds require recorded initial-state verification before comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
