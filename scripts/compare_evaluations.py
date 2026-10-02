"""Compare insertion methods only after checking complete, matched resets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_insertion"))

from evaluation_cases import compare_runs, write_json_new  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6, help="State absolute tolerance; rtol=0.")
    args = parser.parse_args()
    try:
        summary = compare_runs(args.runs, atol=args.atol)
        write_json_new(args.output, summary)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
