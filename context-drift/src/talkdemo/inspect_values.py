"""Which status values do the replies filter on? (the completed_only convention swaps a value,
so "wrong" can mean the other value, both values, or no filter at all)

    python src/talkdemo/inspect_values.py [scenario]      # reads results/runs/<scenario>/
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent  # the experiment directory
sys.path.insert(0, str(SRC))

from talkdemo import experiments as ex  # noqa: E402


def status_values(records: list[dict], target: str = "completed_only") -> dict[tuple, Counter]:
    """(runner, experiment, condition) -> counts of the value sets the replies filter on."""
    out: dict[tuple, Counter] = defaultdict(Counter)
    for r in records:
        if r["target"] != target or target not in r["status"]:
            continue
        values = r["detail"].get(target, {}).get("values")
        label = "unparsed" if values is None else " + ".join(values) if values else "no filter"
        out[(r["runner"], r["experiment"], r["condition"])][label] += 1
    return out


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "shop"
    table = status_values(ex.load_records(ROOT / "results" / "runs" / scenario))
    for (runner, experiment, condition), counts in sorted(table.items()):
        total = sum(counts.values())
        parts = ", ".join(f"{label}: {n}" for label, n in counts.most_common())
        print(f"{runner:22s} {experiment} {condition:20s} n={total:3d}  {parts}")


if __name__ == "__main__":
    main()
