# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Read every result file and print the comparison the essay is built on.

Two tables. The first is the headline: how many rows each system returns on a turn
where nobody asked about the past. The second is what happened to a fact that changed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def row_id(row: dict) -> str:
    """A stable identifier for one stored row, whatever the library calls it."""

    return str(row.get("id") or row.get("key") or row.get("memory") or row.get("value"))


def row_text(row: dict) -> str:
    """The stored text, whatever the library calls it."""

    if "memory" in row:
        return str(row["memory"])
    value = row.get("value")
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value)


def load() -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for path in sorted(RESULTS.glob("*-run*.json")):
        data = json.loads(path.read_text())
        runs[data["system"]].append(data)
    return runs


def fmt_counts(values: list[int]) -> str:
    if not values:
        return "-"
    if len(set(values)) == 1:
        return str(values[0])
    return f"{min(values)}-{max(values)} ({', '.join(map(str, values))})"


def main() -> None:
    runs = load()
    if not runs:
        print("No results yet.")
        return

    probe_order: list[tuple[str, str, str, bool]] = []
    for data in next(iter(runs.values())):
        for p in data["probes"]:
            key = (p["probe_id"], p["query"], p["kind"], p["expect_empty"])
            if key not in probe_order:
                probe_order.append(key)
        break

    systems = sorted(runs)

    print("=" * 78)
    print("ROWS RETURNED PER PROBE, at each system's documented defaults")
    print("=" * 78)
    print(f"{'probe':<34} {'kind':<11} {'want':<6} " + " ".join(f"{s:<12}" for s in systems))
    print("-" * 78)
    for probe_id, query, kind, expect_empty in probe_order:
        want = "empty" if expect_empty else "hit"
        cells = []
        for s in systems:
            counts = []
            for data in runs[s]:
                for p in data["probes"]:
                    if p["probe_id"] == probe_id:
                        counts.append(p["returned"])
            cells.append(f"{fmt_counts(counts):<12}")
        print(f"{query[:33]:<34} {kind:<11} {want:<6} " + " ".join(cells))

    print()
    print("=" * 78)
    print("ROWS RETURNED AS A FRACTION OF THE STORE")
    print("=" * 78)
    print("1.00 means the search returned everything the store held, whatever was asked.")
    print()
    print(f"{'probe':<34} {'want':<6} " + " ".join(f"{s:<12}" for s in systems))
    print("-" * 78)
    for probe_id, query, kind, expect_empty in probe_order:
        want = "empty" if expect_empty else "hit"
        cells = []
        for s in systems:
            fracs = []
            for data in runs[s]:
                stage_key = (
                    "store_after_session_1"
                    if probe_id == "p1_latency_graph"
                    else "store_after_update"
                )
                size = len(data[stage_key])
                for p in data["probes"]:
                    if p["probe_id"] == probe_id and size:
                        fracs.append(p["returned"] / size)
            avg = sum(fracs) / len(fracs) if fracs else 0.0
            cells.append(f"{avg:<12.2f}")
        print(f"{query[:33]:<34} {want:<6} " + " ".join(cells))

    print()
    print("=" * 78)
    print("STORE SIZE")
    print("=" * 78)
    print(f"{'system':<14} {'after January':<18} {'after the update':<18} {'net change'}")
    print("-" * 78)
    for s in systems:
        before = [len(d["store_after_session_1"]) for d in runs[s]]
        after = [len(d["store_after_update"]) for d in runs[s]]
        delta = [a - b for b, a in zip(before, after)]
        print(f"{s:<14} {fmt_counts(before):<18} {fmt_counts(after):<18} {fmt_counts(delta)}")

    print()
    print("=" * 78)
    print("WHAT THE UPDATE TURN DID TO THE JANUARY ROWS")
    print("=" * 78)
    print("Tracked by row id, not by text. A row whose id survives but whose text")
    print("changed was edited in place; the January wording is gone.")
    print()
    for s in systems:
        for data in runs[s]:
            before = {row_id(r): row_text(r) for r in data["store_after_session_1"]}
            after = {row_id(r): row_text(r) for r in data["store_after_update"]}
            kept = [k for k in before if k in after]
            edited = [k for k in kept if before[k] != after[k]]
            added = [k for k in after if k not in before]
            gone = [k for k in before if k not in after]
            print(
                f"  {s:<9} run {data['run']}:  {len(before)} -> {len(after)} rows"
                f"   kept {len(kept)}, edited in place {len(edited)},"
                f" added {len(added)}, removed {len(gone)}"
            )
        print()

    print()
    print("=" * 78)
    print("CONFIG (must be identical across systems for the comparison to hold)")
    print("=" * 78)
    for s in systems:
        c = runs[s][0]["config"]
        print(f"  {s:<12} llm={c.get('llm')}  embedder={c.get('embedder')}  runs={len(runs[s])}")
        if "active_signals" in c:
            print(f"  {'':<12} retrieval signals: {c['active_signals']}")
        print(f"  {'':<12} versions: {runs[s][0]['versions']}")


if __name__ == "__main__":
    main()
