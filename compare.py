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

# Each library's own default cap on a search result. These differ, and the difference
# matters: a system that returns 20 rows and one that returns 8 are not necessarily
# behaving differently, they may just have different caps. What matters is whether
# anything other than the cap ever kept a row out.
CAPS = {"mem0": 20, "langmem": 10, "langmem-nodistractors": 10, "graphiti": 10, "weave": 8, "letta": 5}


def row_id(row: dict) -> str:
    """A stable identifier for one stored row, whatever the library calls it."""

    return str(row.get("id") or row.get("key") or row.get("memory") or row.get("value"))


def row_text(row: dict) -> str:
    """The stored text, whatever the library calls it."""

    if "memory" in row:
        return str(row["memory"])
    value = row.get("value")
    if isinstance(value, dict):
        inner = value.get("content", value)
        if isinstance(inner, dict):
            return str(inner.get("content", inner))
        return str(inner)
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
    for system_runs in runs.values():
        for data in system_runs:
            for p in data["probes"]:
                key = (p["probe_id"], p["query"], p["kind"], p["expect_empty"])
                if key not in probe_order:
                    probe_order.append(key)

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
    print("DOES ANYTHING BUT THE CAP LIMIT THE RESULT?")
    print("=" * 78)
    print("Every probe across every run. A probe that comes back at the cap means")
    print("nothing else excluded a row: no threshold, no floor, no gate. A probe below")
    print("the cap means something other than the cap decided to stop.")
    print()
    print(f"{'system':<14} {'cap':<6} {'probes at the cap':<20} {'probes below it'}")
    print("-" * 78)
    for s in systems:
        cap = CAPS.get(s)
        counts = [p["returned"] for d in runs[s] for p in d["probes"]]
        at_cap = sum(1 for c in counts if cap is not None and c >= cap)
        below = len(counts) - at_cap
        print(f"{s:<14} {str(cap):<6} {f'{at_cap}/{len(counts)}':<20} {below}")

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

    always_on = []
    for s in systems:
        for data in runs[s]:
            for n in data.get("notes", []):
                if n.startswith("Core block characters reach the model"):
                    always_on.append((s, data["run"], n.rsplit(":", 1)[1].strip()))
    if always_on:
        print()
        print("=" * 78)
        print("CHARACTERS REACHING THE MODEL WITH NO SEARCH AT ALL")
        print("=" * 78)
        print("Text that is in the prompt on every turn regardless of what was asked.")
        print("Every other system in this comparison requires a search to surface anything.")
        print()
        for sysname, run, chars in always_on:
            print(f"  {sysname:<10} run {run}:  {chars} characters")

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
