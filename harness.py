"""Shared result schema and dumping.

Every runner emits the same JSON so the systems can be read side by side. The point
of the harness is the raw dumps, not a score: what is literally in the store after
each write, and what literally comes back on each probe.
"""

from __future__ import annotations

import json
import os

# The embedder is cached locally. Without this the Hub is contacted for a revision
# check on every load, which can hang for minutes behind a restrictive network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"

# One provider for every system, so the comparison is not confounded by model choice.
# One provider and one embedder for every system, so the comparison is not
# confounded by model choice. The embedder is local, which means the run needs no
# API access for embeddings and anyone can reproduce it.
LLM_MODEL = os.environ.get("EXPERIMENT_LLM", "gpt-4o")
EMBED_MODEL = os.environ.get("EXPERIMENT_EMBEDDER", "BAAI/bge-m3")
EMBED_DIMS = int(os.environ.get("EXPERIMENT_EMBED_DIMS", "1024"))


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


@dataclass
class Hit:
    """One row a search returned, normalized across libraries."""

    text: str
    score: float | None = None
    id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    probe_id: str
    query: str
    kind: str
    expect_empty: bool
    returned: int
    hits: list[Hit] = field(default_factory=list)
    note: str = ""


@dataclass
class IndirectResult:
    """Where the useful memory landed on a query that never names it."""

    probe_id: str
    query: str
    target: str
    embedding_rank: int
    returned: int
    cap: int | None
    in_store: bool
    target_rank: int | None          # None means it was not returned at all
    outranked_by: list[str] = field(default_factory=list)
    note: str = ""


def find_target_rank(hits: list[Hit], target: str) -> tuple[int | None, list[str]]:
    """Locate the target among returned rows and name whatever beat it.

    Matching is on a distinctive phrase rather than the whole string, because Graphiti
    and AgentCore re-extract rather than storing rows verbatim, so the exact sentence
    never appears in their stores.
    """

    key = _target_key(target)
    for i, h in enumerate(hits):
        if key in (h.text or "").lower():
            return i + 1, [x.text for x in hits[:i]]
    return None, [h.text for h in hits]


def _target_key(target: str) -> str:
    """The shortest distinctive phrase from a target fact."""

    keys = {
        "Secrets are stored": "secrets manager",
        "Migrations must be": "backwards compatible",
        "Type hints are": "type hint",
        "Line length is": "line length",
        "Deploys go out": "github actions",
        "The payments database": "postgres",
        "Dependency updates": "batched weekly",
        "The team writes runbooks": "runbook",
        "Code review requires": "one approval",
    }
    for prefix, key in keys.items():
        if target.startswith(prefix):
            return key
    return target.lower()[:24]


@dataclass
class RunResult:
    system: str
    run: int
    config: dict[str, Any]
    versions: dict[str, str]
    store_after_session_1: list[dict[str, Any]] = field(default_factory=list)
    store_after_update: list[dict[str, Any]] = field(default_factory=list)
    probes: list[ProbeResult] = field(default_factory=list)
    indirect: list[IndirectResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "run": self.run,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "versions": self.versions,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "store_after_session_1": self.store_after_session_1,
            "store_after_update": self.store_after_update,
            "probes": [asdict(p) for p in self.probes],
            "indirect": [asdict(p) for p in self.indirect],
            "notes": self.notes,
        }


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(result: RunResult) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{result.system}-run{result.run}.json"
    path.write_text(json.dumps(result.to_json(), indent=2, default=str) + "\n")
    return path


def summarize(result: RunResult) -> str:
    """One-screen summary printed after a run."""

    lines = [
        f"{result.system}  run {result.run}",
        f"  store after January: {len(result.store_after_session_1)} rows",
        f"  store after update:  {len(result.store_after_update)} rows",
        "",
    ]
    for p in result.probes:
        want = "empty" if p.expect_empty else "a hit"
        ok = (p.returned == 0) == p.expect_empty
        mark = "ok " if ok else "!! "
        lines.append(f"  {mark}[{p.kind:10}] {p.returned:>2} returned (want {want:6})  {p.query}")
        for h in p.hits[:5]:
            score = f"{h.score:.3f}" if isinstance(h.score, (int, float)) else "-"
            lines.append(f"        {score}  {h.text}")
    return "\n".join(lines)
