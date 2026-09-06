# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "memory-weave",
#   "sentence-transformers",
#   "python-dotenv",
# ]
#
# [tool.uv.sources]
# memory-weave = { path = "/Users/adimyth/PersonalProjects/agentic-memory-system", editable = true }
# ///

"""Run the fixed transcript through Memory Weave at its configured defaults.

Two things differ from the other runners and both are deliberate.

**Weave has no extractor.** Phase 10 of its implementation plan specifies a
session-end pass and it is not built, so nothing in the library reads a transcript
and decides what to store. The transcript facts here are written through
``memory_write`` with explicit payloads, which is the role the chatting agent plays
in Weave's design. That means the *write* side of this comparison is hand-driven for
Weave and model-driven for Mem0 and LangMem, and the row counts are not comparable.
The *retrieval* side, which is what the probes measure, is comparable: every system
is asked the same five questions against a store seeded with the same forty rows.

**Weave's search can return nothing.** That is the whole point of its gate, and it
is the only system here where an empty result is a designed outcome rather than a
failure. The floors are uncalibrated starting values, so this run measures what the
current configuration does, not what it is meant to do.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import distractors
import harness
import transcript
from harness import Hit, ProbeResult, RunResult

load_dotenv()

AGENT_ID = "amx-agent"
USER_ID = "user-aditya"


def build_runtime(db_path: Path):
    from memory_weave.config import MemoryWeaveConfig
    from memory_weave.host import MemoryHost
    from memory_weave.index.embedder import BgeM3Embedder
    from memory_weave.index.vector import VectorIndex
    from memory_weave.ingest import Ingestor, NLICrossEncoderJudge, SessionBuffer
    from memory_weave.models import Principal, Scope
    from memory_weave.retrieve import Retriever
    from memory_weave.store import Store
    from memory_weave.tools import ToolHandlers
    from memory_weave.util import now

    config = MemoryWeaveConfig()
    store = Store(str(db_path))
    host = MemoryHost(store)
    scope = Scope(kind="user", id=USER_ID)
    host.grant(AGENT_ID, scope, read=True, write=True)
    host.provision_user(USER_ID, aliases=("Aditya", "Aditya Mishra"))

    principal = Principal(AGENT_ID, USER_ID, "amx-session", None)
    store.create_session("amx-session", AGENT_ID, USER_ID, None, now())

    embedder = BgeM3Embedder(config.embedding)
    judge = NLICrossEncoderJudge(config.ingestion.equivalence)
    buffer = SessionBuffer(store)
    index = VectorIndex(config.embedding)
    ingestor = Ingestor(store, index, embedder, judge, buffer, config)
    retriever = Retriever(store, index, embedder, config)
    handlers = ToolHandlers(retriever, ingestor, store, index)
    return handlers, principal, store, config


def write(handlers, principal, **payload) -> dict:
    result = handlers.memory_write(principal, payload)
    if not result.get("ok"):
        print(f"  write refused: {result}", file=sys.stderr)
    return result


def append_turns(store, messages, *, start: int) -> None:
    """Persist the transcript so evidence quotes can be validated against it."""

    from memory_weave.models import Turn
    from memory_weave.util import now

    for offset, message in enumerate(messages):
        store.append_turn(
            Turn("amx-session", start + offset, message["role"], message["content"], now())
        )


def seed_distractors(handlers, principal) -> None:
    """The same forty strings every other system gets, written verbatim.

    ``agent_inference`` is the only source kind that does not demand an evidence
    quote, and these facts have no transcript to quote from. Weave marks such records
    provisional, which is its own rule about unsourced claims rather than anything
    this harness chose.
    """

    for item in distractors.DISTRACTORS:
        write(
            handlers,
            principal,
            type="semantic",
            content=item.text,
            attribute=item.attribute,
            source_kind="agent_inference",
            entities=[{"name": item.subject, "kind": item.kind, "role": "about"}],
        )


def dump_store(store) -> list[dict]:
    """Every record, including superseded ones.

    The store has no public "list everything" call, because nothing in the library
    needs one: retrieval always filters by scope and lifecycle first. The dump reads
    the ids straight out of SQLite and then goes back through ``get_records`` so the
    rows are real Record objects rather than raw columns.
    """

    ids = [r[0] for r in store.connection.execute("SELECT id FROM records ORDER BY created_at").fetchall()]
    rows = []
    for record in store.get_records(ids):
        rows.append(
            {
                "id": record.id,
                "memory": record.content,
                "type": record.type,
                "status": record.status,
                "attribute": record.attribute,
                "subject": record.subject,
                "source_kind": record.source_kind,
                "evidence": record.evidence,
                "supersedes_id": record.supersedes_id,
            }
        )
    return rows


def probe(handlers, principal, p: transcript.Probe) -> ProbeResult:
    result = handlers.memory_search(principal, {"queries": [p.text]})
    rows = result.get("results", []) if isinstance(result, dict) else []
    hits = [
        Hit(
            text=(r.get("record") or {}).get("content") or "",
            score=r.get("score"),
            id=(r.get("record") or {}).get("id"),
            extra={
                "attribute": (r.get("record") or {}).get("attribute"),
                "status": (r.get("record") or {}).get("status"),
                "explanation": r.get("explanation"),
            },
        )
        for r in rows
    ]
    return ProbeResult(
        probe_id=p.id,
        query=p.text,
        kind=p.kind,
        expect_empty=p.expect_empty,
        returned=len(hits),
        hits=hits,
        note=(p.note + f"  empty_reason={result.get('empty_reason')!r}"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    db_dir = Path(tempfile.mkdtemp(prefix=f"amx-weave-run{args.run}-"))
    handlers, principal, store, config = build_runtime(db_dir / "weave.sqlite")

    result = RunResult(
        system="weave",
        run=args.run,
        started_at=harness.now(),
        config={
            "llm": "none (no extractor; writes are explicit memory_write calls)",
            "embedder": config.embedding.model,
            "search_defaults": f"k=8 default, gate floors from config (uncalibrated starting values)",
            "trigger_mode": config.retrieval.trigger.mode,
            "gate": str(config.retrieval.gate),
        },
        versions={"memory-weave": "local checkout"},
    )
    result.notes.append(
        "Weave has no session extractor (Phase 10 unbuilt). Transcript facts are written "
        "through memory_write by hand, playing the role the chatting agent plays in its "
        "design. Write-side row counts are therefore not comparable with the other systems."
    )

    seed_distractors(handlers, principal)
    result.notes.append(f"Seeded {len(distractors.DISTRACTORS)} distractor rows.")

    # The transcript has to be in the session buffer before the writes, or evidence
    # validation cannot find the quoted text and every record is downgraded to
    # provisional with "evidence not found in session".
    append_turns(store, transcript.SESSION_1.messages, start=1)

    # Session 1, 10 January. Three facts, each with its evidence quote.
    jan = transcript.SESSION_1.messages[0]["content"]
    aditya = [{"name": "Aditya", "kind": "person", "role": "about"}]
    write(handlers, principal, type="semantic", content="Aditya writes tests in pytest",
          attribute="test_runner", source_kind="user_statement",
          evidence="I write tests in pytest.", entities=aditya)
    write(handlers, principal, type="semantic", content="Aditya keeps commit messages conventional",
          attribute="commit_style", source_kind="user_statement",
          evidence="Commit messages stay conventional.", entities=aditya)
    write(handlers, principal, type="semantic", content="Rohan works at Nimbus",
          attribute="employer", source_kind="user_statement",
          evidence="Rohan is on-call; he works at Nimbus, on payments.",
          entities=[{"name": "Rohan", "kind": "person", "role": "about"}])
    result.store_after_session_1 = dump_store(store)

    for p in transcript.PROBES_BY_STAGE["after_session_1"]:
        result.probes.append(probe(handlers, principal, p))

    append_turns(store, transcript.SESSION_2_UPDATE.messages, start=3)

    # The September update. Same current-fact keys, so these should supersede.
    write(handlers, principal, type="semantic", content="Aditya writes tests in unittest",
          attribute="test_runner", source_kind="user_statement",
          evidence="I switched to unittest.", entities=aditya)
    write(handlers, principal, type="semantic", content="Rohan works at Lattice",
          attribute="employer", source_kind="user_statement",
          evidence="Rohan left Nimbus in April and joined Lattice.",
          entities=[{"name": "Rohan", "kind": "person", "role": "about"}])
    result.store_after_update = dump_store(store)

    for p in transcript.PROBES_BY_STAGE["after_update"]:
        result.probes.append(probe(handlers, principal, p))

    result.finished_at = harness.now()
    path = harness.save(result)
    print(harness.summarize(result))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
