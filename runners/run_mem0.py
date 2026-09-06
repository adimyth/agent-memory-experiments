# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mem0ai[nlp,extras]==2.0.20",
#   "sentence-transformers>=5.2.0",
#   "python-dotenv",
# ]
# ///

"""Run the fixed transcript through Mem0 OSS at its documented defaults.

Defaults, taken from the installed package rather than the docs:
``Memory.search(query, *, top_k=20, threshold=0.1, rerank=False, explain=False)``.
Nothing here is tuned. The whole point is to see what the defaults do.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import harness
import transcript
from harness import Hit, ProbeResult, RunResult

load_dotenv()


def build_memory(run: int, store_path: str):
    from mem0 import Memory

    config = {
        # temperature=0 everywhere. Extraction is still not fully deterministic, which
        # is why every system is run three times, but the model is not adding variance
        # on top of that. Note mem0 defaults to temperature 0.1 if this is omitted.
        "llm": {
            "provider": "openai",
            "config": {"model": harness.LLM_MODEL, "temperature": 0.0},
        },
        # Local embedder. bge-m3 is what Memory Weave uses, so every system in this
        # repo is embedded identically and the run needs no embedding API access.
        "embedder": {
            "provider": "huggingface",
            "config": {"model": harness.EMBED_MODEL, "embedding_dims": harness.EMBED_DIMS},
        },
        "vector_store": {
            "provider": "qdrant",
            # A fresh store per run. mem0's local Qdrant defaults to /tmp/qdrant and
            # persists the collection's vector width, so reusing it across runs silently
            # pins the first run's embedding dimensions.
            "config": {
                "collection_name": f"amx_run{run}",
                "path": store_path,
                "on_disk": False,
                "embedding_model_dims": harness.EMBED_DIMS,
            },
        },
    }
    return Memory.from_config(config), config


def dump_store(memory) -> list[dict]:
    rows = memory.get_all(filters={"user_id": transcript.USER_ID})
    results = rows.get("results", rows) if isinstance(rows, dict) else rows
    out = []
    for r in results:
        out.append(
            {
                "id": r.get("id"),
                "memory": r.get("memory"),
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
                "metadata": r.get("metadata"),
            }
        )
    return out


def active_signals() -> dict:
    """Which of Mem0 v3's three retrieval signals are actually available.

    A plain ``pip install mem0ai`` gives semantic search only. BM25 needs fastembed
    (``mem0ai[extras]``) and entity matching needs spaCy (``mem0ai[nlp]``). The
    migration docs describe the fused three-signal search as the v3 behaviour; the
    default install does not have it.
    """

    signals = {"semantic": True, "bm25": False, "entity": False}
    try:
        import fastembed  # noqa: F401

        signals["bm25"] = True
    except ImportError:
        pass
    try:
        import spacy  # noqa: F401

        signals["entity"] = True
    except ImportError:
        pass
    return signals


def probe(memory, p: transcript.Probe) -> ProbeResult:
    res = memory.search(
        p.text,
        filters={"user_id": transcript.USER_ID},
        explain=True,
    )
    results = res.get("results", res) if isinstance(res, dict) else res
    hits = []
    for r in results:
        hits.append(
            Hit(
                text=r.get("memory", ""),
                score=r.get("score"),
                id=r.get("id"),
                extra={k: v for k, v in r.items() if k in ("explain", "signals", "scores")},
            )
        )
    return ProbeResult(
        probe_id=p.id,
        query=p.text,
        kind=p.kind,
        expect_empty=p.expect_empty,
        returned=len(hits),
        hits=hits,
        note=p.note,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    store_dir = tempfile.mkdtemp(prefix=f"amx-mem0-run{args.run}-")
    memory, config = build_memory(args.run, store_dir)
    result = RunResult(
        system="mem0",
        run=args.run,
        started_at=harness.now(),
        config={
            "llm": harness.LLM_MODEL,
            "embedder": harness.EMBED_MODEL,
            "search_defaults": "top_k=20, threshold=0.1, rerank=False (package defaults, untouched)",
            "active_signals": active_signals(),
            "raw": config,
        },
        versions={"mem0ai": harness.pkg_version("mem0ai")},
    )

    # Session 1, 10 January.
    memory.add(transcript.SESSION_1.messages, user_id=transcript.USER_ID)
    result.store_after_session_1 = dump_store(memory)

    # The first September turn is searched against the January store.
    for p in transcript.PROBES_BY_STAGE["after_session_1"]:
        result.probes.append(probe(memory, p))

    # The update turn.
    memory.add(transcript.SESSION_2_UPDATE.messages, user_id=transcript.USER_ID)
    result.store_after_update = dump_store(memory)

    for p in transcript.PROBES_BY_STAGE["after_update"]:
        result.probes.append(probe(memory, p))

    result.finished_at = harness.now()
    path = harness.save(result)
    print(harness.summarize(result))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
