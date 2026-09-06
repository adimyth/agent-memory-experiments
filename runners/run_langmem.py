# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "langmem==0.0.30",
#   "langchain-openai",
#   "sentence-transformers==5.1.2",
#   "python-dotenv",
# ]
# ///

"""Run the fixed transcript through LangMem at its documented defaults.

Two things are being reproduced from the landscape document:

1. The background write path. ``create_memory_store_manager`` is an extractor the
   host runs after the chat; the chatting agent calls no tool. This is the closest
   analogue to Mem0's ``add``, which is why it is used here rather than the hot-path
   ``manage_memory`` tool.
2. The eager read path. The hot path quickstart searches the store with the last
   user message and interpolates the result into the system prompt. There is no
   gate and no score threshold: ``search_memory`` defaults to ``limit=10``.

Defaults taken from the installed package: ``enable_inserts=True``,
``enable_deletes=False``, namespace ``("memories", "{langgraph_user_id}")``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import harness
import transcript
from harness import Hit, ProbeResult, RunResult

load_dotenv()

NAMESPACE = ("memories", transcript.USER_ID)


def build_store():
    """An InMemoryStore indexed with the same local embedder every other system uses."""

    from langgraph.store.memory import InMemoryStore
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(harness.EMBED_MODEL)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    store = InMemoryStore(index={"dims": harness.EMBED_DIMS, "embed": embed})
    return store


def build_manager(store):
    from langchain_openai import ChatOpenAI
    from langmem import create_memory_store_manager

    llm = ChatOpenAI(model=harness.LLM_MODEL)
    return create_memory_store_manager(llm, namespace=NAMESPACE, store=store)


def dump_store(store) -> list[dict]:
    rows = store.search(NAMESPACE, limit=100)
    out = []
    for item in rows:
        out.append(
            {
                "namespace": list(item.namespace),
                "key": item.key,
                "value": item.value,
                "created_at": str(getattr(item, "created_at", "")),
                "updated_at": str(getattr(item, "updated_at", "")),
            }
        )
    return out


def probe(store, p: transcript.Probe) -> ProbeResult:
    # The eager path from the hot path quickstart: search with the turn's own text.
    rows = store.search(NAMESPACE, query=p.text)
    hits = [
        Hit(
            text=str(r.value.get("content", r.value)),
            score=getattr(r, "score", None),
            id=r.key,
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
        note=p.note,
    )


async def ingest(manager, store, write: transcript.Write) -> None:
    config = {"configurable": {"langgraph_user_id": transcript.USER_ID}}
    await manager.ainvoke({"messages": write.messages}, config=config)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")

    store = build_store()
    manager = build_manager(store)

    result = RunResult(
        system="langmem",
        run=args.run,
        started_at=harness.now(),
        config={
            "llm": harness.LLM_MODEL,
            "embedder": harness.EMBED_MODEL,
            "namespace": list(NAMESPACE),
            "write_path": "create_memory_store_manager (background extractor)",
            "read_path": "store.search(namespace, query=turn_text)  [eager, no threshold]",
            "search_defaults": "search_memory limit=10, no similarity threshold",
            "manager_defaults": "enable_inserts=True, enable_deletes=False",
        },
        versions={
            "langmem": harness.pkg_version("langmem"),
            "langgraph": harness.pkg_version("langgraph"),
        },
    )

    asyncio.run(ingest(manager, store, transcript.SESSION_1))
    result.store_after_session_1 = dump_store(store)

    for p in transcript.PROBES_BY_STAGE["after_session_1"]:
        result.probes.append(probe(store, p))

    asyncio.run(ingest(manager, store, transcript.SESSION_2_UPDATE))
    result.store_after_update = dump_store(store)

    for p in transcript.PROBES_BY_STAGE["after_update"]:
        result.probes.append(probe(store, p))

    result.finished_at = harness.now()
    path = harness.save(result)
    print(harness.summarize(result))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
