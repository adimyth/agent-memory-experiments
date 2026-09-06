# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "graphiti-core[falkordb]==0.30.1",
#   "httpx",
#   "openai",
#   "sentence-transformers",
#   "python-dotenv",
# ]
# ///

"""Run the fixed transcript through Graphiti, backed by FalkorDB.

Graphiti is the only system here that cannot be given the shared prior store
verbatim. It does not hold strings; it holds entities and edges, and every episode
goes through an LLM extractor that decides what the triples are. So its distractor
set is whatever its own extractor derived from the same forty sentences, which is a
weaker guarantee than the other runners get. That difference is a property of the
data model, not a shortcut in the harness, and it is recorded in the result file.

It is also the only system here that can be asked what was true on a date. The last
probe is run twice: once as an ordinary search, and once with a ``valid_at`` filter
pinned to March 2026, which is the query the whole temporal-graph family exists for.

Requires FalkorDB:

    docker run -d --name amx-falkordb -p 6389:6379 falkordb/falkordb:latest
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import distractors
import harness
import transcript
from harness import Hit, ProbeResult, RunResult

load_dotenv()

FALKOR_PORT = int(os.environ.get("AMX_FALKOR_PORT", "6389"))
MARCH_2026 = datetime(2026, 3, 15, tzinfo=timezone.utc)


from graphiti_core.embedder.client import EmbedderClient


class LocalEmbedder(EmbedderClient):
    """bge-m3 through Graphiti's embedder interface, so every system embeds alike."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(harness.EMBED_MODEL)

    async def create(self, input_data):
        texts = input_data if isinstance(input_data, list) else [input_data]
        return self._model.encode(texts, normalize_embeddings=True).tolist()[0]

    async def create_batch(self, input_data_list):
        return self._model.encode(input_data_list, normalize_embeddings=True).tolist()


def build_client(run: int):
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.llm_client import LLMConfig, OpenAIClient

    driver = FalkorDriver(host="localhost", port=FALKOR_PORT, database=f"amx_run{run}")
    # Graphiti routes some steps (edge resolution, dedup) to a separate "small model"
    # that defaults to gpt-4.1-nano. Both are pinned to the harness model so the whole
    # pipeline runs on one LLM, like every other system here.
    llm = OpenAIClient(
        config=LLMConfig(model=harness.LLM_MODEL, small_model=harness.LLM_MODEL, temperature=0.0)
    )
    return Graphiti(graph_driver=driver, llm_client=llm, embedder=LocalEmbedder())


async def dump_edges(client) -> list[dict]:
    """Every entity edge in the graph, with its validity window."""

    from graphiti_core.edges import EntityEdge

    edges = await EntityEdge.get_by_group_ids(client.driver, ["_"], limit=500)
    return [
        {
            "id": e.uuid,
            "memory": e.fact,
            "name": e.name,
            "valid_at": str(e.valid_at) if e.valid_at else None,
            "invalid_at": str(e.invalid_at) if e.invalid_at else None,
            "expired_at": str(e.expired_at) if e.expired_at else None,
            "created_at": str(e.created_at),
        }
        for e in edges
    ]


def to_hits(edges) -> list[Hit]:
    return [
        Hit(
            text=e.fact,
            score=None,
            id=e.uuid,
            extra={
                "valid_at": str(e.valid_at) if e.valid_at else None,
                "invalid_at": str(e.invalid_at) if e.invalid_at else None,
            },
        )
        for e in edges
    ]


async def probe(client, p: transcript.Probe) -> ProbeResult:
    edges = await client.search(p.text)
    return ProbeResult(
        probe_id=p.id,
        query=p.text,
        kind=p.kind,
        expect_empty=p.expect_empty,
        returned=len(edges),
        hits=to_hits(edges),
        note=p.note + "  (search defaults to num_results=10)",
    )


async def probe_as_of_march(client) -> ProbeResult:
    """The query only a temporal graph can answer: what held in March 2026."""

    from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

    filters = SearchFilters(
        valid_at=[[DateFilter(date=MARCH_2026, comparison_operator=ComparisonOperator.less_than_equal)]],
        invalid_at=[[DateFilter(date=MARCH_2026, comparison_operator=ComparisonOperator.greater_than)]],
    )
    edges = await client.search("Where was Rohan working?", search_filter=filters)
    return ProbeResult(
        probe_id="p6_march_as_of",
        query="Where was Rohan working? [valid_at <= 2026-03-15 < invalid_at]",
        kind="answerable",
        expect_empty=False,
        returned=len(edges),
        hits=to_hits(edges),
        note="As-of query. Nothing else in this comparison can express it.",
    )


async def run(args) -> None:
    client = build_client(args.run)
    result = RunResult(
        system="graphiti",
        run=args.run,
        started_at=harness.now(),
        config={
            "llm": harness.LLM_MODEL,
            "embedder": harness.EMBED_MODEL,
            "search_defaults": "num_results=10; hybrid over vector, BM25, and graph traversal",
            "backend": f"FalkorDB on port {FALKOR_PORT}",
            "cross_encoder": "library default (OpenAIRerankerClient)",
            "small_model": harness.LLM_MODEL,
        },
        versions={"graphiti-core": harness.pkg_version("graphiti-core")},
    )
    result.notes.append(
        "Graphiti cannot take rows verbatim: every episode goes through its own LLM "
        "entity and edge extractor. Its distractor set is what that extractor derived "
        "from the same forty sentences, so store sizes are not comparable with the "
        "string-based systems."
    )

    try:
        await client.build_indices_and_constraints()

        for i, item in enumerate(distractors.DISTRACTORS):
            await client.add_episode(
                name=f"prior-{i:02d}",
                episode_body=item.text,
                source_description="prior knowledge",
                reference_time=transcript.SESSION_1_AT,
            )

        await client.add_episode(
            name="session-1",
            episode_body=transcript.SESSION_1.messages[0]["content"],
            source_description="January session",
            reference_time=transcript.SESSION_1_AT,
        )
        result.store_after_session_1 = await dump_edges(client)

        for p in transcript.PROBES_BY_STAGE["after_session_1"]:
            result.probes.append(await probe(client, p))

        await client.add_episode(
            name="session-2-update",
            episode_body=transcript.SESSION_2_UPDATE.messages[0]["content"],
            source_description="September session",
            reference_time=transcript.SESSION_2_AT,
        )
        result.store_after_update = await dump_edges(client)

        for p in transcript.PROBES_BY_STAGE["after_update"]:
            result.probes.append(await probe(client, p))

        result.probes.append(await probe_as_of_march(client))
    finally:
        await client.close()

    result.finished_at = harness.now()
    path = harness.save(result)
    print(harness.summarize(result))
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
