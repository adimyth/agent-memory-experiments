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

"""Does Graphiti abstain if you raise its reranker floor?

The main run uses `Graphiti.search()`, which takes no score floor and returns its full
`num_results` every time. That is easy to misread as "Graphiti has no relevance floor".
It has two, and the convenience method just does not expose them:

- `sim_min_score`, default **0.6**, applied per search method at the candidate stage.
- `reranker_min_score`, default **0**, applied after the cross-encoder reranks.

The second is the abstention knob, and zero is why nothing is ever excluded. This sweep
raises it and asks the fair question: is there a setting that keeps the answerable probes
and drops the ordinary and absent ones?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import distractors
import harness
import transcript

load_dotenv()

FALKOR_PORT = int(os.environ.get("AMX_FALKOR_PORT", "6389"))
FLOORS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def build():
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client import LLMConfig, OpenAIClient
    from sentence_transformers import SentenceTransformer

    class LocalEmbedder(EmbedderClient):
        def __init__(self):
            self._m = SentenceTransformer(harness.EMBED_MODEL)

        async def create(self, input_data):
            texts = input_data if isinstance(input_data, list) else [input_data]
            return self._m.encode(texts, normalize_embeddings=True).tolist()[0]

        async def create_batch(self, input_data_list):
            return self._m.encode(input_data_list, normalize_embeddings=True).tolist()

    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

    driver = FalkorDriver(host="localhost", port=FALKOR_PORT, database="amx_sweep")
    cfg = LLMConfig(model=harness.LLM_MODEL, small_model=harness.LLM_MODEL, temperature=0.0)
    llm = OpenAIClient(config=cfg)
    # The cross-encoder reranker is a separate client and defaults to gpt-4.1-nano.
    # It has to be pinned too or the cross-encoder recipe fails on model access.
    return Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=LocalEmbedder(),
        cross_encoder=OpenAIRerankerClient(config=cfg),
    )


async def main() -> None:
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_CROSS_ENCODER,
        EDGE_HYBRID_SEARCH_RRF,
    )

    client = build()
    try:
        await client.build_indices_and_constraints()
        for i, item in enumerate(distractors.DISTRACTORS):
            await client.add_episode(
                name=f"prior-{i:02d}",
                episode_body=item.text,
                source_description="prior knowledge",
                reference_time=transcript.SESSION_1_AT,
            )
        for name, msg, when in (
            ("session-1", transcript.SESSION_1.messages[0]["content"], transcript.SESSION_1_AT),
            ("session-2", transcript.SESSION_2_UPDATE.messages[0]["content"], transcript.SESSION_2_AT),
        ):
            await client.add_episode(
                name=name, episode_body=msg, source_description="session", reference_time=when
            )

        probes = transcript.PROBES_BY_STAGE["after_update"]
        rows = []
        # RRF is what Graphiti.search() uses. Its score is derived from rank, so a floor
        # on it cannot express "nothing here is good enough". The cross-encoder recipe
        # scores relevance directly, so it is the fair second test.
        # A bare SearchConfig leaves edge_config None and searches nothing, silently
        # returning zero at every floor; both recipes below are copied, not constructed.
        for label, recipe in (("RRF (the default)", EDGE_HYBRID_SEARCH_RRF),
                              ("cross-encoder", EDGE_HYBRID_SEARCH_CROSS_ENCODER)):
            print()
            print(f"=== reranker: {label} ===")
            print(f"{'min_score':<12}" + "".join(f"{p.kind[:9]:<11}" for p in probes) + "  verdict")
            print("-" * 78)
            for floor in FLOORS:
                counts, correct = [], True
                for p in probes:
                    cfg = recipe.model_copy(deep=True)
                    cfg.limit = 10
                    cfg.reranker_min_score = floor
                    res = await client.search_(p.text, config=cfg)
                    n = len(res.edges)
                    counts.append(n)
                    if p.expect_empty and n > 0:
                        correct = False
                    if not p.expect_empty and n == 0:
                        correct = False
                rows.append({"reranker": label, "reranker_min_score": floor,
                             "counts": counts, "all_correct": correct})
                print(f"{floor:<12}" + "".join(f"{c:<11}" for c in counts)
                      + f"  {'ALL CORRECT' if correct else ''}")

        print()
        print("probe order: " + ", ".join(f"{p.kind}={p.text[:26]}" for p in probes))
        out = Path(__file__).parent.parent / "results" / "graphiti-threshold-sweep.json"
        out.write_text(json.dumps({"probes": [p.id for p in probes], "rows": rows}, indent=2) + "\n")
        print(f"\nwrote {out}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
