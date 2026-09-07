# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "letta-client",
#   "python-dotenv",
# ]
# ///

"""Run the fixed transcript through a self-hosted Letta server.

**This runs the memory blocks model, not MemFS.** Letta's current documentation says
"All Letta agents use MemFS" and describes a git-backed memory filesystem where files
under ``system/`` load into the system prompt every turn. The newest published server
image, ``letta/letta:0.16.8`` (14 May 2026), has no MemFS: `agents.create` takes
``memory_blocks`` and ``enable_sleeptime``, and a fresh agent's system prompt
describes "memory blocks and external memory". So MemFS is either Letta Cloud only or
not yet in a released server, and what a self-hoster can actually run is core blocks
plus archival plus recall. That gap is the finding; the rest of this file measures
what is there.

Letta's memory splits in a way nothing else in this comparison does, so it is probed
twice:

1. **Core blocks.** Embedded in the system prompt on every turn, with no search and no
   query. Whatever is in here reaches the model on the retry-count turn exactly as it
   reaches it on a question about testing. The other systems have no equivalent.
2. **Archival passages.** The searchable store, and the fair analogue of the other
   systems' stores. The forty prior facts go here verbatim, which is the same
   treatment Mem0 and LangMem get.

Requires a server:

    docker run -d --name amx-letta -p 8283:8283 -e OPENAI_API_KEY=... letta/letta:latest
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import distractors
import harness
import transcript
from harness import Hit, ProbeResult, RunResult

load_dotenv()

BASE_URL = os.environ.get("AMX_LETTA_URL", "http://localhost:8283")

# Letta takes an embedding_config with a custom endpoint, so it can be pointed at the
# same bge-m3 weights every other system here uses. `tools/embedding_shim.py` serves an
# OpenAI-shaped /v1/embeddings from the local model; only embeddings go there, the chat
# model still goes to the real API.
SHIM_URL = os.environ.get("AMX_SHIM_URL", "http://host.docker.internal:8399/v1")

EMBEDDING_CONFIG = {
    "embedding_endpoint_type": "openai",
    "embedding_endpoint": SHIM_URL,
    "embedding_model": harness.EMBED_MODEL,
    "embedding_dim": harness.EMBED_DIMS,
    "embedding_chunk_size": 300,
    "handle": "local/bge-m3",
}


def build_agent(client, run: int):
    return client.agents.create(
        name=f"amx-run{run}",
        model=f"openai/{harness.LLM_MODEL}",
        embedding_config=EMBEDDING_CONFIG,
        memory_blocks=[
            {"label": "human", "value": "The user is Aditya, a software engineer."},
            {"label": "persona", "value": "You are a coding assistant."},
        ],
    )


def seed_distractors(client, agent_id: str) -> None:
    """The same forty strings every other system gets, written verbatim to archival."""

    for item in distractors.DISTRACTORS:
        client.agents.passages.create(agent_id, text=item.text)


def dump_archival(client, agent_id: str) -> list[dict]:
    rows = client.agents.passages.list(agent_id, limit=500)
    return [{"id": p.id, "memory": p.text, "created_at": str(getattr(p, "created_at", ""))} for p in rows]


def dump_blocks(client, agent_id: str) -> list[dict]:
    return [
        {"label": b.label, "value": b.value, "limit": b.limit, "chars": len(b.value or "")}
        for b in client.agents.blocks.list(agent_id)
    ]


def probe(client, agent_id: str, p: transcript.Probe) -> ProbeResult:
    res = client.agents.passages.search(agent_id, query=p.text)
    rows = getattr(res, "results", res)
    hits = []
    for r in rows:
        text = getattr(r, "content", None) or getattr(r, "text", None) or str(r)
        hits.append(Hit(text=text, score=getattr(r, "score", None), id=getattr(r, "id", None)))
    return ProbeResult(
        probe_id=p.id,
        query=p.text,
        kind=p.kind,
        expect_empty=p.expect_empty,
        returned=len(hits),
        hits=hits,
        note=p.note,
    )


def send(client, agent_id: str, text: str) -> None:
    client.agents.messages.create(agent_id, messages=[{"role": "user", "content": text}])



def run_indirect(probe_fn, store_texts: list[str], cap: int) -> list["harness.IndirectResult"]:
    """Run the indirect-association probes through this runner's own search path.

    Reuses the runner's existing probe function so the search is identical to the main
    experiment; only the questions and the scoring differ.
    """

    from harness import IndirectResult, find_target_rank

    blob = " ".join(store_texts).lower()
    out = []
    for ip in transcript.INDIRECT_PROBES:
        shim = transcript.Probe(
            id=ip.id, text=ip.text, kind="indirect", note="", expect_empty=False
        )
        res = probe_fn(shim)
        rank, above = find_target_rank(res.hits, ip.target)
        key = harness._target_key(ip.target)
        out.append(
            IndirectResult(
                probe_id=ip.id,
                query=ip.text,
                target=ip.target,
                embedding_rank=ip.embedding_rank,
                returned=res.returned,
                cap=cap,
                in_store=key in blob,
                target_rank=rank,
                outranked_by=above[:5],
            )
        )
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    from letta_client import Letta

    client = Letta(base_url=BASE_URL)
    agent = build_agent(client, args.run)

    result = RunResult(
        system="letta",
        run=args.run,
        started_at=harness.now(),
        config={
            "llm": harness.LLM_MODEL,
            "embedder": harness.EMBED_MODEL,
            "embedder_note": f"Served locally to Letta via an OpenAI-shaped shim at {SHIM_URL}, so the weights match every other system.",
            "memory_model": "core blocks + archival + recall (server 0.16.8 has no MemFS)",
            "search_defaults": "passages.search, no documented top_k default",
            "server": BASE_URL,
        },
        versions={"letta-client": harness.pkg_version("letta-client"), "letta-server": "0.16.8"},
    )
    result.notes.append(
        "Letta's docs describe MemFS as universal; the newest published server image has "
        "memory_blocks and enable_sleeptime and no MemFS. This runs the blocks model."
    )

    try:
        seed_distractors(client, agent.id)
        result.notes.append(f"Seeded {len(distractors.DISTRACTORS)} archival passages verbatim.")

        send(client, agent.id, transcript.SESSION_1.messages[0]["content"])
        result.store_after_session_1 = dump_archival(client, agent.id)
        result.notes.append(f"Core blocks after January: {dump_blocks(client, agent.id)}")

        for p in transcript.PROBES_BY_STAGE["after_session_1"]:
            result.probes.append(probe(client, agent.id, p))

        send(client, agent.id, transcript.SESSION_2_UPDATE.messages[0]["content"])
        result.store_after_update = dump_archival(client, agent.id)
        blocks = dump_blocks(client, agent.id)
        result.notes.append(f"Core blocks after the update: {blocks}")
        result.notes.append(
            "Core block characters reach the model on every turn with no search: "
            + str(sum(b["chars"] for b in blocks))
        )

        for p in transcript.PROBES_BY_STAGE["after_update"]:
            result.probes.append(probe(client, agent.id, p))

        # Must run before the finally block deletes the agent.
        result.indirect = run_indirect(
            lambda p: probe(client, agent.id, p),
            [r.get("memory") or "" for r in result.store_after_update],
            5,
        )
    finally:
        client.agents.delete(agent.id)

    result.finished_at = harness.now()
    path = harness.save(result)
    print(harness.summarize(result))
    for n in result.notes:
        print("  note:", n)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
