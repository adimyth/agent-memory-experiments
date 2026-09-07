# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["boto3", "python-dotenv"]
# ///

"""Run the fixed transcript through Amazon Bedrock AgentCore Memory.

AgentCore differs from every other system here in one structural way: **extraction is
asynchronous and server-side.** Nothing else in this comparison has a gap between
writing and being able to read. Events go in through `CreateEvent`, then enabled
strategies extract and consolidate in the background, and only then can
`RetrieveMemoryRecords` see anything. The runner polls rather than sleeping blindly,
and records how long the wait actually was, which is a number nothing else here has.

Three built-in strategies are enabled: semantic, summary, and user preference. Built-in
strategies are service-managed, so no execution role is needed and no model is invoked
in the caller's account.

The forty prior facts cannot be written verbatim. AgentCore has no raw-row API on this
path; everything arrives as a conversational event and the strategies decide what a
record is. That is the same limitation Graphiti has, and for the same reason: the
system stores what its extractor derived, not what you handed it.

Requires the setup in docs/agentcore-setup.md. Reads AMX_AWS_PROFILE, defaulting to
`personal-agentcore`, so a work account cannot be reached by accident.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import distractors
import harness
import transcript
from harness import Hit, ProbeResult, RunResult

load_dotenv()

PROFILE = os.environ.get("AMX_AWS_PROFILE", "personal-agentcore")
REGION = os.environ.get("AMX_AWS_REGION", "us-east-1")
ACTOR_ID = "aditya"
EXTRACTION_TIMEOUT_S = int(os.environ.get("AMX_EXTRACTION_TIMEOUT", "900"))


def clients():
    import boto3

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    return session.client("bedrock-agentcore-control"), session.client("bedrock-agentcore")


def create_memory(control, run: int) -> str:
    resp = control.create_memory(
        name=f"amx_run{run}_{int(time.time())}",
        description="agent-memory-experiments fixed transcript",
        eventExpiryDuration=7,
        memoryStrategies=[
            # Namespace templates, not literal paths. The summarization strategy
            # rejects a namespace without {sessionId}, because it writes one record
            # per session rather than one per actor.
            {"semanticMemoryStrategy": {"name": "FactExtractor", "namespaces": ["/facts/{actorId}"]}},
            {"summaryMemoryStrategy": {"name": "SessionSummarizer", "namespaces": ["/summaries/{actorId}/{sessionId}"]}},
            {"userPreferenceMemoryStrategy": {"name": "PreferenceLearner", "namespaces": ["/preferences/{actorId}"]}},
        ],
    )
    memory_id = resp["memory"]["id"]
    deadline = time.time() + 300
    while time.time() < deadline:
        status = control.get_memory(memoryId=memory_id)["memory"]["status"]
        if status == "ACTIVE":
            return memory_id
        if status == "FAILED":
            raise RuntimeError(f"memory creation failed: {resp}")
        time.sleep(5)
    raise TimeoutError("memory did not become ACTIVE")


def put_event(data, memory_id: str, session_id: str, text: str, when: datetime, role: str = "USER") -> None:
    data.create_event(
        memoryId=memory_id,
        actorId=ACTOR_ID,
        sessionId=session_id,
        eventTimestamp=when,
        payload=[{"conversational": {"content": {"text": text}, "role": role}}],
    )


def all_records(data, memory_id: str) -> list[dict]:
    rows, token = [], None
    while True:
        kwargs = {"memoryId": memory_id, "namespacePath": "/", "maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        resp = data.list_memory_records(**kwargs)
        for r in resp.get("memoryRecordSummaries", []):
            rows.append(
                {
                    "id": r.get("memoryRecordId"),
                    "memory": (r.get("content") or {}).get("text", ""),
                    "strategy": r.get("memoryStrategyId"),
                    "namespaces": r.get("namespaces"),
                    "created_at": str(r.get("createdAt")),
                }
            )
        token = resp.get("nextToken")
        if not token:
            return rows


# Extraction has no completion signal on this path, so the stopping condition is a
# record count that stops moving. A first version accepted two stable polls and stopped
# on a slow start, which looks identical to a finished extraction and produced a run
# with one record. Stability now needs four consecutive polls and a floor on elapsed
# time, so a late-starting strategy cannot be mistaken for a finished one.
STABLE_POLLS_REQUIRED = 4
MIN_WAIT_S = 120
POLL_INTERVAL_S = 15


def wait_for_extraction(data, memory_id: str, at_least: int, label: str) -> dict:
    """Poll until the background strategies have produced records and gone quiet."""

    started = time.time()
    last, stable = -1, 0
    while time.time() - started < EXTRACTION_TIMEOUT_S:
        count = len(all_records(data, memory_id))
        stable = stable + 1 if count == last else 0
        last = count
        if (
            stable >= STABLE_POLLS_REQUIRED
            and count > at_least
            and time.time() - started >= MIN_WAIT_S
        ):
            break
        time.sleep(POLL_INTERVAL_S)
    waited = round(time.time() - started, 1)
    print(f"  {label}: {last} records after {waited}s", flush=True)
    return {"stage": label, "records": last, "seconds": waited}


def probe(data, memory_id: str, p: transcript.Probe) -> ProbeResult:
    resp = data.retrieve_memory_records(
        memoryId=memory_id,
        namespacePath="/",
        searchCriteria={"searchQuery": p.text},
    )
    rows = resp.get("memoryRecordSummaries", [])
    hits = [
        Hit(
            text=(r.get("content") or {}).get("text", ""),
            score=r.get("score"),
            id=r.get("memoryRecordId"),
            extra={"strategy": r.get("memoryStrategyId"), "namespaces": r.get("namespaces")},
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



def run_indirect(probe_fn, store_texts: list[str], cap: int):
    """Indirect-association probes through this runner's own search path."""

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
                probe_id=ip.id, query=ip.text, target=ip.target,
                embedding_rank=ip.embedding_rank, returned=res.returned, cap=cap,
                in_store=key in blob, target_rank=rank, outranked_by=above[:5],
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="do not delete the memory afterwards")
    args = ap.parse_args()

    control, data = clients()
    result = RunResult(
        system="agentcore",
        run=args.run,
        started_at=harness.now(),
        config={
            "llm": "service-managed (built-in strategies invoke no model in this account)",
            "embedder": "service-managed",
            "strategies": ["semantic", "summary", "userPreference"],
            "search_defaults": "RetrieveMemoryRecords with no topK; service default",
            "region": REGION,
            "profile": PROFILE,
        },
        versions={"api": "bedrock-agentcore + bedrock-agentcore-control"},
    )
    result.notes.append(
        "Extraction is asynchronous and server-side. Unlike every other system here, "
        "there is a gap between writing and being able to read."
    )
    result.notes.append(
        "Prior facts cannot be written verbatim: everything arrives as a conversational "
        "event and the strategies decide what a record is, so store sizes are not "
        "comparable with the string-based systems."
    )

    memory_id = create_memory(control, args.run)
    print(f"  memory {memory_id} ACTIVE", flush=True)
    try:
        seed_session = f"amx-prior-{args.run}"
        for item in distractors.DISTRACTORS:
            put_event(data, memory_id, seed_session, item.text, transcript.SESSION_1_AT)

        jan_session = f"amx-jan-{args.run}"
        put_event(data, memory_id, jan_session, transcript.SESSION_1.messages[0]["content"], transcript.SESSION_1_AT)
        put_event(data, memory_id, jan_session, transcript.SESSION_1.messages[1]["content"], transcript.SESSION_1_AT, role="ASSISTANT")

        result.notes.append(str(wait_for_extraction(data, memory_id, 1, "after January")))
        result.store_after_session_1 = all_records(data, memory_id)

        for p in transcript.PROBES_BY_STAGE["after_session_1"]:
            result.probes.append(probe(data, memory_id, p))

        sep_session = f"amx-sep-{args.run}"
        put_event(data, memory_id, sep_session, transcript.SESSION_2_UPDATE.messages[0]["content"], transcript.SESSION_2_AT)
        put_event(data, memory_id, sep_session, transcript.SESSION_2_UPDATE.messages[1]["content"], transcript.SESSION_2_AT, role="ASSISTANT")

        before = len(result.store_after_session_1)
        result.notes.append(str(wait_for_extraction(data, memory_id, before, "after the update")))
        result.store_after_update = all_records(data, memory_id)

        for p in transcript.PROBES_BY_STAGE["after_update"]:
            result.probes.append(probe(data, memory_id, p))

        result.indirect = run_indirect(
            lambda pr: probe(data, memory_id, pr),
            [r.get("memory") or "" for r in result.store_after_update],
            10,
        )
    finally:
        if not args.keep:
            control.delete_memory(memoryId=memory_id)
            print(f"  deleted {memory_id}", flush=True)

    result.finished_at = harness.now()
    path = harness.save(result)
    print(harness.summarize(result))
    for n in result.notes:
        print("  note:", n)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
