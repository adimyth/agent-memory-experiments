# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mem0ai[nlp,extras]==2.0.20",
#   "sentence-transformers>=5.2.0",
#   "python-dotenv",
# ]
# ///

"""Does Mem0's threshold work if you raise it off the default?

The main run uses documented defaults, where `threshold=0.1` never excludes anything.
That result is easy to misread as "the relevance score is useless". This sweep tests the
fairer question: with the same store and the same queries, is there a threshold that
keeps the answerable probes and drops the ordinary and absent ones?

If one exists, the finding is about the default being permissive, not about the ranking
being broken, and the essay has to say so.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import distractors
import harness
import transcript

load_dotenv()

THRESHOLDS = [0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]


def build(run: int, store_path: str):
    from mem0 import Memory

    return Memory.from_config(
        {
            "llm": {"provider": "openai", "config": {"model": harness.LLM_MODEL, "temperature": 0.0}},
            "embedder": {
                "provider": "huggingface",
                "config": {"model": harness.EMBED_MODEL, "embedding_dims": harness.EMBED_DIMS},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"amx_sweep{run}",
                    "path": store_path,
                    "on_disk": False,
                    "embedding_model_dims": harness.EMBED_DIMS,
                },
            },
        }
    )


def main() -> None:
    memory = build(1, tempfile.mkdtemp(prefix="amx-sweep-"))
    for fact in distractors.TEXTS:
        memory.add([{"role": "user", "content": fact}], user_id=transcript.USER_ID, infer=False)
    memory.add(transcript.SESSION_1.messages, user_id=transcript.USER_ID)
    memory.add(transcript.SESSION_2_UPDATE.messages, user_id=transcript.USER_ID)

    probes = transcript.PROBES_BY_STAGE["after_update"]
    rows = []
    print(f"{'threshold':<11}" + "".join(f"{p.kind[:9]:<11}" for p in probes) + "  verdict")
    print("-" * 78)
    for t in THRESHOLDS:
        counts, correct = [], True
        for p in probes:
            res = memory.search(p.text, filters={"user_id": transcript.USER_ID}, threshold=t)
            n = len(res.get("results", res) if isinstance(res, dict) else res)
            counts.append(n)
            if p.expect_empty and n > 0:
                correct = False
            if not p.expect_empty and n == 0:
                correct = False
        verdict = "ALL CORRECT" if correct else ""
        rows.append({"threshold": t, "counts": counts, "all_correct": correct})
        print(f"{t:<11}" + "".join(f"{c:<11}" for c in counts) + f"  {verdict}")

    print()
    print("Top score per probe at threshold 0.1 (i.e. unfiltered), same store:")
    for p in probes:
        res = memory.search(p.text, filters={"user_id": transcript.USER_ID}, threshold=0.1)
        hits = res.get("results", res) if isinstance(res, dict) else res
        top = hits[0] if hits else None
        want = "empty" if p.expect_empty else "a hit"
        if top:
            print(f"  want {want:<6} {top.get('score'):.3f}  {p.text[:30]:<32} -> {top.get('memory','')[:46]}")
    print()
    print("probe order: " + ", ".join(f"{p.kind}={p.text[:28]}" for p in probes))
    out = Path(__file__).parent.parent / "results" / "mem0-threshold-sweep.json"
    out.write_text(json.dumps({"probes": [p.id for p in probes], "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
