# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mem0ai[nlp,extras]==2.0.20",
#   "sentence-transformers>=5.2.0",
#   "python-dotenv",
# ]
# ///

"""The obvious rebuttal: "the indirect target was missed because top_k was too small".

True, and it is the whole point. This measures both sides at once. For each value of
top_k it asks:

- how many of the nine indirect targets come back, and
- how many rows come back on `Bump the retry count to 3.`, where the right answer is none.

If the top_k that recovers the useful memories is also the top_k that returns the entire
store on an ordinary turn, then the cap is not a tuning knob, it is a trade between two
failures. That is a much stronger claim than "it missed", and it is falsifiable: if some
middle value recovers most targets while keeping the ordinary turn small, the argument
weakens and the essay has to say so.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import distractors
import harness
import transcript

load_dotenv()

TOP_KS = [5, 10, 20, 30, 40, 44]


def build(path: str):
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
                    "collection_name": "amx_topk",
                    "path": path,
                    "on_disk": False,
                    "embedding_model_dims": harness.EMBED_DIMS,
                },
            },
        }
    )


def main() -> None:
    m = build(tempfile.mkdtemp(prefix="amx-topk-"))
    for fact in distractors.TEXTS:
        m.add([{"role": "user", "content": fact}], user_id=transcript.USER_ID, infer=False)
    m.add(transcript.SESSION_1.messages, user_id=transcript.USER_ID)
    m.add(transcript.SESSION_2_UPDATE.messages, user_id=transcript.USER_ID)

    ordinary = "Bump the retry count to 3."
    rows = []
    print(f"{'top_k':<8}{'indirect targets found':<26}{'rows on an ordinary turn':<26}")
    print("-" * 66)
    for k in TOP_KS:
        found = 0
        for ip in transcript.INDIRECT_PROBES:
            res = m.search(ip.text, filters={"user_id": transcript.USER_ID}, top_k=k)
            hits = res.get("results", res) if isinstance(res, dict) else res
            key = harness._target_key(ip.target)
            if any(key in (h.get("memory") or "").lower() for h in hits):
                found += 1
        res = m.search(ordinary, filters={"user_id": transcript.USER_ID}, top_k=k)
        noise = len(res.get("results", res) if isinstance(res, dict) else res)
        rows.append({"top_k": k, "targets_found": found, "ordinary_rows": noise})
        print(f"{k:<8}{f'{found}/9':<26}{f'{noise} (want 0)':<26}")

    out = Path(__file__).parent.parent / "results" / "mem0-topk-tradeoff.json"
    out.write_text(json.dumps({"rows": rows, "store_size": 44}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
