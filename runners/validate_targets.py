# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["openai", "python-dotenv"]
# ///

"""Independent check that the probe targets are not just the author's opinion.

The obvious objection to the indirect probes is that I chose both the queries and the
facts that "should" surface, so the result is my judgment dressed as a measurement.

This asks a model the same question blind. It sees the forty prior facts and one query,
with no hint about which fact is the intended answer and no mention of the experiment. It
is asked which facts, if any, would change how the request should be carried out.

Agreement makes the target assignment defensible. Disagreement is a finding about the
probe, and the probe should be dropped or restated rather than argued for.

Two guards against the judge simply agreeing with itself:

- The facts are presented in a fixed shuffled order, unrelated to the target's position,
  so the model cannot infer the answer from placement.
- A query is also run with the target fact removed from the list. If the model still
  claims a confident answer, its judgments are not tracking the target and should be
  discounted.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import distractors
import harness
import transcript

load_dotenv()

PROMPT = """You are reviewing a coding agent's stored notes about a team's conventions.

Here are the notes, numbered:

{facts}

The user has just said to the agent:

  "{query}"

Which of the numbered notes, if any, would change how the agent should carry out that
request? Consider notes that are relevant to the task even if they share no words with it.

Answer with a JSON object only: {{"indices": [<numbers>], "reasoning": "<one sentence>"}}
Use an empty list if no note would change how the request is carried out.
Return at most three indices, most important first."""


def ask(client, facts: list[str], query: str) -> dict:
    resp = client.chat.completions.create(
        model=harness.LLM_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)),
                    query=query,
                ),
            }
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def main() -> None:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    client = OpenAI()

    rng = random.Random(20260907)
    shuffled = list(distractors.TEXTS)
    rng.shuffle(shuffled)

    rows, agree = [], 0
    print(f"{'query':<40}{'judge picked':<40}{'agrees'}")
    print("-" * 90)
    for ip in transcript.INDIRECT_PROBES:
        out = ask(client, shuffled, ip.text)
        picks = [shuffled[i - 1] for i in out.get("indices", []) if 1 <= i <= len(shuffled)]
        key = harness._target_key(ip.target)
        hit = any(key in p.lower() for p in picks)
        rank = next((n + 1 for n, p in enumerate(picks) if key in p.lower()), None)
        agree += 1 if hit else 0

        # Ablation: same query with the target removed.
        without = [f for f in shuffled if key not in f.lower()]
        out2 = ask(client, without, ip.text)
        picks2 = [without[i - 1] for i in out2.get("indices", []) if 1 <= i <= len(without)]

        rows.append(
            {
                "probe_id": ip.id,
                "query": ip.text,
                "target": ip.target,
                "judge_picks": picks,
                "target_chosen": hit,
                "target_rank_in_picks": rank,
                "picks_without_target": picks2,
                "reasoning": out.get("reasoning", ""),
            }
        )
        print(f"{ip.text[:38]:<40}{(picks[0][:37] if picks else '(none)'):<40}{'yes' if hit else 'NO'}")

    print(f"\njudge chose the intended target for {agree} of {len(transcript.INDIRECT_PROBES)} probes")
    out_path = Path(__file__).parent.parent / "results" / "target-validation.json"
    out_path.write_text(json.dumps({"agreement": agree, "probes": rows}, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
