# Robustness: how this experiment could be attacked, and what survives

Written adversarially against our own results, before publication. Each objection is one
a careful reader would raise. Where a check weakened a claim, the weakened
version is what stands.

## 1. "The target was missed only because top_k was too small. Raise it."

True, and it is the whole point. Measured on Mem0 with a 44-row store:

| top_k | Indirect targets found | Rows on `Bump the retry count to 3.` (want 0) |
| --- | --- | --- |
| 5 | 3 / 9 | 5 |
| 10 | 6 / 9 | 10 |
| 20 | 7 / 9 | 20 |
| 30 | 8 / 9 | 30 |
| 40 | **9 / 9** | **40** |
| 44 | 9 / 9 | 44 |

Recovering every useful memory requires returning the entire store on a turn where the
correct answer is nothing. There is no middle setting that gets both. The cap is not a
tuning knob, it is a trade between two failures.

This was the strongest objection and it makes the argument stronger rather than weaker.

## 2. "You chose the queries and you chose which fact should surface. That is your opinion."

Fair, so it was checked blind. `runners/validate_targets.py` shows a model the forty facts
in a fixed shuffled order and one query, with no mention of the experiment, no hint at the
intended answer, and asks which notes would change how the request is carried out.

**It selected the intended target for 9 of 9 probes.** It was the judge's first choice on
7 of 9, second on one, third on one.

Guard against the judge agreeing with the prompt: each query was re-run with the target removed. The
judge then picked different facts rather than insisting, so its judgments track the target
rather than the prompt.

**Caveat that belongs in the essay:** on two probes the judge's first choice was a
different fact that also legitimately applies ("Karthik reviews all database migrations"
for the column drop, "Alerting routes through PagerDuty" for the alert). More than one
memory can be useful for a request. The probe measures whether a system surfaced a fact a
blind reviewer called useful, not whether it found the single correct answer.

## 3. "This is an artefact of bge-m3."

Partly true, and this is the check that weakened a claim.

Re-ranked all nine probes with `all-MiniLM-L6-v2`, a different architecture and dimension:

| Query | bge-m3 | MiniLM-L6 |
| --- | --- | --- |
| Add the Stripe API key to the config. | 37/40 | 13/40 |
| Drop the legacy_status column. | 25/40 | **2/40** |
| Add a helper that parses the CSV. | 19/40 | **3/40** |
| Reformat this function. | 10/40 | 3/40 |
| Ship this fix straight to production. | 7/40 | 3/40 |
| Write a query to find duplicate charges. | 5/40 | 1/40 |
| Bump the requests library. | 3/40 | 4/40 |
| Add an alert for high latency. | 2/40 | 5/40 |
| Merge this PR. | 1/40 | 2/40 |

Rank correlation r = 0.657. **Two probes disagree sharply.** "Drop the legacy_status column"
is a hard indirect query under bge-m3 and an easy one under MiniLM.

So per-probe difficulty is substantially embedder-dependent, and the essay must not present
these nine as absolutely hard queries.

**What survives the swap.** Running Mem0 end to end on MiniLM instead:

| | bge-m3 | MiniLM-L6 |
| --- | --- | --- |
| Indirect targets found at top_k=20 | 7 / 9 | 7 / 9 |
| Rows on the ordinary turn | 20 | 20 |

The system-level numbers are identical. Which probes fail changes; how many fail, and the
fact that the ordinary turn still returns everything, does not.

**One probe is robustly hard: "Add the Stripe API key to the config."** It was missed under
both embedders. That is the example to lead with, and the only one to describe as
intrinsically difficult.

## 4. "The distractor store is synthetic and you wrote it to be confusable."

Half true and stated openly. The forty facts were written to be the same user's plausible
work, because a store of unrelated topics would flatter every system: on an ordinary turn
nothing would score highly and every system would look well behaved. The hard case is a
store full of things that look related, which is what production stores are.

They are all in `distractors.py`, verbatim, so anyone can disagree with the choice and
substitute their own.

## 5. "Nine probes and three runs is a small sample."

Correct. No significance is claimed and none should be. What is claimed is the direction
and size of the effect, which is large and monotonic: at the top of the similarity ranking
every system returns the target, at the bottom none do, with no exceptions in either
direction across three systems and three runs.

## 6. "Letta's cap of 5 makes it look worse than the others."

Correct, and caps are always reported alongside rank. Letta cannot return a rank-9 target
because it only ever returns five rows. Cross-system rank comparison is not made anywhere.

## 7. "Your matching could be counting a miss when the fact was returned in other words."

Matching is on a short distinctive phrase (`secrets manager`, `github actions`,
`backwards compatible`), not the whole sentence, because Graphiti and AgentCore
re-extract rather than storing rows verbatim. Every miss recorded so far was verified to
have `in_store: true`, meaning the phrase was present somewhere in that system's store and
the search still did not return it. **Zero misses were extraction failures.**

## What we would tell a critic

The headline claim is not "these systems are bad at retrieval". It is:

> Where a useful memory lands in the results is governed by how close its wording is to
> the query. On a store of one user's real work, that ordering has no relationship to
> whether the memory would improve the answer, and the only knob available trades one
> failure directly against the other.

That claim survives every check above. The per-probe difficulty numbers do not, and are
reported as bge-m3-specific.

## 8. Two claims the indirect run overturned

Both found by running the probes, not by a reader.

### "Retrieval rank tracks embedding similarity and nothing else"

Wrong. AgentCore breaks it, reproducibly across three runs:

| Similarity rank of target | AgentCore returned it at rank |
| --- | --- |
| 25 / 40 | 3, 2, 3 |
| 19 / 40 | 2, 2, 2 |
| 2 / 40 | 8, 8, 9 |

The ordering is partly inverted: a target that ranks 25th by
similarity comes back second or third, while one that ranks 2nd comes back eighth.

The mechanism is the redundancy criticised elsewhere in these notes. AgentCore stores the
same fact three ways: a semantic record, a preference record carrying `context`,
`preference` and `categories` JSON, and a line inside a topic-grouped session summary
under a heading such as "Engineering Standards and Conventions". Each is a different
surface for a query to match, and the re-wording bridges vocabulary the original sentence
did not contain.

So the duplication that looks like waste on the write side is buying recall on the read
side. That is a real trade and the earlier one-sided criticism of it was wrong.

It does not rescue AgentCore from the other failure: it still returns its full cap of ten
on every ordinary turn, in every run.

### "Graphiti abstains correctly"

Needs a heavy caveat. On the indirect probes Graphiti reports `absent` for eight of nine:
those facts never became edges at all. Only the Postgres one survived, because it is the
only target shaped entity-relation-entity. A rule like "Line length is capped at 100
characters" has no two entities to connect, so a graph store structurally cannot hold it.

That matters for the earlier finding that Graphiti's cross-encoder reranker abstains
correctly on ordinary turns. It does. But part of what looks like disciplined silence is
silence about something it never stored. Same output, different cause, and the essay must
separate the two.
