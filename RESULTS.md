# Results

Four systems, three runs each, 6 September 2026. Same transcript, same LLM
(`gpt-4o` at `temperature=0`), same local embedder (`BAAI/bge-m3`). Every system at
its documented defaults, nothing tuned.

Each store is seeded with the same forty prior facts before the transcript, so the
stores hold roughly forty-five rows rather than four. Without that, "the search
returned every row" is true but meaningless against a default result cap of 20.

Raw dumps are in `results/`. Regenerate with `uv run --script compare.py`.

## The headline

The four systems have different default caps on a search result, so raw row counts
are not comparable. The comparable question is whether **anything other than the cap
ever kept a row out**.

| System | Cap | Probes returning the cap | Probes returning fewer |
| --- | --- | --- | --- |
| LangMem | 10 | 15 of 15 | 0 |
| Mem0 | 20 | 15 of 15 | 0 |
| Graphiti | 10 | 15 of 18 | 3 |
| Memory Weave | 8 | 6 of 15 | 9 |

For AgentCore, Mem0, LangMem, and Letta the cap is the only thing limiting output. Not once, on any
probe, in any run, did a threshold or a floor exclude a row. That includes
`What is my dog's name?`, where nothing in the store is about a dog and an empty
result is the only correct answer. Mem0 applies a similarity threshold of `0.1` by
default; LangGraph's `BaseStore`, which LangMem builds on, applies none at all.
Neither is high enough to bind on anything.

Graphiti's three exceptions are the as-of query described below, one per run. Its
ordinary search returns the cap like the others.

### Letta does not need a search to leak

Letta is the only system here that puts memory in the prompt with no query at all. Its
core memory blocks are compiled into the system prompt on every turn:

| Run | Characters in the prompt on every turn |
| --- | --- |
| 1 | 195 |
| 2 | 185 |
| 3 | 200 |

After the update turn, run 1's human block read:

> The user is Aditya, a software engineer.
> Aditya now writes tests in unittest and keeps commit messages conventional.
> Rohan joined Lattice after leaving Nimbus in April.

That text reaches the model on `Bump the retry count to 3.` exactly as it reaches it on
a question about testing. There is no threshold to tune and no gate to pass, because
there is no retrieval step to gate. Every other system in this comparison has to be
asked before it says anything; Letta has already spoken.

It is doubly exposed, because the archival search runs on top of that and returns its
full cap of five on every probe as well.

The agent also rewrote the block in place. The January wording ("Aditya writes tests in
pytest") is gone, with no lineage, in all three runs. Same overwrite semantics LangMem
showed on a small store, reached by a different mechanism.

Memory Weave returned fewer than its cap on nine probes of fifteen. Its gate is the
only mechanism in this comparison that ever decides to stop early. It is also not yet
doing the job it is meant to: on the two ordinary turns it returned a full eight rows
every time, so the case it was built for is the case it currently fails. Its floors
are uncalibrated starting values and this measures the current configuration, not the
design.

## The one query only a temporal graph can answer

`Where was Rohan working in March?` Rohan left Nimbus in April, so the answer is
Nimbus, and every store's current value is Lattice.

Asked as an ordinary search, every system hands back its cap and lets the model sort
it out. Graphiti can be asked differently:

```python
SearchFilters(
    valid_at=[[DateFilter(date=MARCH_2026, comparison_operator=less_than_equal)]],
    invalid_at=[[DateFilter(date=MARCH_2026, comparison_operator=greater_than)]],
)
```

That returns exactly one row, identically in all three runs:
`Rohan works at Nimbus on payments.`

It works because Graphiti read "Rohan left Nimbus in April and joined Lattice" and
closed the interval on the old edge:

| Fact | valid_at | invalid_at |
| --- | --- | --- |
| Rohan works at Nimbus on payments. | 2026-01-10 | 2026-04-01 |
| Rohan joined Lattice. | 2026-04-01 | none |

It inferred April from natural language and wrote it as a boundary. No other system
here recorded when the old fact stopped being true, so no other system can be asked
this question at all. They can only be handed everything and asked to reason.

## What the update turn did

Mem0 never edits and never deletes, in all three runs: two rows in, two new rows out.
That is v3's specified ADD-only extraction and it is perfectly consistent. Afterwards
the store holds both the pytest fact and the unittest fact. Ranking, not the store,
decides which one the model sees.

LangMem is more interesting, because its behaviour changes with the size of the store.
Run both variants and only one thing differs: whether the forty prior facts were seeded.
Same code, same transcript, same model, same temperature.

| Store | Runs | Rows | Edited in place | Added |
| --- | --- | --- | --- | --- |
| Transcript only | 3 of 3 | 3 to 3 | 2 | 0 |
| Plus 40 prior facts | 3 of 3 | 42-43 to 44-45 | 0 | 2 |

On a small store the background manager **overwrote in place** at the same UUIDs. The
row reading `User writes tests in pytest, indicating proficiency with this testing
framework` became `User prefers working with tests written in the unittest framework`.
Same key, January wording gone, no lineage.

On a large store the same code **appended**, leaving the pytest row and the unittest row
both live at different keys.

The cause is almost certainly `create_memory_store_manager`'s `query_limit`, which
defaults to 5: the manager only sees the five most similar existing memories when it
decides whether to update or insert. Once the store is big enough that the pytest row
falls outside that window, the manager cannot update what it never retrieved, so it
inserts a second one.

That is a scaling property worth knowing about, and it is unanimous in both directions
rather than a tendency. A memory layer that consolidates correctly in a demo can quietly
degrade to append-only in production, and nothing in the output says it happened.

Reproduce with `uv run --script runners/run_langmem.py --run 1 --no-distractors`.

## A caution about text matching

An obvious way to check whether the old fact survived is to grep the store for
`pytest`. It gives the wrong answer for LangMem in the small-store case, because the
*replacement* text reads "a shift from pytest". The old row is gone and the word is
still there. Any comparison of these systems has to track row identity, not strings.
The comparison script does; an earlier version of it did not, and reported the
opposite conclusion.

## Mem0's hybrid search needs its extras

A plain `pip install mem0ai` gives semantic search only:

```
fastembed not installed - BM25 keyword search disabled
Failed to load spaCy lemma model: spaCy is not installed
```

Mem0 v3's documented "multi-signal hybrid search (semantic + BM25 keyword + entity
matching)" needs `mem0ai[nlp]` and `mem0ai[extras]`. Everything here ran with all
three signals live, recorded in each result file under `config.active_signals`.

Turning the other two on visibly sharpened the ranking. On the March question the gap
between the two Rohan rows and the unrelated rows widened from about 0.3 to about
0.4. It changed which row ranked first. It did not change how many rows came back:
still the full cap.

## Extraction granularity is the library's decision, not the transcript's

From the identical January paragraph, with the identical model:

- Mem0 wrote **2** rows, every run, merging pytest and conventional commits into one.
- LangMem wrote **3** rows, every run, splitting them.
- Graphiti wrote **12** edges from 41 episodes, discarding most preference-style facts
  entirely because they do not fit an entity-relationship-entity shape.
- Letta wrote **0** archival rows from the transcript, because its agent chose to put
  those facts in the core block instead. Its archival store holds only the forty seeded
  facts, which is why its answerable probe returns five unrelated rows: the answer is
  real and it is in the system prompt, not in the searchable store.

Each is internally consistent across runs. They simply disagree about what a fact is.
Any row count quoted for these systems is a property of the extraction prompt, not of
the conversation.

## Two things this harness found in Memory Weave

Both reproduce in isolation, outside this harness, with two writes and no distractors.

**The supplied attribute is ignored.** Two semantic writes about the same subject with
different explicit attributes (`test_runner` and `commit_style`) both land under
`test_runner`, and the second supersedes the first:

```
write attr='test_runner'  -> created
write attr='commit_style' -> superseded:01a07728-19f3...
  stored: attr='test_runner' status=superseded  :: Aditya writes tests in pytest
  stored: attr='test_runner' status=provisional :: Aditya keeps commit messages conventional
```

Two unrelated facts about one person cannot both be live. That contradicts the
documented contract, where a record's identity is `entity + attribute` and one live
record exists per key.

**Supersession is inconsistent in the other direction too.** In the same run where
Aditya's two unrelated facts collapsed into one chain, Rohan's genuine employer change
did *not* supersede: `Rohan works at Nimbus` and `Rohan works at Lattice` were both
left live. So the same mechanism merges what should stay separate and separates what
should merge.

Neither is a finding about the harness. Both are recorded here because running the
thing is how they surfaced.


## Letta's documentation and Letta's server disagree

Letta's current documentation says "All Letta agents use MemFS" and describes a
git-backed memory filesystem where files under `system/` load into the system prompt
every turn.

`letta/letta:0.16.8`, published 14 May 2026 and the newest image on Docker Hub, has no
MemFS. `agents.create` takes `memory_blocks` and `enable_sleeptime`. A fresh agent's
system prompt says "Your memory consists of memory blocks and external memory". The
1.12.1 client's agent API exposes `blocks`, `archives`, `passages`, and `folders`, and
nothing filesystem-shaped.

So MemFS is either Letta Cloud only or not yet in a released server, and a self-hoster
runs the core-blocks model. This experiment runs what exists.

One more correction while here: the block character limit. Letta's docs examples use
5000. A block created by the 0.16.8 server defaults to a limit of **100,000**.

## A note on Letta's embedder

Letta is the one system whose server owns its embedding pipeline, and the only OpenAI
key available for this work has no embedding model access. Rather than let Letta run on
a different embedder and quietly confound the comparison, `tools/embedding_shim.py`
serves an OpenAI-shaped `/v1/embeddings` from the same local `bge-m3` weights, and the
agent's `embedding_config` points at it. Letta's chat model still goes to the real API.
Every system in this repo is therefore on identical embedding weights, not merely the
same model name.


## AgentCore multiplies where the others append or overwrite

Three built-in strategies were enabled: semantic, summary, and user preference. From
roughly 43 input sentences they produced 78 to 87 records across three runs.

| Strategy | Records (3 runs) |
| --- | --- |
| FactExtractor (semantic) | 45, 45, 46 |
| PreferenceLearner | 26, 30, 36 |
| SessionSummarizer | 6, 6, 6 |

The same claim about pytest exists three times over: as a semantic fact, as a
preference record carrying its own `context` and `categories` JSON, and inside a
session summary. Each strategy consolidates only against its own records, so enabling
a strategy multiplies the store rather than enriching it.

Within a strategy, nothing supersedes. In all three runs `/facts/aditya/` held both
claims afterwards:

```
/facts/aditya/  "The user writes tests in pytest."
/facts/aditya/  "The user switched to unittest."
```

The employer change behaved the same way. So AgentCore sits with Mem0 on this axis
rather than with a system that supersedes: the store accumulates and ranking decides
which claim surfaces. Unlike Mem0, it accumulates in parallel across every enabled
strategy at once.

## AgentCore is the only system with a gap between writing and reading

Extraction is asynchronous and server-side. Every other system here can be searched the
moment the write call returns; AgentCore cannot.

| Run | After January | After the update |
| --- | --- | --- |
| 1 | 74 records after 155.0s | 78 records after 154.5s |
| 2 | 84 records after 124.7s | 87 records after 139.0s |
| 3 | 78 records after 154.8s | 81 records after 123.6s |

The runner polls until the record count stops moving rather than sleeping blindly.
There is no completion signal on this path, which is worth knowing before designing
around it: a first version of the poll accepted two stable counts and stopped on a slow
start, recording a run with one record. Four consecutive stable polls and a floor on
elapsed time were needed to tell a late-starting strategy from a finished one.

That gap is the price of keeping extraction off the message path. The store is briefly,
knowably wrong, and nothing in the API tells you when it stops being wrong.

## Correction: Mem0's ranking works. Its threshold is a different number.

An earlier draft of these notes said there was "no relevance check doing any work". That
is wrong and unfair, and the sweep in `runners/sweep_mem0_threshold.py` is what corrects it.

**The ranking discriminates correctly.** Top score per probe, same store, unfiltered:

| Probe | Want | Top score |
| --- | --- | --- |
| Where was Rohan working in March? | a hit | 0.863 |
| What do I write tests in? | a hit | 0.800 |
| Bump the retry count to 3. | empty | 0.637 |
| What is my dog's name? | empty | 0.461 |

The two questions the store can answer score above the two it should decline. The order
is right. Nothing is broken about the scoring.

**The default threshold is simply far below all of it.** `0.1` sits under every score in
that table, so it never excludes anything. That part of the original finding stands.

**Raising it helps, but no setting gets all four right:**

| threshold | ordinary | answerable | absent | answerable (dated) |
| --- | --- | --- | --- | --- |
| 0.1 | 20 | 20 | 20 | 20 |
| 0.3 | 20 | 20 | 20 | 20 |
| 0.5 | 15 | 3 | **0** | 2 |
| 0.6 | 1 | 1 | 0 | 2 |
| 0.65 | 0 | **0** | 0 | 1 |
| 0.7 | 0 | 0 | 0 | 0 |

At `0.5` the dog question correctly returns nothing, which the default never manages. But
the ordinary turn still returns fifteen rows. By the time the ordinary turn is suppressed,
the answerable ones are going too.

**Why tuning it by inspection does not work.** The threshold and the returned score are
not the same quantity. From `mem0/utils/scoring.py`:

```
threshold: Minimum semantic score required before hybrid scoring.
...results below the threshold are excluded even if BM25/entity would boost them.

    semantic_score = result.get("score") or 0.0
    if semantic_score < threshold:
        continue
    raw_combined = semantic_score + bm25_score + entity_boost
    combined = min(raw_combined / max_possible, 1.0)
    ...
    "score": combined,
```

The threshold filters on the **semantic score alone, before fusion**. The `score` handed
back is the **fused** semantic + BM25 + entity value. So a row can be reported at 0.811
and still be cut by a threshold of 0.65, because its pre-fusion semantic score was lower.
Observed directly:

```
threshold=0.6   returned=1   top score 0.811
threshold=0.65  returned=0
```

This is documented behaviour in their own docstring, not a defect. But it means the one
knob available for abstention cannot be set by looking at the scores the API returns,
which is what anyone would naturally try.

That is a narrower and more defensible claim than "the relevance check does nothing", and
it is an independent demonstration of what MemReranker reports: relevance scores are
miscalibrated, which makes threshold-based filtering difficult.
