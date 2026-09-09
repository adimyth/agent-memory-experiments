# Results

Five systems, three runs each, 6 September 2026. Same transcript, same LLM
(`gpt-4o` at `temperature=0`), same local embedder (`BAAI/bge-m3`). Every system at
its documented defaults, nothing tuned.

Each store is seeded with the same forty prior facts before the transcript, so the
stores hold roughly forty-five rows rather than four. Without that, "the search
returned every row" is true but meaningless against a default result cap of 20.

Raw dumps are in `results/`. Regenerate with `uv run --script compare.py`.

## The headline

These systems have different default caps on a search result, so raw row counts
are not comparable. The comparable question is whether **anything other than the cap
ever kept a row out**.

| System | Cap | Probes returning the cap | Probes returning fewer |
| --- | --- | --- | --- |
| Mem0 | 20 | 15 of 15 | 0 |
| LangMem | 10 | 15 of 15 | 0 |
| AgentCore | 10 | 15 of 15 | 0 |
| Letta | 5 | 15 of 15 | 0 |
| Graphiti | 10 | 15 of 18 | 3 |

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
| 1 | 224 |
| 2 | 176 |
| 3 | 184 |

<!-- persona + human blocks after the update turn; see the `Core blocks after the update`
     note in each results/letta-run*.json -->

After the update turn, run 1's human block read:

> The user is Aditya, a software engineer.
> Aditya now writes tests in unittest and keeps commit messages conventional.
> Rohan joined Lattice after leaving Nimbus in April.

That text reaches the model on `Bump the retry count to 3.` as it reaches it on
a question about testing. There is no threshold to tune and no gate to pass, because
there is no retrieval step to gate. Every other system in this comparison has to be
asked before it says anything; Letta has already spoken.

It is doubly exposed, because the archival search runs on top of that and returns its
full cap of five on every probe as well.

The agent also rewrote the block in place. The January wording ("Aditya writes tests in
pytest") is gone, with no lineage, in all three runs. Same overwrite semantics LangMem
showed on a small store, reached by a different mechanism.

## Only Graphiti can be asked about March as a query

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

That returns one row, the same one in all three runs:
`Rohan works at Nimbus on payments.`

It works because Graphiti read "Rohan left Nimbus in April and joined Lattice" and
closed the interval on the old edge:

| Fact | valid_at | invalid_at |
| --- | --- | --- |
| Rohan works at Nimbus on payments. | 2026-01-10 | 2026-04-01 |
| Rohan joined Lattice. | 2026-04-01 | none |

It inferred April from natural language and wrote it as a boundary. No other system here
recorded when the old fact stopped being true, so no other system exposes this as a query.
That is a difference in precision rather than in capability: on an ordinary search, Mem0,
LangMem and AgentCore all return the Nimbus fact and the April departure at ranks 1 and 2,
which is enough for a model to answer March correctly. Graphiti returns one row and no
reasoning is required; the others return ten or twenty and some is.

## What the update turn did

Mem0 never edits and never deletes, in all three runs: two rows in, two new rows out.
That is v3's specified ADD-only extraction, with no run departing from it. Afterwards
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
rather than a tendency. A memory layer that consolidates correctly in a demo can
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
  because they do not fit an entity-relationship-entity shape.
- Letta wrote **0** archival rows from the transcript, because its agent chose to put
  those facts in the core block instead. Its archival store holds only the forty seeded
  facts, which is why its answerable probe returns five unrelated rows: the answer is
  real and it is in the system prompt, not in the searchable store.

Each is internally consistent across runs. They disagree about what a fact is.
Any row count quoted for these systems is a property of the extraction prompt, not of
the conversation.

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
a different embedder and confound the comparison, `tools/embedding_shim.py`
serves an OpenAI-shaped `/v1/embeddings` from the same local `bge-m3` weights, and the
agent's `embedding_config` points at it. Letta's chat model still goes to the real API.
Every system in this repo is therefore on identical embedding weights rather than the
same model name.


## AgentCore multiplies where the others append or overwrite

Three built-in strategies were enabled: semantic, summary, and user preference. From
roughly 43 input sentences they produced 70 to 80 records across three runs.

| Strategy | Records (3 runs) |
| --- | --- |
| FactExtractor (semantic) | 45, 46, 45 |
| PreferenceLearner | 19, 19, 29 |
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
| 1 | 67 records after 154.8s | 70 records after 139.2s |
| 2 | 67 records after 154.4s | 71 records after 154.3s |
| 3 | 77 records after 169.9s | 80 records after 138.9s |

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
| Where was Rohan working in March? | a hit | 0.858 |
| What do I write tests in? | a hit | 0.798 |
| Bump the retry count to 3. | empty | 0.637 |
| What is my dog's name? | empty | 0.461 |

The two questions the store can answer score above the two it should decline. The order
is right. Nothing is broken about the scoring.

**The default threshold sits far below all of it.** `0.1` is under every score in
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

## Threshold audit: can any of them be tuned to abstain?

The defaults result is only half the story. If a system ships an abstention knob and it
is set permissively, that is a different claim from the system being unable
to abstain. So: does each one have a knob, and does turning it up work?

| System | Abstention knob | Default | Does tuning fix it? |
| --- | --- | --- | --- |
| Mem0 | `threshold` | `0.1` | Partly, and it cannot be tuned by inspection |
| Graphiti | `reranker_min_score`, `sim_min_score` | `0` with RRF | **Yes, with a different reranker** |
| LangMem | none | n/a | No knob, and the scores are not separable |
| Letta | none | n/a | No knob |
| AgentCore | none | n/a | No knob |

Verified from source and signatures, not inferred. `BaseStore.search` is
`(namespace_prefix, *, query, filter, limit=10, offset=0, refresh_ttl=None)` and
`InMemoryStore` contains no occurrence of "threshold". Letta's `passages.search` takes
`query`, date bounds, tags, and `top_k`. AgentCore's `searchCriteria` is
`searchQuery`, `memoryStrategyId`, `topK`, `metadataFilters`.

### Graphiti can abstain. Its default reranker cannot.

`Graphiti.search()` uses `EDGE_HYBRID_SEARCH_RRF` with `reranker_min_score=0`. Raising
that floor truncates uniformly instead of discriminating:

| min_score | ordinary | answerable | absent | dated |
| --- | --- | --- | --- | --- |
| 0.0 | 10 | 10 | 10 | 10 |
| 0.2 | 5 | 5 | 5 | 5 |
| 0.3 | 3 | 3 | 3 | 4 |
| 0.6 | 1 | 1 | 1 | 4 |

The three columns move in lockstep, which is what reciprocal rank fusion should
do: an RRF score is a function of a result's **rank**, not of how good it is. The top
result scores about the same whether or not anything relevant exists, so a floor on it
can express "give me fewer" but never "give me nothing".

Swap in the cross-encoder recipe, which scores relevance directly, and the behaviour
changes:

| min_score | ordinary | answerable | absent | dated |
| --- | --- | --- | --- | --- |
| 0.0 | 10 | 10 | 10 | 10 |
| 0.1 and above | **0** | 0 | **0** | **1** |

At any floor from 0.1 up, Graphiti returns nothing on the ordinary turn, nothing on the
dog question, and one row on the dated question. The zero in the answerable
column is also correct here: Graphiti's extractor never created an edge for the test
runner in any of the three runs, so that fact is genuinely not in its store.

**This is the only configuration in the whole experiment that abstains correctly.** The
capability is shipped and it works. It is not what `Graphiti.search()` gives you.

### LangMem cannot be fixed from the outside either

There is no threshold to set, so the only option is filtering on returned scores in your
own code. On this data that does not work, because the scores are not separable:

- highest top score on a probe that should return nothing: **0.635** (`The checkout latency graph is red.`)
- lowest top score on a probe that should return something: **0.496** (`Where was Rohan working in March?`)

The unanswerable question outranks the answerable one. No single cutoff exists.

### What this changes

The honest headline is not "no system can decline to answer". It is:

> At their documented defaults, none of the five returns nothing on a turn where nothing
> should be returned. Of the five, one ships a mechanism that does the job correctly, and
> it is not switched on: Graphiti's cross-encoder reranker with any floor at or above
> 0.1. Mem0 ships a knob that helps and cannot be tuned by reading its own output. The
> other three ship no knob at all.


## Indirect association: the other direction of failure

The probes above measure whether a system returns rows it should not. These measure the
opposite: nine realistic requests, each with a standing rule already in the store that
should change the answer, and which shares almost no words with the request. Targets were
validated blind by a model that picked the intended one for 9 of 9 probes, and every miss
recorded below was verified to be present in that system's store.

| Query | Similarity rank | AgentCore | Graphiti | LangMem | Letta | Mem0 |
| --- | --- | --- | --- | --- | --- | --- |
| Add the Stripe API key to the config. | 37/40 | – | absent | – | – | – |
| Drop the legacy_status column. | 25/40 | 3 | absent | – | – | – |
| Add a helper that parses the CSV. | 19/40 | 2 | absent | – | – | 19 |
| Reformat this function. | 10/40 | 6 | absent | – | – | 10 |
| Ship this fix straight to production. | 7/40 | 1 | absent | 10 | – | 9 |
| Write a query to find duplicate charges. | 5/40 | 5 | 9 | 4 | 5 | 9 |
| Bump the requests library. | 3/40 | 4 | absent | 6 | 3 | 4 |
| Add an alert for high latency. | 2/40 | 8 | absent | 3 | 2 | 3 |
| Merge this PR. | 1/40 | 2 | absent | 1 | 1 | 1 |

A dash is a retrieval failure: the fact was in the store and the search did not return it.
`absent` is an extraction failure: the fact never entered the store.

For Mem0, LangMem and Letta the pattern is monotonic. Targets near the top of the
similarity ranking come back; targets near the bottom do not. Mem0 runs all three
retrieval signals and recovers nothing beyond raw embedding rank.

AgentCore is the exception and is covered in ROBUSTNESS.md section 8. Graphiti's eight
`absent` results are covered there too.

**The single clearest example.** On `Add the Stripe API key to the config.`, Mem0 returned
twenty rows including "The API gateway enforces a 30 second request timeout", "The March
incident review recommended adding circuit breakers", and "Feature flags are managed in
LaunchDarkly". It did not return "Secrets are stored in AWS Secrets Manager, never in
environment files", which is the one rule in the store that would have changed the answer.
It lost to a row that matched on the word API.

**The two failures are the same system on the same store.** On `Bump the retry count to 3.`
these systems return everything they have. On `Add the Stripe API key to the config.` they
stay quiet about the only row that mattered. Nothing is broken about the ranking; it is
doing what it was built to do. Wording proximity and answer usefulness have come
apart, and nothing here measures the second one.
