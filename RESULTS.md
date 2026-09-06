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
| Graphiti | 10 | 10 of 12 | 2 |
| Memory Weave | 8 | 6 of 15 | 9 |

For Mem0 and LangMem the cap is the only thing limiting output. Not once, on any
probe, in any run, did a threshold or a floor exclude a row. That includes
`What is my dog's name?`, where nothing in the store is about a dog and an empty
result is the only correct answer. Mem0 applies a similarity threshold of `0.1` by
default; LangGraph's `BaseStore`, which LangMem builds on, applies none at all.
Neither is high enough to bind on anything.

Graphiti's two exceptions are both the as-of query described below, not its ordinary
search.

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

That returns exactly one row: `Rohan works at Nimbus on payments.`

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

LangMem is more interesting, because its behaviour changed when the store got bigger.

With only the transcript's own facts in the store, LangMem's background manager
**overwrote two rows in place**, at the same UUIDs. The row that said
`User writes tests in pytest` became `User switched to unittest`. The January wording
was gone with no lineage.

With forty prior facts in the store, the same code on the same transcript
**appended instead**, leaving `The user writes tests using pytest.` and
`The agent switched to using unittest for testing.` both live at different keys.

The likely cause is `create_memory_store_manager`'s `query_limit`, which defaults to
5: the manager only sees the five most similar existing memories when it decides
whether to update or insert. Once the store is large enough that the pytest row falls
outside that window, the manager cannot update what it cannot see, so it inserts.

That is a scaling property worth knowing about. A memory layer that consolidates
correctly in a demo can quietly degrade to append-only in production, and nothing in
the output says it happened.

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
