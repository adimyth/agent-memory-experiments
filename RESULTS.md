# Results

Three runs per system, 6 September 2026. Same transcript, same LLM (`gpt-4o` at
`temperature=0`), same local embedder (`BAAI/bge-m3`). Every system at its documented
defaults, nothing tuned.

Raw dumps are in `results/`. Regenerate this table with `uv run --script compare.py`.

## The headline

Every probe, on both systems, in every run, returned **the entire store**.

That includes `What is my dog's name?`, where nothing in the store is about a dog and
an empty result is the only correct answer. It includes `Bump the retry count to 3.`,
which is a request to change code and contains no question about the past.

Mem0 applies a similarity threshold of `0.1` by default. LangGraph's `BaseStore`,
which LangMem builds on, applies no threshold at all. Neither floor is high enough to
exclude anything, so on this transcript neither system ever returns nothing.

## What the update turn did to the January rows

Tracked by row id rather than by text, because a row whose id survives but whose text
changed was edited in place and the January wording is gone.

| System | Run | Rows | Kept | Edited in place | Added | Removed |
| --- | --- | --- | --- | --- | --- | --- |
| Mem0 | 1 | 2 to 4 | 2 | 0 | 2 | 0 |
| Mem0 | 2 | 2 to 4 | 2 | 0 | 2 | 0 |
| Mem0 | 3 | 2 to 4 | 2 | 0 | 2 | 0 |
| LangMem | 1 | 3 to 3 | 3 | 2 | 0 | 0 |
| LangMem | 2 | 3 to 3 | 3 | 2 | 0 | 0 |
| LangMem | 3 | 3 to 4 | 3 | 1 | 1 | 0 |

Mem0 never edits and never deletes, in all three runs. That is v3's specified
ADD-only extraction, and it is perfectly consistent. After the update the store holds
both `User writes tests in pytest and keeps commit messages conventional` and
`User switched from writing tests in pytest to unittest.` Both are live. Ranking, not
the store, decides which one the model sees.

LangMem overwrote the same UUIDs in two runs out of three. The row that said
`User writes tests in pytest, indicating proficiency with this testing framework`
became `User switched to unittest, indicating a shift from pytest`, at the same key.
The January wording is gone with no lineage and no history. In the third run it
edited one row and inserted a new one instead, so the write behaviour is not stable
across runs even at `temperature=0`.

This is the sharpest contrast in the experiment. Same transcript, same model, same
embedder: one system keeps the superseded fact forever, the other destroys it.

## A caution about text matching

An obvious way to check whether the old fact survived is to grep the store for
`pytest`. It gives the wrong answer for LangMem, because the *replacement* text says
"a shift from pytest". The old row is gone and the word is still there. Any
comparison of these systems has to track identity, not strings.

## Mem0's hybrid search needs its extras

A plain `pip install mem0ai` gives semantic search only. The library logs a warning
and runs one channel:

```
fastembed not installed - BM25 keyword search disabled
Failed to load spaCy lemma model: spaCy is not installed
```

Mem0 v3's documented "multi-signal hybrid search (semantic + BM25 keyword + entity
matching)" needs `mem0ai[nlp]` and `mem0ai[extras]`. The results here were produced
with all three signals live, recorded in each result file under
`config.active_signals`.

Turning the other two signals on visibly sharpened the ranking. On
`Where was Rohan working in March?`, the two Rohan rows separated from the two
unrelated rows far more cleanly once entity matching was active:

| Row | Semantic only | All three signals |
| --- | --- | --- |
| Rohan is on-call and works at Nimbus | 0.509 | 0.593 |
| Rohan left Nimbus in April, joined Lattice | 0.619 | 0.458 |
| User switched to unittest | 0.304 | 0.122 |
| User writes tests in pytest | 0.284 | 0.110 |

The gap between the relevant and irrelevant rows widened from about 0.3 to about 0.4.
It changed which row ranks first. It did not change how many rows came back: still
four out of four.

## The dated question

`Where was Rohan working in March?` Rohan left Nimbus in April, so the answer is
Nimbus, and the store's current value is Lattice.

Mem0 still holds a standalone `Rohan is on-call and works at Nimbus on the payments
team` row, so the correct answer is physically present and ranks first. LangMem
overwrote that row, so after the update the only trace of Nimbus is the clause
"left Nimbus in April" inside the replacement text. The answer is reconstructable in
both cases, but only because the model is handed everything and can reason over it.
Neither system can be asked what was true in March.
