# agent-memory-experiments

One fixed transcript, run through several agent memory systems at their documented
defaults, dumping what is literally in each store and what literally comes back on
each turn.

This is not a benchmark. There is no score and no leaderboard. Published memory
benchmarks (LoCoMo, LongMemEval) only ever search on benchmark questions, so they
cannot measure the case that matters most in production: an ordinary turn where
nobody asked about the past and the right answer is nothing. That case is the
reason this repo exists.

Companion to the agent memory essay. Design notes live with the essay; this repo
holds the code and the raw results.

## The transcript

One user, a coding agent, two sessions eight months apart. Session 2 starts with an
empty message history, so whatever survives is whatever the memory system wrote down.

**Session 1, 10 January 2026**

```
Aditya: I write tests in pytest. Commit messages stay conventional.
        Rohan is on-call; he works at Nimbus, on payments.
Agent:  Okay.
```

**Session 2, 5 September 2026**

```
Aditya: The checkout latency graph is red.
Agent:  ...
Aditya: I switched to unittest. Rohan left Nimbus in April and joined Lattice.
Aditya: Bump the retry count to 3.
Aditya: Where was Rohan working in March?
```

## The probes

Each probe is a search issued with one turn's own text, which is what a host does on
the eager path. Three kinds, following the split in the Memory Weave README:

| Probe | Kind | Right answer |
| --- | --- | --- |
| `The checkout latency graph is red.` | ordinary | empty |
| `Bump the retry count to 3.` | ordinary | empty |
| `What do I write tests in?` | answerable | the current test runner |
| `What is my dog's name?` | absent | empty |
| `Where was Rohan working in March?` | answerable, dated | Nimbus |

The two ordinary turns are the point. `What do I write tests in?` is the control: a
system that returns nothing on everything is not doing well, it is broken. `What is
my dog's name?` is the easy empty, since nothing in the store is about a dog.

`Where was Rohan working in March?` separates the temporal models from the rest.
Rohan left Nimbus in April, so March is Nimbus, and the store's current value is
Lattice.

## Fairness

Every system gets the same transcript in the same order, the same extraction LLM,
and the same embedder.

- **LLM**: `gpt-4o` at `temperature=0`
- **Embedder**: `BAAI/bge-m3`, run locally via `sentence-transformers`

The embedder is local on purpose. It removes an API dependency from the reproduction
path, and it is the model Memory Weave already uses, so no system is advantaged by
being embedded differently.

`gpt-4o` was picked because it accepts `temperature=0`. Some newer models only accept
their default temperature, which adds sampling variance on top of the variance the
extraction prompt already has. Note that Mem0 sends `temperature=0.1` if you do not
override it.

Every system runs at its **documented defaults**. Nothing is tuned. Where a default
is surprising it is recorded in the result file rather than changed.

Each runner declares its own dependencies inline (PEP 723) and runs in an isolated
environment via `uv run --script`. This is not tidiness: Mem0's extras pin
`langchain-core<1` while LangMem requires `>=1`, so the two cannot share an
environment. Isolation also means each result file records the exact versions that
produced it.

### Mem0 needs its extras before it is the system the docs describe

A plain `pip install mem0ai` gives **semantic search only**. Mem0 v3's documented
"multi-signal hybrid search (semantic + BM25 keyword + entity matching)" needs
`mem0ai[nlp]` for spaCy entity matching and `mem0ai[extras]` for the fastembed BM25
model. Without them the library logs a warning and silently runs one channel.

The runner records which signals were actually live in `config.active_signals`, so a
result file can never be mistaken for a configuration it did not have.

## Running it

```bash
cp .env.example .env      # add an OpenAI key
uv sync
uv run python runners/run_mem0.py --run 1
uv run python runners/run_langmem.py --run 1
```

Results land in `results/<system>-run<n>.json` and are committed. The dumps are the
evidence; the summary printed to the terminal is a convenience.

Extraction is nondeterministic, so each system is run three times and the variance
is reported rather than hidden.

## What is and is not reproducible

| System | Status |
| --- | --- |
| Mem0 | Runs here |
| LangMem | Runs here |
| Memory Weave | Runs here |
| Graphiti | Needs Neo4j or FalkorDB in Docker |
| Letta | Needs a Letta server |
| ChatGPT, Claude chat, Hermes, OpenClaw | Products, not libraries. Not reproducible; described from documentation only |
| AgentCore | Needs AWS |

Anything not run here is labelled as such wherever it is described. A schematic
snapshot and a measured one are not the same claim and are not presented as one.
