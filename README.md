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

### Why each runner has its own environment

Every runner starts with a PEP 723 block declaring its own dependencies, and runs
under `uv run --script`, which builds a throwaway environment for that script alone:

```python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mem0ai[nlp,extras]==2.0.20",
#   "sentence-transformers>=5.2.0",
#   "python-dotenv",
# ]
# ///
```

This is not a style preference. These libraries cannot coexist. Mem0's extras pin
`langchain-core<1`; LangMem requires `langchain-core>=1`. Installing both into one
environment fails to resolve, and pinning around it would mean running at least one
system at a version its own maintainers do not ship.

The upside is that you can run any single system without installing the others, and
each result file records the exact versions that produced it. The downside is that
the first run of each script pays an install, and there is no single lockfile for the
whole repo.

If you would rather use ordinary virtualenvs, one per system works the same way:

```bash
python -m venv .venv-mem0 && .venv-mem0/bin/pip install 'mem0ai[nlp,extras]==2.0.20' sentence-transformers python-dotenv
python -m venv .venv-langmem && .venv-langmem/bin/pip install langmem==0.0.30 langchain-openai sentence-transformers python-dotenv
python -m venv .venv-graphiti && .venv-graphiti/bin/pip install 'graphiti-core[falkordb]==0.30.1' httpx openai sentence-transformers python-dotenv
```

Then run each runner with that environment's interpreter. The scripts do not depend
on `uv`; the inline block is ignored by a plain `python runners/run_mem0.py`.

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

# Graphiti needs a graph database
docker run -d --name amx-falkordb -p 6389:6379 falkordb/falkordb:latest

make all                  # three runs of every system, then the comparison
```

Or one system at a time:

```bash
uv run --script runners/run_mem0.py --run 1
uv run --script runners/run_langmem.py --run 1
uv run --script runners/run_weave.py --run 1
uv run --script runners/run_graphiti.py --run 1
uv run --script compare.py
```

`run_weave.py` expects the Memory Weave checkout beside this repo; edit the path in
its `[tool.uv.sources]` block if yours lives elsewhere.

Results land in `results/<system>-run<n>.json` and are committed. The dumps are the
evidence; the summary printed to the terminal is a convenience.

Extraction is nondeterministic, so each system is run three times and the variance
is reported rather than hidden.

## What is and is not reproducible

| System | Status | What it needs |
| --- | --- | --- |
| Mem0 | Runs here | An OpenAI key |
| LangMem | Runs here | An OpenAI key |
| Memory Weave | Runs here | A local checkout, no key |
| Graphiti | Runs here | FalkorDB in Docker, an OpenAI key |
| Letta | Not yet | A Letta server in Docker, plus an agent configured with MemFS |
| AgentCore | Not yet | An AWS account with Bedrock AgentCore enabled, in a supported region. Costs money |
| Claude Code | Partially reproducible | Run a real session and read `~/.claude/projects/<project>/memory/`. Not scriptable against this harness |
| ChatGPT, Claude chat, Hermes, OpenClaw | Not reproducible | Products, not libraries. Described from documentation only |

Anything not run here is labelled as such wherever it is described. A schematic
snapshot and a measured one are not the same claim and are not presented as one.
