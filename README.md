# agent-memory-experiments

One fixed transcript, run through five agent memory systems at their documented defaults,
dumping what is literally in each store and what literally comes back on each turn.

This is not a benchmark. There is no score and no leaderboard. Published memory benchmarks
(LoCoMo, LongMemEval) only ever search on benchmark questions, so they cannot measure the
two cases that matter most in production: an ordinary turn where nobody asked about the
past and the right answer is nothing, and a request whose answer a stored rule should
change but which shares no words with it. Those two cases are the reason this repo exists.

- **Findings:** [`RESULTS.md`](RESULTS.md)
- **How the findings could be wrong:** [`ROBUSTNESS.md`](ROBUSTNESS.md), an adversarial
  audit written against our own results, including one check that weakened a claim.
- **Raw dumps:** `results/*.json`, committed. Regenerate the tables with `make compare`.

## What gets stored

### The prior store

Before the transcript, every system is seeded with the same **forty background facts**
about one engineer's work: services, incidents, colleagues, conventions, tooling. They are
in [`distractors.py`](distractors.py), verbatim, so you can disagree with the selection.

They exist because without them a store holds two to four rows against a default result cap
of twenty, and "the search returned everything" is then an artefact of size rather than a
result. They are deliberately the same user's real work, because a store of unrelated
topics flatters every system: nothing scores highly, so nothing looks wrong.

Each fact also carries a subject and an attribute. Only Memory Weave uses them, because a
record's identity there is `entity + attribute` and it refuses a semantic write without one.
The other systems store the string alone; Graphiti and AgentCore re-extract into their own
shapes, so they never hold these rows verbatim.

### The transcript

One user, a coding agent, two sessions eight months apart. Session 2 starts with an empty
message history, so whatever survives is whatever the memory system wrote down.

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

### Direct probes: does it return rows it should not?

Each is a search issued with one turn's own text, which is what a host does on the eager
path.

| Probe | Kind | Right answer |
| --- | --- | --- |
| `The checkout latency graph is red.` | ordinary | empty |
| `Bump the retry count to 3.` | ordinary | empty |
| `What do I write tests in?` | answerable | the current test runner |
| `What is my dog's name?` | absent | empty |
| `Where was Rohan working in March?` | answerable, dated | Nimbus |

The two ordinary turns are the point. `What do I write tests in?` is a control: a system
that returns nothing on everything is broken, not disciplined. `What is my dog's name?` is
the easy empty. `Where was Rohan working in March?` separates the temporal models: Rohan
left Nimbus in April, so March is Nimbus while every store's current value is Lattice.

### Indirect probes: does it stay silent when it should speak?

Nine realistic requests, each with a standing rule already in the prior store that should
change the answer, and which shares almost no words with that rule.

| Query | Rule that should surface | Similarity rank |
| --- | --- | --- |
| `Add the Stripe API key to the config.` | Secrets are in AWS Secrets Manager, never env files. | 37/40 |
| `Drop the legacy_status column.` | Migrations must be backwards compatible for one release. | 25/40 |
| `Add a helper that parses the CSV.` | Type hints are required on all new Python code. | 19/40 |
| `Reformat this function.` | Line length is capped at 100 characters. | 10/40 |
| `Ship this fix straight to production.` | Deploys go out through GitHub Actions on merge to main. | 7/40 |
| `Write a query to find duplicate charges.` | The payments database is Postgres 16 on RDS. | 5/40 |
| `Bump the requests library.` | Dependency updates are batched weekly. | 3/40 |
| `Add an alert for high latency.` | The team writes runbooks for every alert. | 2/40 |
| `Merge this PR.` | Code review requires one approval before merge. | 1/40 |

"Similarity rank" is where the rule sits among the forty facts by raw `bge-m3` similarity,
measured before any memory system is involved. The spread from 1 to 37 is deliberate:
keeping the easy ones is what stops this being a rigged set.

Every rule is also paired with a direct question (`Where are secrets stored?`) that ranks it
**1 of 40**. That proves the fact is findable and the indirect phrasing is what breaks it.

`make validate` re-checks the target assignments blind: it shows a model the forty facts
shuffled and one query, with no hint at the intended answer, and asks which would change how
the request is carried out. It picked the intended target for 9 of 9.

## Fairness

Every system gets the same transcript in the same order, the same extraction LLM, and the
same embedding weights.

- **LLM**: `gpt-4o` at `temperature=0`, pinned because several newer models accept only
  their default temperature. Note Mem0 sends `temperature=0.1` if you do not override it.
- **Embedder**: `BAAI/bge-m3`, run locally via `sentence-transformers`. No embedding API
  access is required, which keeps the reproduction path independent of your key's model
  permissions.

Letta owns its embedding pipeline and cannot be handed a local model directly, so
[`tools/embedding_shim.py`](tools/embedding_shim.py) serves the same weights over an
OpenAI-shaped `/v1/embeddings` endpoint and Letta's `embedding_config` points at it. Every
system therefore runs on identical weights, not merely the same model name.

Every system runs at its **documented defaults**. Nothing is tuned. Where a default is
surprising it is recorded in the result file rather than changed.

### Mem0 needs its extras before it is the system its docs describe

A plain `pip install mem0ai` gives **semantic search only**. Mem0 v3's documented
"multi-signal hybrid search (semantic + BM25 keyword + entity matching)" needs
`mem0ai[nlp]` for spaCy and `mem0ai[extras]` for the fastembed BM25 model. Without them the
library logs a warning and quietly runs one channel. The runner records which signals were
live in `config.active_signals`, so a result file cannot be mistaken for a configuration it
did not have.

### Why each runner has its own environment

Every runner starts with a PEP 723 block declaring its own dependencies and runs under
`uv run --script`, which builds a throwaway environment for that script alone.

This is not a style preference. These libraries cannot coexist: Mem0's extras pin
`langchain-core<1` while LangMem requires `>=1`. One environment cannot resolve both, and
pinning around it would mean running a system at a version its maintainers do not ship.

The upside is that you can run one system without installing the others, and each result
file records the exact versions that produced it. The downside is an install on first run
and no single lockfile.

If you prefer ordinary virtualenvs, one per system works the same way. The scripts do not
depend on `uv`; a plain `python runners/run_mem0.py` ignores the inline block.

```bash
python -m venv .venv-mem0 && .venv-mem0/bin/pip install 'mem0ai[nlp,extras]==2.0.20' sentence-transformers python-dotenv
python -m venv .venv-langmem && .venv-langmem/bin/pip install langmem==0.0.30 langchain-openai sentence-transformers python-dotenv
python -m venv .venv-graphiti && .venv-graphiti/bin/pip install 'graphiti-core[falkordb]==0.30.1' httpx openai sentence-transformers python-dotenv
python -m venv .venv-letta && .venv-letta/bin/pip install letta-client python-dotenv
python -m venv .venv-agentcore && .venv-agentcore/bin/pip install boto3 python-dotenv
```

## Running it

### 1. Configure

```bash
cp .env.example .env      # add your OpenAI key; leave the model settings alone
```

The embedder downloads from HuggingFace on first use, about 1.6GB. If your network blocks
the Hub's revision check the load can hang for minutes, which is why the harness sets
`HF_HUB_OFFLINE=1` once the model is cached.

### 2. The systems that need nothing else

```bash
make local     # mem0, langmem (both store sizes), weave, then the comparison
```

### 3. The systems that need a service

**Graphiti** needs a graph database:

```bash
docker run -d --name amx-falkordb -p 6389:6379 falkordb/falkordb:latest
make graphiti
```

**Letta** needs its server and the embedding shim. The shim blocks, so run it in its own
terminal:

```bash
docker run -d --name amx-letta -p 8283:8283 -e OPENAI_API_KEY=$OPENAI_API_KEY letta/letta:latest
make shim        # separate terminal, leave running
make letta
```

**AgentCore** needs an AWS account. Follow [`docs/agentcore-setup.md`](docs/agentcore-setup.md)
first: it covers the IAM policy, the named profile, and the cost, which is under a dollar for
the whole experiment. There is no free tier.

```bash
make agentcore
```

The runner reads `AMX_AWS_PROFILE`, defaulting to `personal-agentcore`, so it cannot reach
another account by accident. It deletes its memory resource at the end of each run.

### 4. Compare

```bash
make compare               # regenerates results/comparison.txt
```

### 5. The follow-up experiments

```bash
make sweeps      # threshold sweeps and the top_k trade-off
make validate    # blind re-check of the indirect probe targets
```

`sweep_mem0_threshold.py` asks whether Mem0 can be tuned to abstain.
`sweep_topk_tradeoff.py` answers the obvious rebuttal that the cap was simply too small.
`sweep_graphiti_threshold.py` compares Graphiti's default RRF reranker against its
cross-encoder recipe.

Results land in `results/<system>-run<n>.json` and are committed. The dumps are the
evidence; the terminal summary is a convenience. Extraction is nondeterministic, so each
system runs three times and the variance is reported rather than hidden.

## What is and is not reproducible

| System | Status | What it needs |
| --- | --- | --- |
| Mem0 | Runs here, 3 runs | An OpenAI key |
| LangMem | Runs here, 3 runs plus a small-store variant | An OpenAI key |
| Letta | Runs here, 3 runs | Letta server in Docker, plus the embedding shim |
| Graphiti | Runs here, 3 runs | FalkorDB in Docker, an OpenAI key |
| AgentCore | Runs here, 3 runs | An AWS account, see `docs/agentcore-setup.md`. Costs money |
| Memory Weave | Runs here | A local checkout beside this repo, no key |
| Claude Code | Partially | Run a real session and read `~/.claude/projects/<project>/memory/`. Not scriptable against this harness |
| ChatGPT, Claude chat, Hermes, OpenClaw | No | Products, not libraries. Described from their documentation only |

Anything not run here is labelled as such wherever it is described. A schematic snapshot and
a measured one are not the same claim and are not presented as one.

`run_weave.py` expects the Memory Weave checkout beside this repo; edit the path in its
`[tool.uv.sources]` block if yours lives elsewhere.

## Known limitations

Stated here rather than buried, and expanded in [`ROBUSTNESS.md`](ROBUSTNESS.md):

- **Per-probe difficulty is embedder-specific.** Re-ranking the indirect probes with
  `all-MiniLM-L6-v2` gives a rank correlation of 0.657 with `bge-m3`, and two probes
  disagree sharply. System-level numbers held identical under the swap. Only
  `Add the Stripe API key to the config.` was hard under both.
- **Nine probes, three runs.** No statistical significance is claimed, only the direction
  and size of an effect that is large and monotonic.
- **The prior store is written, not sampled.** Forty facts chosen to be one user's plausible
  work. All of them are in `distractors.py` to be argued with.
- **Result caps differ** (Mem0 20, Graphiti 10, LangMem 10, AgentCore 10, Letta 5), so raw
  ranks are never compared across systems. Caps are reported alongside every rank.
