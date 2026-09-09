.PHONY: all local mem0 langmem letta graphiti agentcore sweeps validate compare shim

RUNS ?= 1 2 3

# Everything that needs no external service beyond an OpenAI key.
local: mem0 langmem compare

# Everything, including the systems that need Docker or AWS. Read the README first:
# letta needs the server and the embedding shim, graphiti needs FalkorDB, agentcore
# needs an AWS profile and costs money.
all: mem0 langmem letta graphiti agentcore compare

mem0:
	@for r in $(RUNS); do uv run --script runners/run_mem0.py --run $$r; done

langmem:
	@for r in $(RUNS); do uv run --script runners/run_langmem.py --run $$r; done
	@for r in $(RUNS); do uv run --script runners/run_langmem.py --run $$r --no-distractors; done

letta:
	@for r in $(RUNS); do uv run --script runners/run_letta.py --run $$r; done

graphiti:
	@for r in $(RUNS); do uv run --script runners/run_graphiti.py --run $$r; done

agentcore:
	@for r in $(RUNS); do uv run --script runners/run_agentcore.py --run $$r; done

# Serves bge-m3 over an OpenAI-shaped endpoint so Letta can use the same weights.
# Run in its own terminal; it blocks.
shim:
	@uv run --script tools/embedding_shim.py

sweeps:
	@uv run --script runners/sweep_mem0_threshold.py
	@uv run --script runners/sweep_topk_tradeoff.py
	@uv run --script runners/sweep_graphiti_threshold.py

validate:
	@uv run --script runners/validate_targets.py

compare:
	@uv run --script compare.py
