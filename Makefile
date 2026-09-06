.PHONY: all mem0 langmem compare clean

RUNS ?= 1 2 3

all: mem0 langmem compare

mem0:
	@for r in $(RUNS); do uv run --script runners/run_mem0.py --run $$r; done

langmem:
	@for r in $(RUNS); do uv run --script runners/run_langmem.py --run $$r; done

compare:
	@uv run --script compare.py

clean:
	rm -f results/*.json
