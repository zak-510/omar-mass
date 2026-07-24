# omar-mass

Run a question through a multi-agent topology and see whether teamwork beats a single answer (spoiler it does).

You pick a topology (how many agents, and whether they reflect, debate, or vote); it spawns the agents, routes the prompts, collects the answers, and cleans up. Topologies come from (Zhou et al. 2025, arXiv:2502.02533), this covers the building blocks and a HARDMath benchmark (arXiv:2410.09988).

## The building blocks

Five, run in order: Summarize, Reflect, Debate, Aggregate, with Execute on the predictor. `aggregate` is how many agents answer in parallel. The final pick is a majority vote; pass `--aggregator llm` for an agent judge.

## Running it

```bash
# one question through a topology
omar-mass run --topology '{"aggregate":5}' --question "..."

# HARDMath benchmark (300-problem subset in data/hardmath_subset.jsonl)
omar-mass bench --method sc5 --n 100        # cot | self-refine | sc5 | debate
omar-mass grade --dir <bench-out-dir>       # paired judge pass across saved predictions

# graph topologies: chain, ring, scatter-gather
omar-mass graph --kind ring --n 10 --question "..."

# try one block on a tiny built-in example
omar-mass demo-block --block aggregate

# pick the backend and model
omar-mass bench --method cot --n 100 --backend opencode --model deepseek-v3

# kill agents a crashed run left behind
omar-mass teardown
```

Each method is capped at 10 agent calls, the same limit the paper uses (the grader call is separate). On a memory-bound machine, add `--max-concurrent 1` so a wide topology like SC@5 runs its agents one at a time.

## Grading

One LLM judge scores the answers, not a normalizer. The `grade` pass makes a single blind, shuffled call per problem that scores every method together against the ground truth, so the same answer can't get different scores across methods. Returns a 0-1 score; accuracy is the mean (partial credit). Grader runs on Sonnet; solvers stay on their own model, used Haiku here.

## Notes

`data/hardmath_subset.jsonl` has 300 problems, 50 per question_type, from HARDMath. `--seed N` rotates the slice; the same seed gives every method the same problems. Rebuild with `scripts/prepare_hardmath.py`.
`cargo test -p omar-mass` covers the parsers, judge, topology math, and mailbox. The smoke test is `tests/ci/mass_hardmath_smoke.sh`, run only when `OMAR_MASS_E2E=1`.

## Acknowledgments

This work uses OMAR, a multi-agent orchestration system for creating and evaluating hierarchical agent organizations from a single terminal.

GitHub Repo: https://github.com/lsk567/omar
Overview: [https://omar.tech/blog/introducing-omar/](https://omar.rs/blog/introducing-omar/)

The author is advised by Shaokai Lin.
