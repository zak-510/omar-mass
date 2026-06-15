# omar-mass

Run a question through a multi-agent topology and see whether teamwork beats a single answer.

A small Rust CLI and library on OMAR. You pick a topology (how many agents, and whether they reflect, debate, or vote); it spawns the agents, routes the prompts, collects the answers, and cleans up. Topologies come from the MASS paper (Zhou et al. 2025, arXiv:2502.02533). This covers the building blocks and a MATH benchmark, not the optimizer.

## The building blocks

Five, run in order: Summarize, Reflect, Debate, Aggregate, with Execute on the predictor. `aggregate` is how many agents answer in parallel. For MATH the final pick is a majority vote; pass `--aggregator llm` for an agent judge.

## Running it

```bash
# one question through a topology
omar-mass run --topology '{"aggregate":9}' --question "..."
omar-mass run --topology '{"debate":2,"aggregate":3}' --aggregator llm --question "..."

# try one block on a tiny built-in example
omar-mass demo-block --block aggregate

# MATH benchmark (100-problem subset in data/math_subset.jsonl)
omar-mass bench --method cot --n 20        # single answer
omar-mass bench --method sc9 --n 20        # 9 answers, majority vote
omar-mass bench --method reflect --n 20    # self-refine: answer, critique, revise
omar-mass bench --method mad --n 20        # 3 agents, 3 rounds, judge
omar-mass bench --method sc9-tuned --n 20  # sc9 with the paper's tuned prompt

# pick the backend and model
omar-mass bench --method cot --n 20 --backend opencode --model deepseek-v3

# kill agents a crashed run left behind
omar-mass teardown
```

Every run is capped at 10 agent calls, the same limit the paper uses. On a memory-bound machine, add `--max-concurrent 1` so a wide topology like SC@5 runs its agents one at a time.

## Notes

Run it from a folder your backend CLI already trusts, or a first-run trust prompt will hang the spawn. It finds the omar binary via `OMAR_BIN`, then next to itself, then PATH.

`data/math_subset.jsonl` has 100 problems, 20 per difficulty level, from MATH-500. `--seed N` rotates the slice; `--stratified` spreads it across all five levels.

`cargo test -p omar-mass` covers the parsers, normalizer, topology math, and mailbox. The real-agent smoke test is `tests/ci/mass_math_smoke.sh`, run only when `OMAR_MASS_E2E=1`.

## Limitations

A couple of things we hit. Haiku gets pricey on the wide topologies, since SC@9 fires nine calls per question and that adds up fast across a full run. Going cheap with a local model was the opposite problem: too slow to be practical, because each agent takes a couple of minutes to start and the wide topologies serialize through one model server. We are still hunting for a backbone that is cheap, fast, and weak enough to leave the topologies something to improve.

## Acknowledgments

This work uses OMAR, a multi-agent orchestration system for creating and evaluating hierarchical agent organizations from a single terminal.

GitHub Repo: https://github.com/lsk567/omar
Overview: https://omar.tech/blog/introducing-omar/

The author is advised by Shaokai Lin and Karim Elmaaroufi.