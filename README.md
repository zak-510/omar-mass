# omar-mass

Run a question through a multi-agent topology and see whether teamwork beats a single answer.

You pick a topology (how many agents, and whether they reflect, debate, or vote); it spawns the agents, routes the prompts, collects the answers, and cleans up. Topologies come from the MASS paper (Zhou et al. 2025, arXiv:2502.02533), this covers the building blocks and a HARDMath benchmark.

## The building blocks

Five, run in order: Summarize, Reflect, Debate, Aggregate, with Execute on the predictor. `aggregate` is how many agents answer in parallel. The final pick is a majority vote; pass `--aggregator llm` for an agent judge.

## Running it

```bash
# one question through a topology
omar-mass run --topology '{"aggregate":5}' --question "..."
omar-mass run --topology '{"debate":2,"aggregate":3}' --aggregator llm --question "..."

# try one block on a tiny built-in example
omar-mass demo-block --block aggregate

# HARDMath benchmark (300-problem subset in data/hardmath_subset.jsonl)
omar-mass bench --method cot --n 100          # single answer
omar-mass bench --method self-refine --n 100  # answer, critique, revise
omar-mass bench --method sc5 --n 100          # 5 answers, LLM aggregator picks
omar-mass bench --method debate --n 100       # 3 agents, 2 rounds, LLM judge

# pick the backend and model
omar-mass bench --method cot --n 100 --backend opencode --model deepseek-v3

# kill agents a crashed run left behind
omar-mass teardown
```

Each method is capped at 10 agent calls, the same limit the paper uses (the grader call is separate). On a memory-bound machine, add `--max-concurrent 1` so a wide topology like SC@5 runs its agents one at a time.

## Grading

Answers are scored by an LLM judge (arXiv:2410.09988), not a normalizer. One grader call per problem sees the predicted answer plus the ground-truth solution and a per-type rubric, and returns a 0-1 score. Accuracy is the mean score (partial credit). Because the judge has the gold solution, Haiku is a strong enough grader.

## Notes

`data/hardmath_subset.jsonl` has 300 problems, 50 per question_type, from HARDMath. `--seed N` rotates the slice; the same seed gives every method the same problems. Rebuild with `scripts/prepare_hardmath.py`.

`cargo test -p omar-mass` covers the parsers, judge, topology math, and mailbox. The smoke test is `tests/ci/mass_hardmath_smoke.sh`, run only when `OMAR_MASS_E2E=1`.

## Limitations

The rule-based majority vote buckets answers by a crude string normalization, which can't match equivalent open-form HARDMath expressions, so SC@5 uses the LLM aggregator instead (6 calls). Local models are too slow to be practical: each agent takes a couple of minutes to start and wide topologies serialize through one model server.

## Acknowledgments

This work uses OMAR, a multi-agent orchestration system for creating and evaluating hierarchical agent organizations from a single terminal.

GitHub Repo: https://github.com/lsk567/omar
Overview: https://omar.tech/blog/introducing-omar/

The author is advised by Shaokai Lin and Karim Elmaaroufi.