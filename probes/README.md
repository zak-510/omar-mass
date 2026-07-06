# Topology probes

Spawn large graph topologies and watch for any surprising runtime behavior. Runs on Haiku.

## Run

```
cargo build --release              # from repo root, builds the binary
python3 probes/sweep.py            # scaling sweep -> probes/results/sweep.csv
python3 probes/svp.py              # structure-vs-prompt -> probes/results/svp.csv
```

Add `--daemon` to detach for long unattended runs.
Paths default to the repo layout; override with `OMAR_MASS_BIN` (binary) and
`OMAR_MASS_OUT` (output dir) if yours differ.

## What each probe tests

- `sweep.py`: Chain N=20/50/100 (does the tail still produce?),
  ring N=10/25/40 (does passing ever stop?), scatter-gather with corrupted and
  missing workers, strict vs relaxed barrier.
- `svp.py`: Same topology / different prompts, same prompt
  / different topology. Checks whether behavior comes from the wiring or the
  agents.

## Results

`results/RESULTS.md` is the writeup. `results/*.csv` are the raw runs to diff
against. Overall, behavior seems to comes from the agents, not the topology. Ring self-
terminates at exactly one lap regardless of N or prompt; a smart gather stays correct at 4/5 corrupt.
