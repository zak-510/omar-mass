## Chain: does the tail produce?

- N=20: all 5 fields arrive, run completes
- N=50: all 5 fields arrive, run completes
- N=100: one run stalled at node 41 with no output

Worth noting that fidelity does not degrade slowly as the chain gets longer. At
20 and 50 nodes every field comes through untouched. The N=100 failure was not
the message getting garbled on the way down. A single node went quiet on a
payload that was only 1.6 KB, and everything before it had relayed fine, so it
looks like a transient hiccup on one call rather than a real ceiling. I would
not read it as "chains break past 50" without running it a few more times. For
now N=50 is a solid yes and N=100 is unconfirmed.

## Ring: does the passing ever stop?

- N=10: stops after 10 hops (one lap)
- N=25: stops after 25 hops (one lap)
- N=40: stops after 40 hops (one lap)

Every ring halts itself after exactly one lap, no matter how large. This held
even when the message explicitly told it to keep circulating with no end.

## Scatter-gather: waiting for everyone

Corruption test, 5 workers, some forced to answer 999 instead of the correct 144:

- 0 through 4 corrupt workers: the gather returned 144 every time

The answer never flipped to 999, even with 4 of the 5 workers emitting 999. The gather
node is an LLM that knows 12x12, so it recognized the junk and dropped it. A
plain majority vote would have returned 999 once three or more workers were
corrupt. So there is no corruption threshold here for a question the aggregator
can check itself.

Missing-worker test, strict versus relaxed barrier:

- 1 missing, strict: refused (4 of 5 arrived)
- 1 missing, relaxed: returned 144 (4 of 5)
- 3 missing, strict: refused (2 of 5 arrived)
- 3 missing, relaxed: returned 144 (2 of 5)

Strict waits for all five and refuses if any doesn't arrive. Relaxed ships whatever 
showed up and will hand back a confident answer with only two workers reporting.

## Structure versus prompt

The one lever available from the CLI is the task payload, not the agent's
underlying role prompt. With that caveat:

- Ring termination, same size, three payloads (circulate forever, do three laps,
  pass once): all stopped at one lap. The prompt moved termination not at all.
- Chain fidelity, same size, three payloads (preserve, summarize, replace):
  5/5, 5/5, 4/5. Only the "ignore it and replace with your own number"
  instruction changed anything, and only one field out of five. "Summarize in
  five words" was ignored.
- Same preserve payload through a chain and a ring: 5/5 both, and the ring still
  stopped at one lap. Changing the wiring changed nothing.
