#!/usr/bin/env python3
"""Scaling sweep for mass graph topologies. Writes CSV + DONE marker.

Paths default to the repo layout; override with env vars if needed:
  OMAR_MASS_BIN  omar-mass binary (default: ../target/release/omar-mass)
  OMAR_MASS_OUT  output dir       (default: ./results next to this script)
Add --daemon to detach and run in the background.
"""
import csv, json, subprocess, sys, time, os

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("OMAR_MASS_BIN", os.path.normpath(os.path.join(HERE, "..", "target", "release", "omar-mass")))
DIR = os.environ.get("OMAR_MASS_OUT", os.path.join(HERE, "results"))
os.makedirs(DIR, exist_ok=True)
OUT = os.path.join(DIR, "sweep.csv")
DONE = os.path.join(DIR, "DONE")
LOG = os.path.join(DIR, "sweep.log")

def daemonize():
    """Detach so the run survives a parent exit. Opt-in via --daemon."""
    if os.fork() > 0: os._exit(0)
    os.setsid()
    if os.fork() > 0: os._exit(0)
    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1); os.dup2(fd, 2)

if "--daemon" in sys.argv:
    daemonize()

FACTS = ["order_id=4471", "ship_date=09-30", "city=Lisbon", "sku=XZ12", "qty=37"]
CHAIN_Q = "PAYLOAD (preserve every field exactly, carry forward): " + ", ".join(FACTS) + ". Pass onward."
RING_Q = "Continuous token-passing heartbeat with no natural end. Keep circulating; pass onward each hop."
SG_Q = "What is 12 times 12? Give ONLY the number."
GOOD, CORRUPT = "144", "999"

rows = []
def run(args, per_wave_timeout=120, hard_timeout=2400):
    cmd = [BIN, "graph"] + args + ["--timeout-secs", str(per_wave_timeout)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=hard_timeout)
        return json.loads(p.stdout)
    except Exception as e:
        return {"error": str(e)[:80]}

def log(family, n, param, trial, d, metric):
    row = {"family": family, "n": n, "param": param, "trial": trial,
           "hops": d.get("hops"), "completed": d.get("completed"),
           "reason": d.get("reason", d.get("error", "")), "metric": metric,
           "output": (d.get("output") or "")[:60]}
    rows.append(row)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[{time.strftime('%H:%M:%S')}] {family} n={n} {param} t{trial}: "
          f"hops={row['hops']} completed={row['completed']} metric={metric}", flush=True)

if os.path.exists(DONE): os.remove(DONE)

for n in [20, 50, 100]:
    d = run(["--kind", "chain", "--n", str(n), "--question", CHAIN_Q])
    out = d.get("output") or ""
    survived = sum(1 for fct in FACTS if fct in out)
    log("chain", n, "-", 1, d, f"facts={survived}/{len(FACTS)} len={len(out)}")

for n in [10, 25, 40]:
    d = run(["--kind", "ring", "--n", str(n), "--max-hops", str(2 * n), "--question", RING_Q])
    stop = d.get("hops")
    laps = round(stop / n, 2) if stop else None
    log("ring", n, "-", 1, d, f"stop_hop={stop} laps={laps}")

for c in [0, 1, 2, 3, 4]:
    d = run(["--kind", "scatter-gather", "--n", "5", "--corrupt-count", str(c), "--question", SG_Q])
    out = d.get("output") or ""
    verdict = "good" if GOOD in out else ("corrupt" if CORRUPT in out else "other/none")
    log("sg-corrupt", 5, f"corrupt={c}", 1, d, f"verdict={verdict}")

for fc in [1, 3]:
    for relaxed in [False, True]:
        args = ["--kind", "scatter-gather", "--n", "5", "--fail-count", str(fc), "--question", SG_Q]
        if relaxed: args.append("--relaxed")
        d = run(args)
        avail = "yes" if d.get("completed") else "no"
        log("sg-missing", 5, f"fail={fc} relaxed={relaxed}", 1, d, f"available={avail}")

open(DONE, "w").write(f"{len(rows)} runs\n")
print(f"\nDONE. {len(rows)} runs -> {OUT}", flush=True)
