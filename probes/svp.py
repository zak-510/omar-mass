#!/usr/bin/env python3
"""Structure-vs-prompt experiment. Writes svp.csv + SVP_DONE marker.

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
OUT = os.path.join(DIR, "svp.csv")
DONE = os.path.join(DIR, "SVP_DONE")
LOG = os.path.join(DIR, "svp.log")

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
PAYLOAD = "PAYLOAD: " + ", ".join(FACTS) + "."

rows = []
def run(args, per_wave=120, hard=2400):
    cmd = [BIN, "graph"] + args + ["--timeout-secs", str(per_wave)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=hard)
        return json.loads(p.stdout)
    except Exception as e:
        return {"error": str(e)[:80]}

def log(exp, label, d, metric):
    row = {"exp": exp, "label": label, "hops": d.get("hops"),
           "completed": d.get("completed"), "reason": d.get("reason", d.get("error", "")),
           "metric": metric, "output": (d.get("output") or "")[:60]}
    rows.append(row)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[{time.strftime('%H:%M:%S')}] {exp}/{label}: hops={row['hops']} "
          f"completed={row['completed']} {metric}", flush=True)

if os.path.exists(DONE): os.remove(DONE)

def facts_in(d):
    out = d.get("output") or ""
    return sum(1 for f in FACTS if f in out)

# A: same topology (ring N=10), different stop-instructions. Does prompt move the lap count?
ring_prompts = {
    "forever": "Continuous heartbeat with no natural end. Keep circulating; pass onward each hop.",
    "three_laps": "Do at least THREE full laps around the ring before anyone stops. Keep passing; pass onward.",
    "pass_once": "This is a single relay. Pass exactly once, then the next node must STOP. Pass onward.",
}
for label, q in ring_prompts.items():
    d = run(["--kind", "ring", "--n", "10", "--max-hops", "30", "--question", q])
    stop = d.get("hops")
    laps = round(stop / 10, 2) if stop else None
    log("A_ring_termination", label, d, f"stop_hop={stop} laps={laps}")

# B: same topology (chain N=10), different payload-handling. Does prompt move fidelity?
chain_prompts = {
    "preserve": PAYLOAD + " Preserve every field exactly, carry forward. Pass onward.",
    "summarize": PAYLOAD + " Summarize what you received in at most 5 words. Pass onward.",
    "replace": PAYLOAD + " Ignore the payload, replace it with your own favorite number. Pass onward.",
}
for label, q in chain_prompts.items():
    d = run(["--kind", "chain", "--n", "10", "--question", q])
    log("B_chain_fidelity", label, d, f"facts={facts_in(d)}/5 len={len(d.get('output') or '')}")

# C: same prompt, different topology. Preserve-facts payload through chain vs ring.
p = PAYLOAD + " Preserve every field exactly, carry forward. Pass onward."
d = run(["--kind", "chain", "--n", "10", "--question", p])
log("C_same_prompt", "chain", d, f"facts={facts_in(d)}/5")
d = run(["--kind", "ring", "--n", "10", "--max-hops", "20", "--question", p])
log("C_same_prompt", "ring", d, f"facts={facts_in(d)}/5 stop_hop={d.get('hops')}")

open(DONE, "w").write(f"{len(rows)} runs\n")
print(f"\nSVP DONE. {len(rows)} runs -> {OUT}", flush=True)
