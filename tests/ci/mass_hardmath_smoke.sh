#!/usr/bin/env bash
# MASS e2e smoke: bench cot n=2 on a real backend; asserts answers parse, score
# in [0,1], reset keeps the protocol. Gated on OMAR_MASS_E2E=1 (else skip, exit 0).

set -euo pipefail

OMAR_BIN="${OMAR_BIN:-target/debug/omar}"
MASS_BIN="${MASS_BIN:-target/debug/omar-mass}"
BACKEND="${OMAR_MASS_BACKEND:-claude}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "${OMAR_MASS_E2E:-0}" != "1" ]; then
  echo "SKIP: OMAR_MASS_E2E!=1 (set to 1 in nightly/manual runs)"
  exit 0
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
command -v "$BACKEND" >/dev/null 2>&1 || { echo "SKIP: $BACKEND CLI not on PATH"; exit 0; }
[ -x "$OMAR_BIN" ] || { echo "omar binary not found: $OMAR_BIN" >&2; exit 1; }
[ -x "$MASS_BIN" ] || { echo "omar-mass binary not found: $MASS_BIN" >&2; exit 1; }

server="omar-mass-smoke-${RANDOM}-$$"
omar_dir="$(mktemp -d)"
summary="$omar_dir/summary.json"

cleanup() {
  tmux -L "$server" kill-server >/dev/null 2>&1 || true
  rm -rf "$omar_dir"
}
trap cleanup EXIT

cat >"$omar_dir/config.toml" <<EOF
[dashboard]
refresh_interval = 1
session_prefix = "omar-agent-"

[agent]
default_command = "$BACKEND"
default_workdir = "."
EOF

cd "$REPO_ROOT"
OMAR_BIN="$OMAR_BIN" OMAR_DIR="$omar_dir" OMAR_TMUX_SERVER="$server" \
  "$MASS_BIN" bench --method cot --n 2 --backend "$BACKEND" \
  --timeout-secs 300 --out "$summary"

# Assert both instances parsed, scored in [0,1], and no sessions leaked.
python3 - "$summary" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
assert s["n"] == 2, s
missing = [i["task_id"] for i in s["instances"] if not i["predicted"]]
# Empty `missing` also proves the between-instance reset kept the protocol.
assert not missing, f"instances without parsed answers (reset may have broken protocol): {missing}"
for i in s["instances"]:
    assert 0.0 <= i["score"] <= 1.0, i
print(f"PASS: {s['n']} instances parsed and judged (reset survived), mean score {s['accuracy']:.2f}")
EOF

leaked="$(tmux -L "$server" list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^omar-agent-.*mass' || true)"
if [ "${leaked}" != "0" ]; then
  echo "FAIL: leaked MASS agent sessions:" >&2
  tmux -L "$server" list-sessions >&2 || true
  exit 1
fi

echo "PASS: mass math smoke"
