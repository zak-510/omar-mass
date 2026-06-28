#!/usr/bin/env python3
# Build the vendored HARDMath subset: 50 problems per question_type (300 total).
# Source: https://github.com/sarahmart/HARDMath (data/HARDMath.json).
# Usage: python3 prepare_hardmath.py /path/to/HARDMath.json
import json, re, sys
from pathlib import Path

PER_TYPE = 50
TYPES = [
    "integral",
    "ODE",
    "polynomial_roots",
    "polynomial_roots_corrections",
    "nondimensionalization_symbolic",
    "nondimensionalization_numeric",
]


def clean_answer(raw):
    # Strip the dataset's $$, \[, \] wrappers around the boxed gold answer.
    s = (raw or "").strip()
    s = re.sub(r"\$+|\\\[|\\\]", "", s).strip()
    return s


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("HARDMath.json")
    out = Path(__file__).resolve().parent.parent / "data" / "hardmath_subset.jsonl"
    problems = list(json.load(open(src)).values())
    buckets = {t: [] for t in TYPES}
    for p in problems:
        t = p.get("question_type")
        if t in buckets and len(buckets[t]) < PER_TYPE:
            buckets[t].append(p)
    rows = []
    for t in TYPES:
        got = buckets[t]
        if len(got) < PER_TYPE:
            sys.exit(f"only {len(got)} problems for type {t}, need {PER_TYPE}")
        for i, p in enumerate(got):
            rows.append(
                {
                    "id": f"{t}-{i:03d}",
                    "question": p["question"].strip(),
                    "solution": p["solution"].strip(),
                    "answer": clean_answer(p.get("answer_val")),
                    "question_type": t,
                }
            )
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} problems to {out}")


if __name__ == "__main__":
    main()
