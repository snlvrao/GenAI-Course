#!/usr/bin/env python3
"""Checks verdict.json for the Module 5 mini-project. Run: python check.py
Reads only your recorded file. No API key, no network, no cost."""
import json, sys
from pathlib import Path

PATH = Path(__file__).with_name("verdict.json")
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def lst(x):                         # anything that should be a list but is not is empty
    return x if isinstance(x, list) else []

def num(d, k):                      # a missing or non-numeric field counts as 0
    v = d.get(k, 0)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

OFFLINE = False        # set once verdict.json is read

def cost(run, prices):              # same formula as the lab, cached input at one tenth
    if OFFLINE:
        # A local model costs no money, so the offline path measures the same thing
        # in tokens. More thinking still costs more, which is the whole point.
        return num(run, "input") + num(run, "output")
    p = list(prices.get(run.get("model")) or []) + [0, 0]
    fresh = max(num(run, "input") - num(run, "cached"), 0)
    return (fresh * p[0] + num(run, "cached") * p[0] * 0.10 + num(run, "output") * p[1]) / 1_000_000

if not PATH.exists():
    sys.exit(f"FAIL no verdict.json next to check.py (looked in {PATH.parent}). Write it first.")
try:
    d = json.loads(PATH.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    sys.exit(f"FAIL verdict.json is not valid JSON: {e.msg}, line {e.lineno}")

need = ["fast_model", "reasoning_model", "prices", "cases", "rule"]
gone = [k for k in need if k not in d]
if gone:
    sys.exit("FAIL verdict.json is missing top-level keys: " + ", ".join(gone))

OFFLINE = d.get("path") == "offline"
fast, slow = d["fast_model"], d["reasoning_model"]
prices = d["prices"] if isinstance(d["prices"], dict) else {}
if OFFLINE:
    print("PASS offline path: grading on token counts, not money")
cases = {c.get("role"): c for c in lst(d["cases"]) if isinstance(c, dict)}
check(fast != slow,
      "the two runs are labelled differently"
      + (" (offline: use one model with two modes, e.g. '... (direct)' "
         "and '... (think first)')" if OFFLINE else ""))
if OFFLINE:
    check(d.get("cost_unit") == "tokens",
          'offline runs set "cost_unit": "tokens", since your model is free')
else:
    check(all(isinstance(prices.get(m), list) and len(prices[m]) == 2 for m in (fast, slow)),
          "prices has an [input, output] pair for both models")
check(set(cases) == {"winner", "waste"},
      f"two cases, roles winner and waste (found {sorted(cases)})")
if set(cases) != {"winner", "waste"}:
    sys.exit("1 check failed and the rest need both cases. Stopping.")

for role in ("winner", "waste"):
    c = cases[role]
    runs = [r for r in lst(c.get("runs")) if isinstance(r, dict)]
    per = {m: [r for r in runs if r.get("model") == m] for m in (fast, slow)}
    q = str(c.get("question", ""))
    check(len(q) >= 30 and "Four servers (A, B, C, D)" not in q,
          f"{role}: question is your own and at least 30 characters")
    check(len(str(c.get("expected", "")).strip()) >= 2,
          f"{role}: expected answer or pass criterion recorded")
    check(len(per[fast]) == 3 and len(per[slow]) == 3,
          f"{role}: 3 runs per model (got {len(per[fast])} fast, {len(per[slow])} reasoning)")
    check(all(isinstance(r.get("pass"), bool) for r in runs),
          f"{role}: every run graded true or false, not a 1-5 score")
    over = [r for r in runs if num(r, "reasoning") > num(r, "output")]
    check(not over, f"{role}: reasoning tokens sit inside output tokens ({len(over)} row(s) above)")
    bad = [r for r in runs if abs(cost(r, prices) - num(r, "cost")) > 1e-9]
    check(not bad, f"{role}: every recorded cost matches your price table ({len(bad)} off)")
    fp = sum(1 for r in per[fast] if r.get("pass") is True)
    sp = sum(1 for r in per[slow] if r.get("pass") is True)
    if role == "winner":
        check(sp == 3 and fp == 0, f"winner: reasoning 3/3 and fast 0/3 (got {sp}/3 and {fp}/3)")
    else:
        check(sp == 3 and fp == 3, f"waste: both models 3/3 (got {sp}/3 and {fp}/3)")
    mean = lambda rs: sum(cost(r, prices) for r in rs) / len(rs) if rs else 0.0
    got = mean(per[slow]) / mean(per[fast]) if mean(per[fast]) > 0 else 0.0
    check(abs(got - num(c, "cost_multiple")) <= 0.05 * max(got, 1e-9),
          f"{role}: cost_multiple {c.get('cost_multiple')} agrees with the runs ({got:.1f})")
    if role == "waste":
        check(got >= 2.0, f"waste: reasoning did cost more ({got:.1f}x the fast model)")

rule = d["rule"] if isinstance(d["rule"], str) else ""
check(len(rule.split()) >= 25, f"rule is written out ({len(rule.split())} words, 25 minimum)")
print("NOTE  whether the rule is a good rule is not checked automatically. Read it back and")
print("      ask whether a teammate could apply it without rerunning your calls.")
print()
if fails:
    print(f"{len(fails)} check(s) failed")
    sys.exit(1)
print("ALL CHECKS PASSED")
sys.exit(0)
