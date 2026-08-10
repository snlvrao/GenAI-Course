"""check.py - checks findings.json from the Module 4 mini-project.
Run it from my-work/labs/lab04/:  python check.py
Standard library only. No model load, no API key, no internet."""
import json, os, sys

PATH, fails = "findings.json", []

def check(ok, msg):                      # one PASS/FAIL line per assertion
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def stop():
    print("%d check(s) failed" % len(fails))
    sys.exit(1)

# 1. load the artefact, with a readable message instead of a traceback
if not os.path.exists(PATH):
    print("FAIL findings.json not found in " + os.getcwd())
    print("Save it next to check.py, then run this again.")
    sys.exit(1)
try:
    with open(PATH, encoding="utf-8") as f:
        d = json.load(f)
except json.JSONDecodeError as e:
    print("FAIL findings.json is not valid JSON: line %d, %s" % (e.lineno, e.msg))
    sys.exit(1)
print("PASS findings.json loaded")

# 2. every key the report must carry
need = ["model", "n_layers", "n_heads", "pronoun",
        "head_positions_checked", "pair", "picks", "claim"]
missing = [k for k in need if k not in d]
check(not missing, "all 8 top-level keys present"
      + (" (missing: %s)" % ", ".join(missing) if missing else ""))
if missing:
    stop()
L, H = d["n_layers"], d["n_heads"]
# 3. the number you claim to have searched must match the grid you searched
check(d["head_positions_checked"] == L * H,
      "head_positions_checked (%s) equals n_layers x n_heads = %d"
      % (d["head_positions_checked"], L * H))
# 4. the minimal pair has to be minimal, or it tests two things at once
ok_pair = isinstance(d["pair"], list) and len(d["pair"]) == 2
check(ok_pair, "pair holds exactly two sentences")
if not ok_pair:
    stop()
a, b = d["pair"]
wa, wb = a["sentence"].split(), b["sentence"].split()
diff = sum(1 for x, y in zip(wa, wb) if x != y) + abs(len(wa) - len(wb))
check(diff == 1, "the two sentences differ in exactly one word (found %d)" % diff)
# 5. an answer missing from its own sentence is a typo, not a finding
bare = lambda ws: [w.strip(".,;:!?\"'").lower() for w in ws]
pron = d["pronoun"].lower()
check(a["answer"].lower() in a["sentence"].lower()
      and b["answer"].lower() in b["sentence"].lower()
      and pron in bare(wa) and pron in bare(wb),
      "both answers appear in their own sentence and %r appears in both" % d["pronoun"])
check(a["answer"] != b["answer"],
      "the two answers differ, so the pair really does flip the referent")
# 6. both recorded grids must be n_layers x n_heads of token strings
def grid_ok(g):
    return (isinstance(g, list) and len(g) == L
            and all(isinstance(r, list) and len(r) == H for r in g)
            and all(isinstance(t, str) and t.strip() for r in g for t in r))
ok_g = isinstance(d["picks"], dict) and grid_ok(d["picks"].get("A")) and grid_ok(d["picks"].get("B"))
check(ok_g, "picks.A and picks.B are both %d x %d grids of token strings" % (L, H))
if not ok_g:
    print("skipping the claim check, the grids are the wrong shape")
    stop()
# 7. re-run the flip test over your own grids and hold your claim to the result
hits = [(i, j) for i in range(L) for j in range(H)
        if d["picks"]["A"][i][j] == a["answer"] and d["picks"]["B"][i][j] == b["answer"]]
c = d["claim"]
if c is None:
    check(not hits, "claim is null and no head in your grids flips correctly"
          + ("" if not hits else "; but %s does" % (hits[:3],)))
elif not (0 <= c.get("layer", -1) < L and 0 <= c.get("head", -1) < H):
    check(False, "claim layer/head is inside 0..%d and 0..%d" % (L - 1, H - 1))
else:
    i, j = c["layer"], c["head"]
    check((i, j) in hits, "layer %d head %d picks %r for A and %r for B"
          % (i, j, a["answer"], b["answer"]) + ("" if (i, j) in hits else
          "; your grids show %r and %r" % (d["picks"]["A"][i][j], d["picks"]["B"][i][j])))
print("not checked automatically: that the grids really came from the model,\n"
      "and whether the head still works on sentences you did not find it with.")
if fails:
    stop()
print("ALL CHECKS PASSED")
