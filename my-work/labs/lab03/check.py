# check.py - checks odd_one_out.json from the Module 3 mini-project.
# Run it in the same folder as your script: python check.py
import json, os, re, sys

PATH = "odd_one_out.json"
fails = []

def ok(cond, msg):
    """Print one result line and remember the failures."""
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)
    return cond

def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None

def near(a, b, tol):
    a, b = num(a), num(b)
    return a is not None and b is not None and abs(a - b) <= tol

if not os.path.exists(PATH):
    print(f"FAIL {PATH} not found. Run your script from this folder so it writes the file here.")
    sys.exit(1)
try:
    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"FAIL {PATH} is not valid JSON (line {e.lineno}): {e.msg}")
    sys.exit(1)

sets = {s.get("name"): s for s in data.get("sets", []) if isinstance(s, dict)}
ok(set(sets) == {"easy", "hard"}, "the file holds exactly two sets, named easy and hard")

for name in ("easy", "hard"):
    s = sets.get(name)
    if s is None:
        continue
    sent, M, fit = s.get("sentences", []), s.get("similarity_matrix", []), s.get("fit_scores", [])
    if not ok(len(sent) == 5 and len(set(sent)) == 5, f"[{name}] five different sentences"):
        continue
    square = len(M) == 5 and all(len(r) == 5 and all(num(v) is not None for v in r) for r in M)
    if not ok(square, f"[{name}] similarity_matrix is 5 rows of 5 numbers"):
        continue
    ok(all(near(M[i][i], 1.0, 0.01) for i in range(5)), f"[{name}] each sentence scores 1.0 against itself")
    ok(all(near(M[i][j], M[j][i], 0.002) for i in range(5) for j in range(5)), f"[{name}] the matrix is symmetric")
    means = [(sum(M[i]) - M[i][i]) / 4 for i in range(5)]  # the diagonal must not count
    if not ok(len(fit) == 5 and all(near(fit[i], means[i], 0.002) for i in range(5)),
              f"[{name}] fit_scores are the mean of the OTHER four scores, self excluded"):
        continue
    low = min(range(5), key=lambda i: fit[i])
    ok(s.get("predicted_odd_index") == low, f"[{name}] predicted_odd_index is the lowest fit score (index {low})")
    ranked = sorted(fit)
    ok(near(s.get("confidence_gap"), ranked[1] - ranked[0], 0.002),
       f"[{name}] confidence_gap = second lowest minus lowest")
    ok(s.get("correct") is (s.get("predicted_odd_index") == s.get("expected_odd_index")),
       f"[{name}] the correct flag matches predicted against expected")

if "easy" in sets:
    ok(sets["easy"].get("correct") is True, "[easy] the tool picked the sentence you expected")
if "hard" in sets:
    h, e = sets["hard"], sets["hard"].get("expected_odd_index")
    if len(h.get("sentences", [])) == 5 and isinstance(e, int) and 0 <= e < 5:
        words = lambda t: set(re.findall(r"[a-z]{4,}", t.lower()))
        # crude stemming: apple and apples count as the same word
        overlap = lambda x, y: any(a.startswith(b) or b.startswith(a) for a in x for b in y)
        odd = words(h["sentences"][e])
        shared = sum(1 for i in range(5) if i != e and overlap(odd, words(h["sentences"][i])))
        ok(shared >= 3, f"[hard] the odd sentence shares a word with 3 or more of the others (it shares with {shared})")
    print("NOTE  whether the hard set fools the tool is not checked. Either answer is a real result.")

# Strongest check: re-embed your sentences and compare. Offline, and skipped if the model is not cached.
os.environ["HF_HUB_OFFLINE"] = "1"
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(data.get("model", "sentence-transformers/all-MiniLM-L6-v2"))
    for name, s in sets.items():
        V = model.encode(s["sentences"], normalize_embeddings=True)
        fresh = (V @ V.T).tolist()
        worst = max(abs(fresh[i][j] - s["similarity_matrix"][i][j]) for i in range(5) for j in range(5))
        ok(worst < 0.02, f"[{name}] your scores match a fresh run of the model (worst gap {worst:.4f})")
except Exception as err:
    print(f"NOTE  could not re-run the model offline ({type(err).__name__}), so the numbers were not re-checked")

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
