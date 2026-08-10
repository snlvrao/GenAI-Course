"""Checker for B1. Run: python check_tokenizer.py  (optionally: ... mytext.txt)"""
import json, pathlib, sys

SHIPPED = 1115394          # size of the corpus.txt that ships with the course
res = []

def check(name, ok, detail=""):
    res.append((name, ok, detail))

p = pathlib.Path("tokenizer.json")
if not p.exists():
    print("FAIL  tokenizer.json not found. Run: python bpe.py mytext.txt 300")
    sys.exit(1)
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    print(f"FAIL  tokenizer.json is not valid JSON: {e}")
    sys.exit(1)

need = ["n_merges", "vocab_size", "compression", "corpus_bytes",
        "token_count", "merges", "sample_tokens"]
missing = [k for k in need if k not in d]
if missing:
    print(f"FAIL  tokenizer.json is missing keys: {missing}")
    sys.exit(1)

merges = list(d["merges"].items())          # JSON keeps insertion order
check("own corpus, not the shipped corpus.txt",
      d["corpus_bytes"] != SHIPPED, f"corpus_bytes={d['corpus_bytes']:,}")
check("merge count matches the table",
      d["n_merges"] == len(merges), f"n_merges={d['n_merges']} table={len(merges)}")
check("vocab size = 256 + merges",
      d["vocab_size"] == 256 + len(merges), f"vocab_size={d['vocab_size']}")

# merge table ordered and consistent: ids run 256, 257, ... and every pair
# refers only to tokens that already exist when the merge is applied.
ordered, consistent, vocab = True, True, {i: bytes([i]) for i in range(256)}
for k, (key, new_id) in enumerate(merges):
    a, b = (int(x) for x in key.split(","))
    if new_id != 256 + k:
        ordered = False
    if a not in vocab or b not in vocab or a >= new_id or b >= new_id:
        consistent = False
        break
    vocab[new_id] = vocab[a] + vocab[b]
check("merge ids are consecutive from 256", ordered)
check("every merge builds on earlier tokens", consistent)

ratio = d["corpus_bytes"] / max(d["token_count"], 1)
check("compression above 1.5", ratio > 1.5, f"{ratio:.2f} bytes per token")
check("recorded compression matches the counts",
      abs(ratio - d["compression"]) < 0.01, f"file says {d['compression']}")

def encode(text):
    ids = list(text.encode("utf-8"))
    for key, new_id in merges:
        pair = tuple(int(x) for x in key.split(","))
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        ids = out
    return ids

def roundtrip(text):
    try:
        return b"".join(vocab[i] for i in encode(text)).decode("utf-8", "replace")
    except KeyError:
        return None                      # merge table is broken, nothing to decode

probe = "".join(d["sample_tokens"])
if chr(0xFFFD) in probe or not probe:   # a token split a character in two
    probe = "round trip must be lossless: 1, 2, 3.\n"
check("round trip on the sample is lossless", roundtrip(probe) == probe)
check("file claims a lossless round trip", d.get("round_trip_ok") is True)

if len(sys.argv) > 1 and pathlib.Path(sys.argv[1]).exists():
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")[:5000]
    check(f"round trip on {sys.argv[1]} is lossless", roundtrip(t) == t)

for name, ok, detail in res:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
bad = sum(1 for _, ok, _ in res if not ok)
print(f"\n{len(res) - bad}/{len(res)} checks passed")
sys.exit(1 if bad else 0)
