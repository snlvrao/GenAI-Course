"""
Build track B1 - train your own tokenizer.

Module 3 showed you what tokens are. This builds the thing that makes them,
using byte pair encoding, the same algorithm behind the tokenizers real models
use. It is about forty lines and there is nothing hidden in it.

The idea in one sentence: start with single characters, then repeatedly glue
together whichever adjacent pair appears most often, and record the order you
did it in. That recorded order IS the tokenizer.

    python bpe.py                 trains on corpus.txt
    python bpe.py mytext.txt 500  trains 500 merges on your own file
"""

import json
import pathlib
import sys
from collections import Counter

# Reading the command line happens inside main(), not out here, so that
# importing this file does nothing except define the functions. Out here it
# would read whatever arguments the IMPORTING script was given, which means
# "import bpe" inside a test run picks up the test runner's arguments and
# either trains on the wrong file or dies on int("tokenizer").
DEFAULT_CORPUS = pathlib.Path(__file__).resolve().parents[1] / "lab21" / "corpus.txt"
DEFAULT_MERGES = 300


def pair_counts(ids: list[int]) -> Counter:
    """How often each adjacent pair appears. (1,2,3) has pairs (1,2) and (2,3)."""
    return Counter(zip(ids, ids[1:]))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with the single token `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2          # skip both halves of the pair we just replaced
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text: str, n_merges: int):
    """Return the merge table and the vocabulary it produces."""
    # Checked here rather than left to range() below. range() rejects a list
    # with "'list' object cannot be interpreted as an integer", which points at
    # the loop instead of at the argument that is actually wrong.
    if isinstance(n_merges, bool) or not isinstance(n_merges, int):
        raise TypeError(f"n_merges must be a whole number, not {type(n_merges).__name__}")
    if n_merges < 0:
        raise ValueError(f"n_merges must be 0 or more, not {n_merges}")

    ids = list(text.encode("utf-8"))          # start from raw bytes, 0 to 255
    merges: dict[tuple[int, int], int] = {}
    vocab = {i: bytes([i]) for i in range(256)}

    for k in range(n_merges):
        counts = pair_counts(ids)
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        if freq < 2:                           # nothing repeats, stop early
            print(f"  stopped at merge {k}: no pair appears twice")
            break
        new_id = 256 + k
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if k < 5 or k % 100 == 0:
            shown = vocab[new_id].decode("utf-8", errors="replace")
            print(f"  merge {k:>4}: {shown!r} appeared {freq} times")
    return merges, vocab, ids


def encode(text: str, merges: dict) -> list[int]:
    """Apply the merges in the order they were learned. Order matters."""
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        ids = merge(ids, pair, new_id)
    return ids


def decode(ids: list[int], vocab: dict) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def main(argv):
    corpus = argv[1] if len(argv) > 1 else str(DEFAULT_CORPUS)
    if len(argv) > 2:
        try:
            n_merges = int(argv[2])
        except ValueError:
            sys.exit(f"FAIL the merge count has to be a number, and {argv[2]!r} is not. "
                     "Try: python bpe.py mytext.txt 300")
    else:
        n_merges = DEFAULT_MERGES

    path = pathlib.Path(corpus)
    if not path.exists():
        sys.exit(f"FAIL no {corpus} here. Point this at a .txt file of your own, "
                 "or copy the corpus.txt from lab21.")
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"corpus {len(text):,} characters, {len(text.encode('utf-8')):,} bytes")
    print(f"training {n_merges} merges\n")

    merges, vocab, ids = train(text, n_merges)

    raw = len(text.encode("utf-8"))
    print(f"\nbytes before {raw:,}  ->  tokens after {len(ids):,}")
    print(f"compression  {raw / max(len(ids), 1):.2f}x  "
          f"({raw / max(len(ids), 1):.2f} bytes per token)")
    print(f"vocabulary   {len(vocab):,} entries")

    # The test that matters: encode then decode must give back exactly what
    # you started with. A tokenizer that loses information is a broken one.
    probe = text[:2000]
    ok = decode(encode(probe, merges), vocab) == probe
    print(f"round trip   {'lossless' if ok else 'LOSSY - something is wrong'}")

    out = {
        "n_merges": len(merges),
        "vocab_size": len(vocab),
        "compression": round(raw / max(len(ids), 1), 4),
        "round_trip_ok": ok,
        "corpus_bytes": raw,
        "token_count": len(ids),
        # JSON keys must be strings, so the pairs are written "a,b"
        "merges": {f"{a},{b}": n for (a, b), n in merges.items()},
        "sample_tokens": [
            vocab[i].decode("utf-8", errors="replace") for i in ids[:40]
        ],
    }
    pathlib.Path("tokenizer.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nwrote tokenizer.json")
    print("first 20 tokens:", out["sample_tokens"][:20])


if __name__ == "__main__":
    main(sys.argv)
