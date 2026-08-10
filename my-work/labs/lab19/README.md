# Lab 19: Train a tokenizer

**Module 19: B1 · Train your own tokenizer**

You are going to read and run my-work/labs/lab19/bpe.py, the whole tokenizer in about forty lines. Run it once, take the two core functions apart in a Python prompt, then break the merge order on purpose so you can see what a broken tokenizer looks like.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Run it once and read the log

Move into my-work/labs/lab19 and train 300 merges on the corpus that ships with the course. The first argument is the text file, the second is the number of merges, and both have defaults so a bare python bpe.py works too. Watch the first merges: the most common adjacent pair in English prose is not a word, it is the letter e followed by a space, which is the end of a word. Nothing prints between merge 4 and merge 100 because the code only logs the first five and then every hundredth. By merge 100 the log shows the whole word 'with' and by merge 200 it shows 'sha', neither of which anyone listed anywhere.

```python
cd my-work/labs/lab19
python bpe.py corpus.txt 300

corpus 1,115,394 characters, 1,115,394 bytes
training 300 merges

  merge    0: 'e ' appeared 27643 times
  merge    1: 'th' appeared 22739 times
  merge    2: 't ' appeared 16508 times
  merge    3: 's ' appeared 15364 times
(merges scroll past, the next log lines are merge 100 with 'with' and merge 200 with 'sha')

bytes before 1,115,394  ->  tokens after 549,002
compression  2.03x  (2.03 bytes per token)
vocabulary   556 entries
round trip   lossless

wrote tokenizer.json
first 20 tokens: ['F', 'ir', 'st ', 'C', 'it', 'i', 'z', 'en', ':\n', ...]
```

- `python bpe.py corpus.txt 300`: File first, merge count second. Read at the top of the file: CORPUS = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt".
- `'e ' appeared 27643 times`: BPE learns word boundaries before it learns words, because the space is the most predictable neighbour any letter has.
- `'with' at merge 100`: Merges compound. By merge 100 the algorithm is gluing tokens that earlier merges produced, not single letters.
- `first 20 tokens: 'F', 'ir', 'st '`: This is the split your model would actually see. Note that 'st ' includes the trailing space, which is normal and is why token counts do not match word counts.

**The maths, spelled out**

```
compression = corpus bytes / token count = 1,115,394 / 549,002 = 2.03 bytes per token
vocabulary = 256 byte values + one per merge = 256 + 300 = 556
```

> **Watch out:** Training 300 merges takes about a minute on a laptop processor with no graphics card, and 600 takes about two minutes, because every merge rescans the whole sequence in pure Python. It is not stuck. If you see 'FAIL no corpus.txt here' you are running from the wrong folder.

### 2. Take apart the two functions that do all the work

Everything else in the file is bookkeeping around pair_counts and merge. pair_counts pairs the list with itself shifted by one position, so Counter(zip(ids, ids[1:])) counts every adjacent pair in a single line. merge walks the list once and rebuilds it, replacing the target pair with one new id. Open a Python prompt in my-work/labs/lab19 and drive both by hand on a string short enough to check with your eyes. Importing bpe is safe because the training run sits behind if __name__ == "__main__".

```python
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

# in a Python prompt, real output:
>>> import bpe
>>> ids = list("banana banana".encode())
>>> ids
[98, 97, 110, 97, 110, 97, 32, 98, 97, 110, 97, 110, 97]
>>> bpe.pair_counts(ids).most_common(3)
[((97, 110), 4), ((110, 97), 4), ((98, 97), 2)]
>>> bpe.merge(ids, (97, 110), 256)
[98, 256, 256, 97, 32, 98, 256, 256, 97]
```

- `Counter(zip(ids, ids[1:]))`: ids[1:] is the same list starting one step later, so zipping them yields every adjacent pair. The whole counting step is one line.
- `(ids[i], ids[i + 1]) == pair`: The only comparison in the algorithm. There is no linguistics anywhere in this file, only counting.
- `i += 2`: Skips both halves so the token just created cannot be merged again in the same pass. Left to right, greedy.
- `new_id`: merge is handed the id and never asks what it means. Meaning lives entirely in the vocabulary dictionary.

**The maths, spelled out**

```
"banana banana" is 13 bytes. One merge of (97,110) removes 4 occurrences, one byte each: 13 - 4 = 9 tokens.
```

> **Watch out:** Because of that i += 2, overlapping pairs are consumed left to right. pair_counts on b'aaa' reports the pair (97,97) twice, but merging it produces [256, 97], only one replacement. The counts are a good enough guide, not a promise.

### 3. Read train() and watch the ordering being created

train is where the ordered table comes from. It starts ids from raw bytes and seeds the vocabulary with all 256 byte values, so nothing is ever out of vocabulary. Each round it hands out new_id = 256 + k, so ids are issued strictly in order and merge number 300 can never exist before merge 299. The new token's text is simply its two halves concatenated, which is why 'with' can appear at merge 100 without any word list. If no pair repeats it stops early, because there is nothing left to compress.

```python
def train(text: str, n_merges: int):
    """Return the merge table and the vocabulary it produces."""
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

# real output, three merges on a tiny string:
>>> m, v, ids = bpe.train("banana banana", 3)
>>> m
{(97, 110): 256, (98, 256): 257, (257, 256): 258}
>>> {k: v[k] for k in sorted(v) if k >= 256}
{256: b'an', 257: b'ban', 258: b'banan'}
```

- `ids = list(text.encode("utf-8"))`: Bytes, not characters. This one choice is why the tokenizer can encode any text on earth without an unknown token.
- `vocab = {i: bytes([i]) for i in range(256)}`: The 256 starting tokens are the 256 byte values. Your vocabulary is 256 entries before training does anything.
- `new_id = 256 + k`: Ids are handed out in loop order, so the id itself records when the merge was learned. That is the ordering you must preserve forever.
- `vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]`: Two byte strings concatenated. (98,256) is b'b' + b'an' = b'ban', built out of a token that did not exist one round earlier.
- `if freq < 2: break`: Guards tiny or highly varied corpora. If you ask for 5,000 merges on a small file you will get fewer and a printed reason.

**The maths, spelled out**

```
vocabulary size = 256 + number of merges. For the tiny run: 256 + 3 = 259 entries, of which 3 are new.
```

> **Watch out:** Three merges took 13 bytes down to 5 tokens, which looks like enormous compression. Tiny corpora always flatter themselves, and so does measuring compression on the exact text you trained on. The honest number comes from text the tokenizer has not seen.

### 4. Encode, decode, then break the order on purpose

encode replays the merges in the order the dictionary holds them, which works because Python dictionaries keep insertion order. decode joins the byte strings and decodes once. The round trip check inside bpe.py is four lines and it is the only test that really matters. Now do the destructive experiment: reverse the merge table and encode the same string. You get nine tokens instead of five and totally different ids, and the text still decodes perfectly, which is exactly why this bug is so easy to ship.

```python
def encode(text: str, merges: dict) -> list[int]:
    """Apply the merges in the order they were learned. Order matters."""
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        ids = merge(ids, pair, new_id)
    return ids


def decode(ids: list[int], vocab: dict) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

# the experiment, real output:
>>> m, v, _ = bpe.train("banana banana", 3)
>>> bpe.encode("banana banana", m)
[258, 97, 32, 258, 97]
>>> shuffled = {k: n for k, n in reversed(list(m.items()))}
>>> bpe.encode("banana banana", shuffled)
[98, 256, 256, 97, 32, 98, 256, 256, 97]
>>> bpe.decode(bpe.encode("banana banana", shuffled), v)
'banana banana'
```

- `for pair, new_id in merges.items()`: Replays the merges in learned order. If you ever rebuild this table from a set, a sorted list, or a JSON parser that reorders keys, your encoder is quietly wrong.
- `errors="replace" in decode`: A merge can split a multi-byte character, so decoding one token alone may not be valid UTF-8. The full sequence decodes exactly; this only softens the single-token case.
- `decode(encode(probe, merges), vocab) == probe`: The one assertion to run against any tokenizer you build or receive. Lossy in means garbage out, forever.
- `[98, 256, 256, 97, ...]`: The shuffled table never produced 257 or 258 because their inputs did not exist yet when their rows were applied.

**The maths, spelled out**

```
learned order: 13 bytes / 5 tokens = 2.60 bytes per token
shuffled order: 13 bytes / 9 tokens = 1.44 bytes per token, 80% more tokens for identical text
```

> **Watch out:** The shuffled tokenizer raises nothing, loses nothing, and passes a naive round trip test. The only symptoms are a worse compression ratio and different ids. Feed those ids to a model trained on the correct ids and you get noise with no error message anywhere.

### 5. Read the artefact it wrote

The run wrote tokenizer.json into the folder you ran from. JSON keys must be strings, so each pair (a, b) is stored as the string "a,b", and the file keeps them in learned order. Find the row "32,257" and work out what it is: 32 is a space, 257 is the merge that made 'th', so that row is the token ' th'. That single row is the proof that this table is a sequence and not a set. sample_tokens holds the first 40 tokens of your corpus already decoded, so you can see the split without writing any code.

```python
>>> import json
>>> d = json.load(open("tokenizer.json", encoding="utf-8"))
>>> d["n_merges"], d["vocab_size"], d["compression"], d["round_trip_ok"]
(300, 556, 2.0317, True)
>>> list(d["merges"].items())[:4]
[('101,32', 256), ('116,104', 257), ('116,32', 258), ('115,32', 259)]
>>> d["merges"]["32,257"]
273
>>> d["sample_tokens"][:9]
['F', 'ir', 'st ', 'C', 'it', 'i', 'z', 'en', ':\n']
```

- `('101,32', 256)`: Byte 101 is 'e', byte 32 is a space. The very first token you trained is a word ending.
- `d["merges"]["32,257"] is 273`: A merge whose right half is another merge. Any code that applies row 273 before row 257 produces a different, silently wrong encoding.
- `sample_tokens`: Written with vocab[i].decode(...), so it is the human-readable view of the first 40 ids. ':\n' being one token is the model learning that speaker labels end with a line break.

**The maths, spelled out**

```
Recompute the file's own claim: corpus_bytes / token_count = 1,115,394 / 549,002 = 2.0317, which is the compression field rounded to four places. Any mismatch means the file was edited by hand.
```

> **Watch out:** tokenizer.json is written into the current folder and every run overwrites it. Rename the ones you want to compare before rerunning.

### 6. Change the merge count and see the curve flatten

Run the trainer at 100 merges and again at 600, then compare the compression lines. More merges always help, but each one helps less than the last, and each one is a permanent row in your vocabulary. That row costs real parameters later: in B2 the embedding table is vocabulary times embedding width, so a 556-entry vocabulary at width 128 needs 71,168 numbers there, where the 65-character model you build in B2 needs only 8,320. Moving that model to this vocabulary would take it from 824,897 parameters to 951,084. Pick the smallest vocabulary that gets you the compression you need, then stop.

```python
python bpe.py corpus.txt 100
python bpe.py corpus.txt 600

# measured on 1.1 MB, laptop processor, no graphics card
# merges  vocab   tokens   bytes/token   training time
#    100    356  688,066      1.62           21s
#    150    406  635,062      1.76           32s
#    200    456  598,357      1.86           45s
#    300    556  549,002      2.03           67s
#    450    706  502,377      2.22           99s
#    600    856  470,334      2.37          133s
```

- `100 to 300 merges`: 200 extra merges buy 0.41 bytes per token, about 0.0021 each.
- `300 to 600 merges`: 300 extra merges buy 0.34 bytes per token, about 0.0011 each. Half the return for more vocabulary.
- `training time column`: Time grows faster than linearly at first because every merge rescans the whole sequence. This is the honest cost of a forty-line implementation; production trainers keep incremental pair counts.

**The maths, spelled out**

```
embedding parameters = vocabulary x embedding width. The 65-character model you build in B2 spends 65 x 128 = 8,320 there, 1.0% of its 824,897 parameters. A 556-entry vocabulary would need 71,168, and because the output layer grows with the vocabulary too, that model becomes 951,084 parameters. At 856 entries the embedding table is 109,568 and the model is 1,028,184.
```

> **Watch out:** Do not read these bytes-per-token numbers as your future bill on other text. They are measured on the same corpus the tokenizer was trained on, which is the best case it will ever have.

## You are done when

You can open any tokenizer's merge table, say what each row means, predict what reordering it would do, and prove the round trip. You built the thing that makes tokens, so tokens are no longer something that happens to you.

---

## Mini-project: Your own tokenizer

Train a tokenizer on your own writing. Not the shipped corpus, your text: notes, chat logs, code you wrote, old emails, blog drafts. The first merges will tell you something true about how you write, and the compression number will tell you what your text costs.

- Make a working folder, copy my-work/labs/lab19/bpe.py into it, and paste at least 50 KB of your own text into one plain UTF-8 file named mytext.txt. Under about 20 KB there is not enough repetition to compress and you will fail the ratio check.
- Run: python bpe.py mytext.txt 300. Read the merge log before anything else. Prose gives you word endings first, code gives you indentation runs, chat logs give you your own filler words.
- Check the compression line. If it is under 1.5 bytes per token, add more text or raise the merge count and rerun, for example python bpe.py mytext.txt 600.
- Save check_tokenizer.py next to tokenizer.json and run: python check_tokenizer.py mytext.txt. Passing the corpus filename is optional and adds a real round trip over the first 5,000 characters of your text.
- Keep both tokenizer.json and mytext.txt. B3 trains a model on text like this, and the vocabulary size you just chose is what sets the size of that model's embedding table.

### Check it

`check_tokenizer.py` is in this folder. Run it:

```bash
Save it in the same folder as tokenizer.json and run python check_tokenizer.py, or python check_tokenizer.py mytext.txt to add the round trip over your own corpus. Exit code 0 means everything passed. It rebuilds the vocabulary from the merge table and re-implements encode itself, so it is checking your file rather than trusting it. The first check rejects the corpus that ships with the course, so the artefact has to come from your own text.
```


**You are done when** check_tokenizer.py prints PASS on every line and ends with 10/10 checks passed, and tokenizer.json holds an ordered merge table trained on words you actually wrote.

**If you want more:** Train the same file at 100, 300 and 600 merges, record bytes per token for each, and write one sentence saying which you would ship and why. Then take the tokenizer trained on your text and encode 20 KB of something completely different, such as source code if you trained on prose. Compression will drop sharply and the round trip will still be lossless. That gap, measured on your own files, is the whole reason tokenizers are trained on the data they will serve.
