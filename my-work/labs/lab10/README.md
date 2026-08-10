# Lab 10: Build a RAG system over your files

**Module 10: RAG - giving a model your own documents**

RAG stands for retrieval augmented generation, and there is less to it than the name suggests: you search your own files for a few relevant passages, paste that text into the prompt, and ask the model to answer using only those passages. The model is never retrained and it forgets your files the moment the reply ends, so everything that decides whether the answer is good or bad happens before the model sees anything. In this lab you build the whole pipeline over your own PDFs: read the files, cut them into chunks, index them two ways, merge the two ranked lists, rerank the survivors with a slower and better model, and answer with citations you can check by hand. Everything runs on your own machine except the final answer, with no vector database server and no Docker, so before you start make sure python llm.py works (see setup.html).

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Set up the folder and install four packages

Make a folder my-work/labs/lab10/ and a subfolder docs/ inside it, then drop 3 to 10 PDFs you actually care about into docs/, for example manuals, papers, or a company handbook. Use text based PDFs only for now, because a scanned page is a photograph of text: pypdf will pull an empty string out of it and you will build an index of nothing without any error message telling you so. The four packages do four separate jobs, and it is worth knowing which is which before you start, because when something breaks the error usually names one of them. pypdf reads PDFs, sentence-transformers turns text into numbers, sqlite-vec lets an ordinary SQLite file search those numbers, and rank-bm25 does old fashioned keyword scoring. The first time you run the code in later steps, sentence-transformers downloads two small models from Hugging Face, so expect a one time pause of a minute or two with a progress bar, and note that no API key is involved in any of it. Choosing documents you know well matters more than it sounds: if you can tell instantly when an answer is wrong, you will learn something from every run, and if you cannot, you will just be nodding at plausible text.

```python
pip install pypdf sentence-transformers sqlite-vec rank-bm25
```

- `pypdf`: Opens a PDF and hands you the text of each page. It does not do OCR (turning a picture of text back into text), so it only works on PDFs that already contain a text layer underneath the visible page.
- `sentence-transformers`: Downloads and runs the two small models this lab uses: the embedding model that turns any text into 384 numbers, and the cross-encoder that reranks results in step 5. Both run on your CPU, with no key and no internet needed after the first download.
- `sqlite-vec`: A SQLite extension that adds a table type able to store lists of numbers and find the closest ones. It is what lets your entire vector database be a single file called rag.db, with nothing to install and no server to start.
- `rank-bm25`: A small pure Python implementation of BM25, a keyword scoring formula from the 1990s. It is here because meaning based search is genuinely bad at part numbers, error codes and surnames, and this fills that gap.

**The maths, spelled out**

```
How big is "two small models"?

Size on disk = number of parameters x bytes per parameter

A parameter is one number the model learned during training. These models store each one as a 32 bit float, which is 4 bytes.

all-MiniLM-L6-v2 has about 22.7 million parameters.
  22,700,000 x 4 bytes = 90,800,000 bytes
  90,800,000 / 1,048,576 = about 87 MB

Where do 22.7 million parameters come from? Mostly the vocabulary table: 30,522 known word pieces x 384 numbers each = 11.7 million, which is over half the model before you reach a single transformer layer. The other 11 million sit in 6 stacked layers.

Being honest about the download figure: the cross-encoder used in step 5 (ms-marco-MiniLM-L-6-v2) is built the same way and is about the same size, so both together land nearer 180 MB than the "roughly 100 MB" you may see quoted. Budget a couple of hundred MB of disk and you will not be surprised.

Intuitively: these are tiny by 2026 standards. A chat model you call over the internet is hundreds of times larger, which is exactly why these two can sit on your laptop and run for free forever.
```

> **Watch out:** If your chunk count in step 3 comes out as 0, your PDFs are scanned images rather than text; open one in a reader and try to select a sentence with the mouse, and if you cannot, that file has no text layer for pypdf to find.

### 2. Read the PDFs and cut them into chunks

Create ingest.py and put this in it. A chunk is a slice of a document small enough to search and small enough to paste into a prompt, usually a few hundred words. You cut documents up for two reasons: you cannot paste a 40 page PDF into every prompt, and searching whole documents only tells you which file is about pensions rather than handing you the one sentence you need. This code walks each PDF page by page, keeps the page number so a citation can point at something you can actually open, and cuts each page into overlapping windows of 200 words. Chunking per page is a deliberate trade, not an oversight: you get exact page citations, but a sentence that runs across a page break gets split in half and neither half makes sense on its own. Nothing runs yet because main() arrives in the next step, so save the file and keep going. If you run it now you will see no output at all, which is correct.

```python
# ingest.py  (part 1 of 2)
import glob
import os
import sqlite3

import sqlite_vec
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlite_vec import serialize_float32

DB = "rag.db"
DIM = 384
EMBED = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def pages(path):
    reader = PdfReader(path)
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield number, text


def chunk(text, size=200, overlap=40):
    words = text.split()
    step = size - overlap
    i = 0
    while i < len(words):
        yield " ".join(words[i:i + size])
        if i + size >= len(words):
            break
        i += step
```

- `EMBED = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`: Loads the embedding model once, at import time, into a module level variable. Loading takes several seconds, so you do it once here rather than inside a loop, and the very first run also downloads it.
- `DIM = 384`: This model outputs exactly 384 numbers for any input text, whether that text is three words or three hundred. The table you create in step 3 must declare the same number, so it lives in one constant to stop the two values drifting apart.
- `text = (page.extract_text() or "").strip()`: extract_text() returns None when a page has no text layer, and None has no .strip() method, so the `or ""` turns None into an empty string first. The `if text:` line underneath then skips blank pages instead of storing them as empty chunks.
- `yield (used in both pages and chunk)`: Both functions hand back one item at a time instead of building a complete list in memory. For a 300 page PDF this keeps memory use flat, and it lets the caller start work on page 1 before page 300 has been read.
- `step = size - overlap`: The window slides forward by 160 words rather than 200, so the last 40 words of each chunk reappear as the first 40 words of the next. That repetition is what stops a sentence being cut in half and lost at every boundary.
- `if i + size >= len(words): break`: Stops the loop as soon as the current window already reaches the end of the page. Without it you would emit a final chunk of only 20 or 30 leftover words that repeats text you already stored, and short orphan chunks score badly and pollute results.

**The maths, spelled out**

```
The sliding window

step = size - overlap = 200 - 40 = 160 words

Number of chunks for a page of N words, when N is larger than size:

  chunks = ceil( (N - size) / step ) + 1

Worked example, a page holding 500 words:

  chunks = ceil( (500 - 200) / 160 ) + 1 = ceil(1.875) + 1 = 2 + 1 = 3

Trace it by hand and you get the same answer:
  i = 0    -> words 0 to 199    (200 words)
  i = 160  -> words 160 to 359  (200 words)
  i = 320  -> words 320 to 499  (180 words, then the break fires)

How much text ends up stored twice:
  words stored = 200 + 200 + 180 = 580
  duplication  = 580 / 500 = 1.16, so 16 percent more text than the page actually contains

Overlap fraction = 40 / 200 = 0.20, so every chunk repeats a fifth of the chunk before it.

Intuitively: overlap is insurance. You pay about 16 percent more storage and 16 percent more embedding time so that a sentence sitting on a boundary appears whole inside at least one chunk. Be honest about the defaults: there is no measured best value for 200 and 40, they are common starting points, and the right numbers for your documents can only be found by looking at your own chunks.
```

> **Watch out:** If a page contains a table, pypdf hands the words back as one long run of numbers with no row structure, so those chunks will read as nonsense; that is a real limit of text extraction, not a bug in your code.

### 3. Build the index in one SQLite file

Add this to the bottom of the same ingest.py file. There are two tables and they work as a pair. chunks is an ordinary SQLite table holding the text plus where it came from, and vec_chunks is a sqlite-vec virtual table holding only the 384 numbers for each chunk. A virtual table is a table whose behaviour is supplied by an extension rather than by SQLite itself, and that is what makes MATCH on a list of numbers possible at all. The two tables are joined by row id, so chunk 7 in one is chunk 7 in the other, which is why the insert loop deliberately uses the same i for both. Run it with python ingest.py and you should see a chunk count, a sample chunk, a progress bar while the model encodes, and finally the line wrote rag.db. Read that printed sample chunk properly rather than glancing at it: if it looks like broken table rows or repeated page headers, that is your first real lesson about chunking, and no amount of clever search later will rescue text that was cut up badly here.

```python
# ingest.py  (part 2 of 2)

def connect():
    db = sqlite3.connect(DB)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def main():
    db = connect()
    db.execute("DROP TABLE IF EXISTS chunks")
    db.execute("DROP TABLE IF EXISTS vec_chunks")
    db.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc TEXT, page INT, text TEXT)")
    db.execute(f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{DIM}])")

    rows = []
    for path in glob.glob(os.path.join("docs", "*.pdf")):
        name = os.path.basename(path)
        for page_no, text in pages(path):
            for piece in chunk(text):
                rows.append((name, page_no, piece))

    print(len(rows), "chunks")
    print("sample:", rows[0][2][:300] if rows else "none")

    vectors = EMBED.encode([r[2] for r in rows], batch_size=32, show_progress_bar=True)
    for i, (name, page_no, piece) in enumerate(rows, start=1):
        db.execute("INSERT INTO chunks (id, doc, page, text) VALUES (?,?,?,?)",
                   (i, name, page_no, piece))
        db.execute("INSERT INTO vec_chunks (rowid, embedding) VALUES (?,?)",
                   (i, serialize_float32(vectors[i - 1].tolist())))
    db.commit()
    print("wrote", DB)


if __name__ == "__main__":
    main()
```

- `db.enable_load_extension(True) ... sqlite_vec.load(db) ... db.enable_load_extension(False)`: Python's built in sqlite3 refuses to load C extensions by default, for safety. You switch loading on, load sqlite-vec, then switch it straight back off so nothing else can push code into this database connection later.
- `db.execute("DROP TABLE IF EXISTS chunks")`: Every run rebuilds the whole index from scratch, both tables. That keeps ingest.py simple and makes re-running it after you add a PDF completely safe, but be clear about what you are giving up: there is no incremental update, so a big corpus is re-embedded every time.
- `CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{DIM}])`: Declares a vector column of exactly 384 floats. The length is fixed at creation time, so if you later swap in a model that outputs 768 numbers the insert fails loudly instead of silently storing something meaningless.
- `EMBED.encode([r[2] for r in rows], batch_size=32, show_progress_bar=True)`: Encodes every chunk in one call, 32 chunks at a time. Batching matters a lot: 2,000 separate calls are far slower than 63 batches, because each batch runs as a single large matrix operation instead of 32 small ones.
- `serialize_float32(vectors[i - 1].tolist())`: sqlite-vec stores vectors as raw bytes, not as JSON or text. serialize_float32 packs the 384 Python floats into exactly 1,536 bytes in the byte layout the extension expects to read back.
- `enumerate(rows, start=1) paired with vectors[i - 1]`: SQLite row ids start at 1 while the Python vectors list is indexed from 0, so the ids and the list are off by one. The `i - 1` is that offset written out on purpose rather than hidden, and getting it wrong would silently attach every chunk to its neighbour's vector.

**The maths, spelled out**

```
How big does rag.db get, and how much work is one search?

Bytes per vector = dimensions x bytes per float = 384 x 4 = 1,536 bytes (1.5 KB)

Worked example: 3 PDFs of 40 pages each, about 500 words a page.
  pages           = 3 x 40 = 120
  chunks per page = 3 (from step 2's arithmetic)
  chunks          = 120 x 3 = 360
  vector storage  = 360 x 1,536 = 552,960 bytes, about 0.53 MB
  text storage    = 360 x 200 words x about 6 bytes a word = 432,000 bytes, about 0.41 MB

So rag.db lands near 1 MB. Even 50,000 chunks would only be 50,000 x 1,536 = 76,800,000 bytes, about 73 MB of vectors.

Search cost. sqlite-vec here compares your question vector against every stored vector, one at a time, with no shortcut index:

  operations = chunks x dimensions = 360 x 384 = 138,240 multiply-and-add steps

A laptop CPU does hundreds of millions of those per second, so this finishes in well under a millisecond.

Intuitively: brute force is completely fine at this size, and stays fine into the tens of thousands of chunks. Approximate indexes exist because at ten million chunks that same scan becomes 10,000,000 x 384 = 3.84 billion operations per query, and only at that point is it worth trading exact answers for speed.
```

> **Watch out:** If you see sqlite3.OperationalError: no such module: vec0, then sqlite_vec.load() never ran on that connection, which almost always means you opened the database with plain sqlite3.connect() somewhere instead of calling connect().

### 4. Search two ways and merge with RRF

Create a second file, search.py. This step is the retrieval half of RAG, and retrieval is where almost all RAG quality comes from, which is why the search code is longer than the prompt code. vector_ids asks sqlite-vec for the 20 chunks whose numbers sit closest to your question's numbers, and that is what finds paraphrases: "how much time off do I get" can match a chunk that says "annual leave entitlement" with no shared words at all. bm25_ids scores every chunk by keyword overlap and takes the top 20, which catches exactly what meaning search is worst at, such as P019, a part number, or a surname. rrf merges the two ranked lists using positions only and throws the actual scores away, because a cosine distance of 0.31 and a BM25 score of 8.4 are on completely different scales and adding them would be meaningless. Rebuilding the BM25 index on every single query is fine for a few thousand chunks and would be silly for a million, which is what a dedicated keyword engine such as Elasticsearch exists for. Nothing runs yet, because search() itself arrives in the next step.

```python
# search.py  (part 1 of 2)
import sqlite3

import sqlite_vec
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlite_vec import serialize_float32

DB = "rag.db"
EMBED = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
RERANK = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def connect():
    db = sqlite3.connect(DB)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def vector_ids(db, query, n=20):
    q = EMBED.encode(query).tolist()
    sql = f"SELECT rowid FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT {int(n)}"
    return [r[0] for r in db.execute(sql, (serialize_float32(q),)).fetchall()]


def bm25_ids(chunks, query, n=20):
    corpus = [c[3].lower().split() for c in chunks]
    bm = BM25Okapi(corpus)
    scores = bm.get_scores(query.lower().split())
    best = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
    return [chunks[i][0] for i in best]


def rrf(lists, k=60):
    points = {}
    for ranked in lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            points[chunk_id] = points.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(points, key=lambda c: -points[c])
```

- `RERANK = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`: Loaded here at import time even though nothing uses it until step 5. That is why the very first import of search.py sits silent for a minute: it is downloading the second model, not hanging.
- `q = EMBED.encode(query).tolist()`: The question goes through exactly the same model the chunks went through in ingest.py. This is not optional: vectors from two different embedding models live in different number spaces and comparing them produces confident nonsense, so changing the model means rebuilding rag.db.
- `f"SELECT rowid FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT {int(n)}"`: sqlite-vec exposes a hidden `distance` column on a MATCH query and requires a LIMIT so it knows how many neighbours to return. The `int(n)` is not cosmetic: n is pasted straight into the SQL text, and forcing it to an integer is what stops anything other than a number ending up in your query.
- `corpus = [c[3].lower().split() for c in chunks]`: BM25 wants a list of word lists, so each chunk's text (column index 3) is lowercased and split on whitespace. Lowercasing is what makes "Leave" match "leave", and note there is no stemming here, so "leaves" still will not match "leave".
- `bm = BM25Okapi(corpus)`: Builds the entire keyword index from scratch on every single query. Honest simplification: this is work proportional to your whole corpus per question, and it is only acceptable because your corpus is small enough that you will not notice.
- `points[chunk_id] = points.get(chunk_id, 0.0) + 1.0 / (k + rank)`: This one line is the whole of RRF. A chunk collects points from every list it appears in, and a chunk missing from one list simply collects nothing from that list rather than being actively punished.

**The maths, spelled out**

```
Two separate formulas live in this step.

1. Reciprocal Rank Fusion (RRF)

  score(chunk) = sum over each list of  1 / (k + rank)

  rank = the chunk's position in that list, counting from 1
  k    = a constant, 60 here, which flattens the gap between top positions
  a chunk that never appears in a list contributes nothing for that list

Worked example. Chunk A is rank 1 in the vector list and does not appear in the BM25 top 20 at all. Chunk B is rank 4 in the vector list and rank 5 in the BM25 list.

  A = 1 / (60 + 1) = 1 / 61 = 0.01639
  B = 1 / (60 + 4) + 1 / (60 + 5) = 0.015625 + 0.015385 = 0.031010

B wins, even though A was somebody's number one. That is RRF doing its job: agreement between two different searches beats enthusiasm from one.

What k actually does. Compare rank 1 against rank 20:
  with k = 60:  1/61 = 0.01639  versus  1/80 = 0.01250   ->  a ratio of 1.31
  with k = 0:   1/1  = 1.0      versus  1/20 = 0.05      ->  a ratio of 20

So k = 60 says "rank 1 is a bit better than rank 20", while k = 0 would say "rank 1 is twenty times better". 60 comes from the original 2009 paper and is a convention, not a value anyone tuned for your documents.

2. BM25

For each word in your query, BM25 adds this to the chunk's score:

  IDF(word) x ( f x (k1 + 1) ) / ( f + k1 x (1 - b + b x dl / avgdl) )

  f     = how many times that word appears in this chunk
  dl    = length of this chunk in words
  avgdl = average chunk length across the whole index
  k1    = 1.5, controls how quickly repeats stop helping
  b     = 0.75, controls how much short chunks are favoured
  IDF   = ln( (N - n + 0.5) / (n + 0.5) ), a rarity score for the word
  N     = total number of chunks
  n     = number of chunks that contain that word

Worked example. You search for P019. Your index has N = 1000 chunks and only n = 2 of them contain P019. The chunk being scored has f = 2 occurrences, is dl = 200 words long, and the average chunk is avgdl = 180 words.

  IDF         = ln( (1000 - 2 + 0.5) / (2 + 0.5) ) = ln(998.5 / 2.5) = ln(399.4) = 5.99
  length term = 1 - 0.75 + 0.75 x (200 / 180) = 0.25 + 0.8333 = 1.0833
  denominator = 2 + 1.5 x 1.0833 = 2 + 1.625 = 3.625
  numerator   = 2 x (1.5 + 1) = 5
  tf part     = 5 / 3.625 = 1.379
  contribution = 5.99 x 1.379 = 8.26

Now the same arithmetic for the word "the", which appears in n = 900 of the 1000 chunks:

  IDF = ln( (1000 - 900 + 0.5) / (900 + 0.5) ) = ln(100.5 / 900.5) = ln(0.1116) = -2.19

The rarity score goes negative for very common words. rank_bm25 patches negative values up to a small floor so common words cannot drag a score downwards, but the point stands: rare words carry almost the entire score and "the" carries none of it.

Intuitively: BM25 rewards a chunk for containing your rarest words, several times, in a short chunk. It has no idea what any word means, which is precisely why it finds P019 while the meaning model shrugs and hands you general pages about error handling.
```

> **Watch out:** Running python search.py right now prints nothing at all because there is no main block yet, and once you add step 5 it will fail with sqlite3.OperationalError: no such table: vec_chunks unless you have already run python ingest.py from the same folder.

### 5. Rerank the survivors with a cross-encoder

Add this to the bottom of search.py. Everything up to now compared the question and each chunk separately: the question became 384 numbers on its own, each chunk became 384 numbers on its own, and you measured the gap between them. That is fast, and it is crude, because every chunk was turned into numbers months before your question existed and with no knowledge of it. A cross-encoder is a different kind of model that reads the question and one chunk together, in a single pass, and scores how well that specific chunk answers that specific question. It is much more accurate and far too slow to run against 50,000 chunks, so the standard shape is exactly what you see here: retrieve 20 candidates cheaply, rerank only those 20, keep the best 5. Run python search.py what is the warranty period with a question of your own, and read the five chunks it prints before you let any model near them, because if the answer is not visibly sitting in those five chunks then no prompt in step 6 can save you.

```python
# search.py  (part 2 of 2)

def search(query, keep=5, pool=20, use_bm25=True):
    db = connect()
    chunks = db.execute("SELECT id, doc, page, text FROM chunks ORDER BY id").fetchall()
    by_id = {c[0]: c for c in chunks}

    lists = [vector_ids(db, query, pool)]
    if use_bm25:
        lists.append(bm25_ids(chunks, query, pool))

    merged = rrf(lists)[:pool]
    scores = RERANK.predict([(query, by_id[c][3]) for c in merged])
    ranked = sorted(zip(merged, scores), key=lambda p: -p[1])[:keep]
    return [(by_id[c], float(s)) for c, s in ranked]


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "what is this document about"
    for row, score in search(question):
        print(f"--- {row[1]} page {row[2]}  score {score:.2f}")
        print(row[3][:300], "\n")
```

- `chunks = db.execute("SELECT id, doc, page, text FROM chunks ORDER BY id").fetchall()`: Pulls every chunk into memory as tuples of (id, doc, page, text). BM25 needs the whole corpus anyway so nothing is wasted here, but be clear that at a million chunks this single line is the thing that would run out of memory first.
- `by_id = {c[0]: c for c in chunks}`: A lookup table from chunk id to the full row. RRF returns bare ids, so you need this to get the text and page back without firing off one extra SQL query per id.
- `lists = [vector_ids(db, query, pool)] then if use_bm25: lists.append(...)`: This is the switch you flip in step 7. When use_bm25 is False, RRF receives a single list, and its output is then just the vector ranking in its original order, which is what makes the two runs directly comparable.
- `merged = rrf(lists)[:pool]`: RRF returns every chunk it saw, which can be up to 40 distinct ids when two lists of 20 barely overlap. Slicing back down to pool (20) is what caps how much work the slow reranker has to do.
- `scores = RERANK.predict([(query, by_id[c][3]) for c in merged])`: Builds 20 pairs of (question, chunk text) and runs the cross-encoder over all of them in one call. This is by a wide margin the slowest line in the file, and it is where the quality gain comes from.
- `ranked = sorted(zip(merged, scores), key=lambda p: -p[1])[:keep]`: zip pairs each chunk id with its score, the minus sign in the key sorts highest first, and the slice keeps the best 5. float(s) on the next line converts the model's numpy number into a plain Python float so printing and comparing behave normally.

**The maths, spelled out**

```
What is that score, and why is it not a percentage?

The cross-encoder ends in a single output number called a logit. A logit is what a model produces just before it would be squashed into a probability. You convert a logit into a probability with the sigmoid function:

  probability = 1 / (1 + e^(-score))

where e is 2.71828 and score is the number printed on screen.

Worked examples:
  score =  6.2  ->  e^(-6.2) = 0.00203  ->  1 / 1.00203 = 0.998
  score =  0.0  ->  e^(0)    = 1.0      ->  1 / 2.0     = 0.500
  score = -5.0  ->  e^(5.0)  = 148.41   ->  1 / 149.41  = 0.0067

So this model's scores usually run from about -11 (certainly not relevant) up to about +11 (certainly relevant), with 0 sitting at a coin flip. A score of 6.2 is not "6.2 out of 10", and the distance from 8 to 6 is not the same amount of confidence as the distance from 2 to 0. Compare scores against each other and against scores you have personally seen on your own documents, never against some fixed idea of what a good number looks like.

Why only 20 pairs. The cross-encoder runs one full model pass per pair, so the work grows with however many pairs you hand it. On a laptop CPU this small model handles very roughly 30 to 100 pairs a second, and that range depends heavily on your machine:

  20 pairs at 50 a second      = 0.4 seconds per query
  50,000 pairs at 50 a second  = 1,000 seconds, about 17 minutes per query

Intuitively: the cheap search narrows 50,000 chunks down to 20 in under a millisecond, and the expensive model then spends under a second putting those 20 into a much better order. That two stage shape is the biggest single quality gain in this lab for the fewest lines of code.
```

> **Watch out:** The first run of this file pauses for up to a minute with no output while the cross-encoder downloads, so give it time before you assume it has frozen.

### 6. Answer the question with citations

Create ask.py, the third and last file. This is the generation half of RAG, and it is deliberately the smallest part of the lab: it numbers the five surviving chunks, pastes them into one prompt, and tells the model to mark every sentence with the number of the source it came from. The refusal line is a fixed exact string on purpose, so that later you can test for it with a plain string comparison rather than trying to detect whether the model "sounded unsure". Run it with python ask.py "what is the warranty period", keeping the quotes. The sources are printed underneath whatever the model says, together with their rerank scores, so you can always check the answer against the real text on the real page. Be honest with yourself about what this prompt can and cannot do: telling a model "do not use any other knowledge" reduces invented answers, it does not stop them, and the printed source list exists precisely because the instruction is not a guarantee.

```python
# ask.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "_shared"))
from llm import chat  # noqa: E402

from search import search  # noqa: E402

REFUSAL = "I don't have that in these documents."

TEMPLATE = """Answer the question using only the numbered sources below.
After every sentence that uses a source, put its number in square brackets, like [2].
Do not use any other knowledge.
If the sources do not contain the answer, reply with exactly this line and nothing else:
{refusal}

Sources:
{sources}

Question: {question}"""


def answer(question):
    hits = search(question)
    sources = "\n\n".join(
        f"[{i}] {row[1]}, page {row[2]}\n{row[3]}"
        for i, (row, _) in enumerate(hits, start=1)
    )
    reply = chat(TEMPLATE.format(refusal=REFUSAL, sources=sources, question=question))
    print(reply, "\n")
    print("Sources offered to the model:")
    for i, (row, score) in enumerate(hits, start=1):
        print(f"[{i}] {row[1]}, page {row[2]}  (rerank {score:.2f})")


if __name__ == "__main__":
    answer(" ".join(sys.argv[1:]))
```

- `sys.path.append(str(Path(__file__).resolve().parents[1] / "_shared"))`: Adds the my-work/labs/_shared folder to Python's import path so `from llm import chat` works from inside my-work/labs/lab10. parents[1] means "two levels up from this file", which is the labs folder, and resolve() makes it an absolute path so it works whatever directory you launched from.
- `from llm import chat`: The single place this whole course talks to a language model. You switch provider by editing LLM_PROVIDER in your .env file and never by editing this code, which is why no company name appears anywhere in ask.py.
- `REFUSAL = "I don't have that in these documents."`: One constant, used inside the prompt now and used again by the mini-project's checks later. If the string were typed out twice, the two copies would eventually drift apart by one character and your string comparison would silently stop matching.
- `the TEMPLATE string`: Four instructions in a deliberate order: use only these sources, cite with [n], no outside knowledge, and here is the exact line to say when the answer is not present. The sources sit above the question because the question is then the last thing the model reads, which keeps it fresh.
- `f"[{i}] {row[1]}, page {row[2]}\n{row[3]}" with enumerate(hits, start=1)`: Numbers the sources 1 to 5 for the model, and the same numbers are reprinted at the bottom for you, so [3] in the answer means the same thing in both places. Starting at 1 rather than 0 matters because [0] reads as a bug to a human and confuses a model.
- `print(reply, "\n") followed by the source loop`: The sources print every single time, whatever the model said, including when it refused. That unconditional printing is the receipt, and it is the difference between a demo and something you would let another person use.

**The maths, spelled out**

```
How big is this prompt, and what does it cost?

A token is roughly three quarters of an English word. The usual rule of thumb:

  tokens = words x 1.3

Worked example with this lab's defaults (keep = 5 chunks of about 200 words each):

  source text  = 5 x 200 = 1,000 words  ->  1,000 x 1.3 = 1,300 tokens
  template and source headers                          about  100 tokens
  your question                                        about   15 tokens
  total input                                          about 1,415 tokens

At a price of $0.50 per million input tokens:

  1,415 / 1,000,000 x $0.50 = $0.0007, so about seven hundredths of a cent per question.

Now try keep = 20 instead of 5:

  20 x 200 = 4,000 words -> 5,200 tokens of sources, about 5,315 tokens in total
  cost = 5,315 / 1,000,000 x $0.50 = $0.0027, about 3.8 times more

The money is the cheap part of that trade. The expensive part is that answer quality often gets worse rather than better, because the one paragraph that mattered is now buried among nineteen that did not.

Temperature. chat() sends temperature 0.2 by default. Temperature reshapes the model's probabilities for the next word:

  probability(word) = e^(logit / T) / sum of e^(logit / T) over all candidate words

Worked example with just two candidate words, logits 3.0 and 2.0:

  T = 1.0:  e^3 = 20.09, e^2 = 7.39, total 27.48        ->  73.1% and 26.9%
  T = 0.2:  e^15 = 3,269,017, e^10 = 22,026, total 3,291,043  ->  99.33% and 0.67%

Intuitively: a low temperature makes the model far more likely to take its top choice at every step. That is exactly what you want when the job is copying facts out of sources you handed it, and it is why 0.2 rather than 1.0 is the default in this course.
```

> **Watch out:** On Windows run it as python ask.py "your question" with the quotes, because without them a question mark or ampersand is swallowed by the shell, and running it with no question at all sends an empty string to the model and returns confident nonsense.

### 7. Break it on purpose

This step adds no code, and it is the most important one, because a RAG system nobody has tried to break is only a demo. Ask three deliberately different kinds of question and write down what happens each time, in a file you keep. First, a question whose answer you already know sits on one specific page, then check that the printed citation names that page, which proves retrieval and citation are wired together honestly rather than by luck. Second, a question containing an exact code, part number or surname: run it as is, then edit the last section of search.py so it calls search(question, use_bm25=False), run the identical question again, and compare the top five results side by side. Third, a question your PDFs genuinely do not answer, such as the price of a train ticket to Manchester, and watch whether the system prints the exact refusal line or invents something plausible with a citation attached. Retrieval here is fully deterministic, so the same question always returns the same five chunks, and only the model's wording varies between runs. Keep all three results written down, because they are the direct input to the mini-project: you cannot pick a sensible refusal cut-off until you have seen what real scores look like on your own documents.

**The maths, spelled out**

```
Putting a number on "the top five changed"

Two simple ways to compare the two lists of five chunk ids.

  Overlap = (number of ids in both lists) / 5
  Jaccard = (number of ids in both lists) / (number of distinct ids across both lists)

Worked example. With BM25 on you get ids [12, 47, 3, 88, 91]. With BM25 off you get [47, 3, 205, 12, 60].

  ids appearing in both:      12, 47, 3                  -> 3
  distinct ids across both:   12, 47, 3, 88, 91, 205, 60 -> 7

  Overlap = 3 / 5 = 0.60, so 60 percent of the top five survived
  Jaccard = 3 / 7 = 0.43

Intuitively: an overlap of 1.0 means BM25 bought you nothing at all on that question, and an overlap near 0 means the two searches disagree completely about what is relevant. For a plainly worded English question expect a high overlap. For a question containing a part number expect a low one, and that gap is the entire reason hybrid search exists. One number from one question proves nothing, so run three or four questions of each kind before you believe what you are seeing.
```

> **Watch out:** Only the model's wording should differ between two runs of the same question, so if the five retrieved chunks change too, something in your docs/ folder changed and ingest.py rebuilt the index behind you.

## You are done when

You can ask a question in your own words, get back an answer with [1] style markers inside it, then open the named PDF at the named page and put your finger on the exact sentence the answer came from. You can also show one specific query where switching use_bm25 to False changes the top five results, and say by how much (for example three of the five ids survived, an overlap of 0.60), so you have seen with your own eyes what the keyword half actually bought you. And you have three written notes from step 7: the page-specific question with its citation checked, the exact-code question run both ways with both lists recorded, and the unanswerable question with exactly what the system said.

---

## Mini-project: Answer with receipts

Turn the lab into something you would trust: every claim points at chunk text you can read on screen, and the system refuses instead of guessing when your PDFs do not cover the question. You record what it did on ten fixed questions in my-work/labs/lab10/rag_report.json, and check.py verifies that file.

- Fix a test set of ten questions: five your PDFs answer, five they clearly do not (the price of a train ticket to Manchester, the capital of Peru). Give each an id from 1 to 10 and an answerable flag of true or false. check.py reports failures by id, so keep the ids stable.
- Add a refusal gate that runs before chat(). Run search() on all ten questions, record the top rerank score for each, look at the two groups of numbers, and pick one threshold. When the top score falls below it, return the exact line "I don't have that in these documents." and never call the model. check.py recomputes refused == (top_score < threshold) on every row, so one threshold has to explain all ten.
- Make citations carry text. Store each as {"n": 1, "doc": "handbook.pdf", "page": 7, "chunk_text": "..."}, where chunk_text is the full chunk exactly as it sits in rag.db, not a 300 character preview. Every [n] marker in the answer needs a citation with that n. check.py matches chunk_text against rag.db word for word.
- Judge each answered sentence. Send the sentence back with only its cited chunk and ask one yes or no question: does this chunk support this sentence. Record {"sentence": "...", "cite": 1, "supported": true} in a "judged" list, and keep supported a true or false, not a 1 to 5 score.
- Write my-work/labs/lab10/rag_report.json with three top level keys: threshold (a number), questions (the ten rows, each with id, question, answerable, top_score, refused, answer, citations, judged), and rates holding the two rates separately, answered_when_answerable and refused_when_unanswerable, each a fraction out of 5. A refused row carries the exact refusal line as its answer, an empty citations list and an empty judged list.
- Save check.py in my-work/labs/lab10 next to rag_report.json and rag.db, then run python check.py and fix what it names.

### Check it

`check.py` is in this folder. Run it:

```bash
cd my-work/labs/lab10 && python check.py
```


**You are done when** check.py prints one PASS line per check and ends with ALL CHECKS PASSED, exit code 0. A failure names the rows, for example "FAIL every citation carries its chunk text, 50 characters or more -> ids [3, 7]" or "FAIL refused always equals top_score < threshold -> ids [9] disagree with the rule". It also prints a NOT CHECKED line, because whether each answer is factually right and whether your threshold sits in the right place are still yours to judge: read those two yourself.

**If you want more:** Add contextual retrieval. Before embedding, ask a cheap model for one sentence saying where each chunk sits in its document, prepend it, rebuild rag.db, and re-run the same ten questions into rag_report_contextual.json. Point the checker at it by editing REPORT at the top of check.py, then compare the two rates against your first run. Anthropic reported 49% fewer retrieval failures from this on their own data, and it costs one model call per chunk, so try one PDF first.
