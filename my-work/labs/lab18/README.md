# Lab 18: Build and ship the capstone

**Module 18: Capstone - your research and document agent**

You are going to bolt the whole course into one program. It takes a question, searches the documents you indexed back in Module 10, writes a short answer where every fact carries a source label, stops itself if the cost crosses a limit you set, writes down every step it took to a file, and then gets marked pass or fail by a second model. Everything runs on your own laptop with the packages you already installed, so there is no Docker, no server and no graphics card anywhere in this lab. Before you start, run python llm.py and check it prints a working line, because every step from step 5 onwards calls a model and a broken key will look like a broken agent.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Write the spec first, in one screen

A spec is a short written description of what you are building and how you will know it is finished. Write it before any Python, because a goal like "build a research agent" has no end and you will keep adding features until you get bored. Two lines carry most of the weight. The Fixed path line and the Model decides line split the program into the part you write as ordinary code and the one small part where the model chooses what happens next, and that split is the whole reason this program stays debuggable. The Done when line is the only thing in the file that can tell you to stop building. Nothing runs in this step, so all you should see afterwards is a new folder my-work/labs/lab18/capstone/ with one file called SPEC.md in it. Keep that file open in a second window while you build, because every later step maps back to one line in it.

```python
SPEC.md  (plain text, not Python)

Job:      answer questions about the documents already indexed in notes.db.
Input:    one question, plain text.
Output:   under 200 words. Every sentence stating a fact ends with a label
          like [S2]. Plus the list of labels used and the document behind each.
Refuses:  questions the documents do not cover (it says so instead of guessing).
          anything needing the live web. There is no web tool and no send tool.

Fixed path (code):   retrieve -> draft -> check citations -> print -> trace.
Model decides:       which searches to run and when it has enough. Max 4.
Ceiling:             25 cents per question, enforced in code, not in the prompt.

Done when: 8 test questions run, every label in every answer checks out,
           no run exceeds the ceiling, traces/run.jsonl has one line per step,
           and a judge from a different model family mostly agrees with my
           own pass/fail labels.
```

- `Refuses:  questions the documents do not cover (it says so instead of guessing).`: Writing down what the program will not do is what keeps the build small enough to finish. A refusal you planned for is a feature you can test; a refusal nobody planned is a bug report.
- `Fixed path (code):   retrieve -> draft -> check citations -> print -> trace.`: This is a workflow, meaning a path written in code where step two always follows step one. You can read it top to bottom and predict it, so when an answer looks wrong you know which box to open.
- `Model decides:       which searches to run and when it has enough. Max 4.`: This is the only agent part, meaning the only loop where the model picks the next move instead of you. It gets a hard number bolted to it, so even the unpredictable step has a predictable worst case.
- `Ceiling:             25 cents per question, enforced in code, not in the prompt.`: A model can ignore an instruction written in a prompt, because a prompt is just text it is free to reinterpret. A counter in Python that raises an error cannot be talked out of it, so the limit lives in code.
- `Done when: 8 test questions run, every label in every answer checks out,`: Every clause here is something you check by running a command, not an opinion you form while looking at the screen. That is the difference between a finish line and a wish.

**The maths, spelled out**

```
Where does "25 cents per question" come from, and what does it actually buy?

  price per token      = price per million / 1,000,000
  tokens you can afford = ceiling in dollars / price per token

Symbols: ceiling is the dollar limit you chose (0.25 here). Price per million is what your provider charges for a million tokens. A token is a chunk of text roughly three quarters of an English word.

Worked example, using the input price this lab uses later (0.20 dollars per million):
  price per token       = 0.20 / 1,000,000 = 0.0000002 dollars
  tokens you can afford = 0.25 / 0.0000002 = 1,250,000 tokens

So 25 cents buys about 1.25 million input tokens, which is roughly 900,000 English words pushed through the model. Four searches will never come close to that. That is the point: the ceiling is a fence for the run that goes wrong, not a budget you expect to spend.
```

> **Watch out:** The easiest mistake here is treating SPEC.md as paperwork and skipping it, and the tell is simple: if you cannot write the "Done when" line, you do not yet know what you are building.

### 2. Settings, a money counter, and a trace file

Two small files hold everything the rest of the program shares. config.py is just named values in one place, so when the price or the search cap changes you edit one line and every other file sees it. guard.py does two jobs: Budget counts tokens after every model call and raises an error the moment the running cost crosses your ceiling, and trace() writes one line of JSON per step to a file. Without Budget, a loop that keeps deciding it needs one more search keeps spending, and a printed warning is easy to scroll past, so this raises an error instead, which actually stops the program. Without trace() you only ever see the final paragraph, and agents almost always break in the middle, producing a confident answer built on a search that found nothing useful. Nothing prints when you create these two files, so the first thing you will see is a traces/run.jsonl file appearing after step 6 runs. The numbers in PRICE_PER_M are the cheap-tier list price from August 2026, so open your provider's pricing page and replace them with the real ones before you trust any cost you print.

```python
# ---- config.py ----
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
DB_PATH     = ROOT / "notes.db"          # the index you built in Module 10
TRACE_PATH  = ROOT / "traces" / "run.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEARCHES = 4
USD_CEILING  = 0.25
PRICE_PER_M  = {"in": 0.20, "out": 1.20}   # dollars per million tokens

# ---- guard.py ----
import json, time, uuid
from config import TRACE_PATH, USD_CEILING, PRICE_PER_M

RUN_ID = uuid.uuid4().hex[:8]

class OverBudget(Exception):
    pass

class Budget:
    def __init__(self, ceiling=USD_CEILING):
        self.ceiling, self.in_tokens, self.out_tokens = ceiling, 0, 0

    @property
    def usd(self):
        return (self.in_tokens / 1e6) * PRICE_PER_M["in"] + \
               (self.out_tokens / 1e6) * PRICE_PER_M["out"]

    def add(self, usage):
        self.in_tokens  += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.out_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        trace("cost", usd=round(self.usd, 4),
              tokens_in=self.in_tokens, tokens_out=self.out_tokens)
        if self.usd > self.ceiling:
            raise OverBudget(f"stopped at ${self.usd:.4f}, ceiling ${self.ceiling:.2f}")

def trace(step, **fields):
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"run": RUN_ID, "ts": round(time.time(), 3), "step": step}
    row.update(fields)
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
```

- `PRICE_PER_M  = {"in": 0.20, "out": 1.20}   # dollars per million tokens`: Input and output tokens are priced differently, and output is usually several times more expensive per token. Keeping both in one dictionary lets the cost formula below stay a single expression.
- `the usd property`: A property is a method you read like a plain value, so you write budget.usd and not budget.usd(). It recalculates the total from the two counters every time you read it, so the number can never quietly drift out of date.
- `self.in_tokens  += int(getattr(usage, "prompt_tokens", 0) or 0)`: getattr with a default keeps this line working even when the provider hands back no usage object at all, which some local servers do. The `or 0` catches the case where the field exists but is None, and int() makes sure you are adding a number rather than a string.
- `raise OverBudget(f"stopped at ${self.usd:.4f}, ceiling ${self.ceiling:.2f}")`: Raising an exception stops the loop where it stands and hands control back to whoever called it. A print statement would scroll past and the loop would carry on spending your money.
- `row = {"run": RUN_ID, "ts": round(time.time(), 3), "step": step}`: Every traced line carries the same run id and a timestamp. That lets you pull one run out of a file holding fifty of them, and put its steps back in the order they happened.
- `f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")`: One JSON object per line is called JSONL, and it means a crash halfway through still leaves you every line written up to that point. default=str stops json.dumps from blowing up on an object it does not know how to convert.

**The maths, spelled out**

```
The cost formula inside the usd property, written out in plain characters:

  usd = (input tokens / 1,000,000) * price_in + (output tokens / 1,000,000) * price_out

Symbols: input tokens is everything you sent the model on that call (system prompt, the question, and every tool result so far). Output tokens is what the model wrote back. price_in and price_out are dollars per million tokens, 0.20 and 1.20 in this lab.

Worked example for one ordinary question, 12,000 input tokens and 300 output tokens:
  input part  = (12,000 / 1,000,000) * 0.20 = 0.012  * 0.20 = 0.00240 dollars
  output part = (300    / 1,000,000) * 1.20 = 0.0003 * 1.20 = 0.00036 dollars
  usd = 0.00240 + 0.00036 = 0.00276 dollars, about a quarter of one cent

Look at the ratio. The input side is nearly seven times the output side even though each output token costs six times more, because you sent forty times more tokens than you got back. Agent runs are lopsided that way, so the input price is the number that decides your bill, not the headline output price.
```

> **Watch out:** If your provider names its usage fields differently, every cost line will read 0.0 and the ceiling will never fire, so check that the very first cost line in run.jsonl has a non-zero tokens_in.

### 3. One retrieval function over the Module 10 index

This step gives the rest of the program exactly one way in to your documents: pass a string, get back a small list of chunks, each with an id, a document name and its text. A chunk is one small slice of a document, usually a few hundred words, because whole documents are too big to hand to a model in one go. Two different searches run over the same chunks. BM25 is a keyword search that scores a chunk by how many of your query's words it contains and how rare those words are across all your chunks, and vector search compares meaning by turning text into a list of numbers and finding the closest ones. They fail in different ways, so the function merges the two ranked lists with Reciprocal Rank Fusion, which throws away the raw scores and looks only at each chunk's position in each list. Running python retrieve.py should print four lines, each showing a chunk id, a file name and the first 90 characters of the text. Two SQL strings assume the schema from Module 10, a chunks table with id, doc and text, and a vec0 table chunk_vec whose rowid lines up, so if your column names differ, change those two strings and nothing else in the file.

```python
# ---- retrieve.py ----
import re, sqlite3, sqlite_vec
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import DB_PATH, EMBED_MODEL

_model = None

def _embed(text):
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode([text])[0].tolist()

def _open():
    con = sqlite3.connect(DB_PATH)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con

def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def search(query, k=4):
    con = _open()
    rows = con.execute("SELECT rowid, id, doc, text FROM chunks").fetchall()
    by_id   = {r[1]: {"id": r[1], "doc": r[2], "text": r[3]} for r in rows}
    by_row  = {r[0]: r[1] for r in rows}

    bm = BM25Okapi([_words(r[3]) for r in rows])
    scores = bm.get_scores(_words(query))
    keyword = [rows[i][1] for i in sorted(range(len(rows)), key=lambda i: -scores[i])[:20]]

    vec = con.execute(
        "SELECT rowid FROM chunk_vec WHERE embedding MATCH ? ORDER BY distance LIMIT 20",
        (sqlite_vec.serialize_float32(_embed(query)),)).fetchall()
    meaning = [by_row[r[0]] for r in vec if r[0] in by_row]
    con.close()

    fused = {}
    for ranked in (keyword, meaning):                 # Reciprocal Rank Fusion, k=60
        for position, cid in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + position)
    best = sorted(fused, key=lambda c: -fused[c])[:k]
    return [by_id[c] for c in best]

if __name__ == "__main__":
    for hit in search("what did I write about reranking"):
        print(hit["id"], hit["doc"], hit["text"][:90])
```

- `the _embed function and its global _model`: SentenceTransformer has to download and load a model file, which takes several seconds and a few hundred megabytes of memory. Loading it once into a module-level variable means only the first search pays that cost, not all four searches in a run.
- `sqlite_vec.load(con)`: sqlite-vec is an extension that teaches plain SQLite how to store vectors and find nearest neighbours. Without this line SQLite has no idea what chunk_vec is and the vector query fails.
- `bm = BM25Okapi([_words(r[3]) for r in rows])`: BM25Okapi wants every chunk already split into a list of lowercase words, which is exactly what _words returns. This rebuilds the keyword index on every single call, which is fine for a few thousand chunks on a laptop and would not be fine for a million.
- `"SELECT rowid FROM chunk_vec WHERE embedding MATCH ? ORDER BY distance LIMIT 20"`: MATCH is how sqlite-vec asks for nearest neighbours, and distance is smaller when two pieces of text mean similar things. LIMIT 20 keeps the list deep enough that the fusion step has something to work with, since only the top 4 survive at the end.
- `fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + position)`: This one line is the whole of Reciprocal Rank Fusion. It deliberately ignores the raw BM25 score and the vector distance, which live on completely different scales and cannot be sensibly added, and uses only the position in each list.
- `best = sorted(fused, key=lambda c: -fused[c])[:k]`: The minus sign flips the order so the highest score comes first, because Python sorts smallest-first by default. Slicing to k keeps only the four chunks the model will actually be shown.

**The maths, spelled out**

```
Three numbers are hiding in this step. Two of them you never see printed.

1) RECIPROCAL RANK FUSION, the line you can see

  score(chunk) = sum, over each list the chunk appears in, of  1 / (60 + position)

Symbols: position is 1 for the top hit, 2 for the next, and so on. 60 is a constant usually called k, and it flattens the curve so the top result is not worth wildly more than the second. A chunk missing from a list simply adds nothing.

Worked example. Keyword list: A, B, C. Vector list: C, A, D.
  A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
  C: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
  B: 1/(60+2)            = 0.01613
  D: 1/(60+3)            = 0.01587
  Final order: A, C, B, D.
A wins because both searches liked it, even though C was the vector search's own favourite.
Why 60 matters: set k to 0 instead and position 1 scores 1.000 while position 2 scores 0.500, so whichever list you looked at first would dominate everything. With k = 60 the gap between first and second place is under 2 percent, so agreement between the two searches counts for more than being first in one of them.

2) BM25, the keyword score you never see

  score = sum over query words of  idf(word) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * L / Lavg))

Symbols: f is how many times that word appears in this chunk, L is this chunk's length in words, Lavg is the average chunk length, and k1 (about 1.5) and b (about 0.75) are settings rank_bm25 fills in for you. idf means inverse document frequency and is roughly log(N / n), where N is the total number of chunks and n is how many contain that word.

Worked example of just the idf part, with 1,000 chunks:
  "reranking" appears in 5 chunks   -> idf = log(1000 / 5)   = log(200) = 5.30
  "the" appears in 990 chunks       -> idf = log(1000 / 990) = 0.01
  Matching on "reranking" is worth about 500 times more than matching on "the".
That is the whole reason BM25 beats plain word counting: rare words carry the signal.

3) VECTOR DISTANCE, the meaning score you never see

all-MiniLM-L6-v2 turns any text into 384 numbers. Texts about the same thing end up pointing in a similar direction, and the usual measure of that is cosine similarity:

  cosine = (sum of a_i * b_i) / (length of a * length of b)

Worked example using 2 numbers instead of 384, with a = (3, 4) and b = (4, 3):
  top         = 3*4 + 4*3 = 24
  length of a = square root of (9 + 16) = 5, and length of b = 5 as well
  cosine      = 24 / 25 = 0.96, so the two are close in meaning
Cosine runs from 1 (same direction) through 0 (unrelated) to -1 (opposite). sqlite-vec reports a distance rather than a similarity, and the exact distance measure depends on how you created the table back in Module 10, but in every case a smaller distance means closer, which is why the query says ORDER BY distance and not ORDER BY score.
```

> **Watch out:** The very first run downloads the embedding model, so a long silent pause is normal, but an error saying "no such table: chunk_vec" means your Module 10 schema uses different names and you must edit the two SQL strings.

### 4. Put a second door on it: the MCP server

MCP, the Model Context Protocol, is an agreed way for one program to offer tools to another program. This step puts the same search function behind that standard, so any MCP client can search your notes without importing a line of your Python. Watch the import: in the official SDK version 2.0.0 the class is MCPServer, and mcp.server.fastmcp no longer exists, so any tutorial importing FastMCP from there is from the old era (there is also an unrelated package on PyPI called fastmcp, which is a different project by different people). The protocol has been stateless since 2026-07-28, meaning there is no initialize handshake to write and no session id to carry around, so videos showing those steps are out of date too. Keep this tool read-only and give it no way to send anything anywhere, which is the cheapest security decision in the whole module, because a model cannot reliably tell your instructions apart from text that arrived inside one of your documents. Running python server.py prints nothing and looks frozen, and that is correct: it is waiting for a client to talk to it over standard input, so test it with a client script exactly as you did in Module 13.

```python
# ---- server.py ----
from mcp.server import MCPServer          # NOT mcp.server.fastmcp, that module is gone
from retrieve import search

mcp = MCPServer("capstone-notes")

@mcp.tool()
def search_notes(query: str, k: int = 4) -> dict:
    """Search the local notes index. Read only. Returns chunks with source ids."""
    hits = search(query, k=k)
    return {"hits": [{"source_id": h["id"], "doc": h["doc"], "text": h["text"][:800]}
                     for h in hits]}

if __name__ == "__main__":
    mcp.run()        # stdio transport, the default

# Test it the same way you tested Module 13's server.
# Returning a dict gives the client structured output. On the wire that field is
# camelCase (structuredContent); in Python it is snake_case (structured_content).
# For deployment you switch to the Streamable HTTP transport; check the SDK
# README for the exact flag, since standalone HTTP+SSE was removed.
```

- `from mcp.server import MCPServer          # NOT mcp.server.fastmcp, that module is gone`: This is the single line most likely to be wrong if you copy from an older blog post or video. The class moved, and the old import path fails with ModuleNotFoundError rather than anything that explains itself.
- `@mcp.tool()`: The decorator registers the function as a tool that clients can list and call. The SDK reads your type hints (query: str, k: int = 4) and builds the tool's input schema from them, so you never hand-write that JSON.
- `"""Search the local notes index. Read only. Returns chunks with source ids."""`: The docstring is not only for humans. It becomes the tool description the model reads when deciding whether this tool is the right one, so vague wording here shows up later as the model calling the wrong tool or not calling anything.
- `h["text"][:800]`: Truncating each chunk keeps the tool result small. Everything you return travels back into the model as input tokens and gets paid for on that call and on every call afterwards.
- `mcp.run()        # stdio transport, the default`: stdio means the client launches this file as a child process and talks to it over standard input and output. Nothing opens a network port, which is exactly why the program looks frozen when you run it by hand.

**The maths, spelled out**

```
One number here is worth doing arithmetic on: the 800 in h["text"][:800].

  tokens is roughly characters / 4                 (English prose, rough rule)
  tool result tokens is roughly k * chunk characters / 4

Symbols: k is how many chunks you return, 4 by default. The 4 characters per token is an average for ordinary English; code, URLs and unusual words come out worse than that.

Worked example with 4 chunks of 800 characters each:
  4 * 800 = 3,200 characters
  3,200 / 4 = about 800 tokens for that one tool result
  cost of sending it once, at 0.20 dollars per million input tokens:
  800 * 0.20 / 1,000,000 = 0.00016 dollars, about one sixtieth of a cent
And it is resent on every later turn of the loop, so multiply by the number of turns that follow it.

Intuitively the truncation number is a dial with cost on one side and evidence on the other. Raise it to 4,000 and each result becomes about 1,000 tokens per chunk, five times the price, and you also give the model more room to lose the one useful sentence inside a wall of text.
```

> **Watch out:** python server.py sitting silently is the stdio transport waiting for a client, not a hang, so drive it from a client script instead of staring at an empty terminal.

### 5. The loop: fixed everywhere except the searching

This is the agent, and it is the only place in the program where the model decides what happens next. It works as a loop: you send the conversation, the model either asks to call the search tool or writes the final answer, and if it asked for a search you run the search, paste the results back in as a new message, and go round again. A plain integer counter caps it at four searches, and once the cap is reached the tools list is not even sent, so the model has nothing to call and has to answer. Every chunk gets a short label like S1 or S2 the first time it is seen, and the model is told to cite those labels, which is the thing that makes the citations checkable by ordinary code in the next step. Without the cap, a model that keeps deciding it needs one more search will run until the money ceiling fires, which works but is a much worse way to stop. When you run this through step 6 you should see an answer printed with labels in square brackets, and traces/run.jsonl should hold one search line per search the model made. The two lines marked ADAPT read the OpenAI-style response shape, which is what the shared llm.py returns, so leave them alone unless you swapped that helper out.

```python
# ---- agent.py ----
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from llm import chat_raw                       # ADAPT if your helper differs

from config import MAX_SEARCHES
from guard import Budget, trace
from retrieve import search

TOOLS = [{"type": "function", "function": {
    "name": "search_notes",
    "description": "Search the local notes. Returns chunks with source labels.",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}]

SYSTEM = (
    "You answer questions using only the search_notes tool.\n"
    f"You may search at most {MAX_SEARCHES} times, then you must answer.\n"
    "Every sentence that states a fact ends with a source label in square "
    "brackets, like [S2]. Use only labels that appeared in tool results.\n"
    "If the notes do not answer the question, say so plainly.\n"
    "Keep the answer under 200 words.")

def research(question, budget=None):
    budget = budget or Budget()
    labels, label_of, searches = {}, {}, 0

    def label_for(hit):
        if hit["id"] not in label_of:
            name = f"S{len(labels) + 1}"
            label_of[hit["id"]] = name
            labels[name] = hit
        return label_of[hit["id"]]

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]

    for turn in range(MAX_SEARCHES + 2):
        if searches < MAX_SEARCHES:
            resp = chat_raw(messages=messages, tools=TOOLS)
        else:
            resp = chat_raw(messages=messages)
        msg = resp.choices[0].message                        # ADAPT
        budget.add(getattr(resp, "usage", None))
        calls = getattr(msg, "tool_calls", None) or []

        if not calls:
            answer = msg.content or ""
            trace("answer", turn=turn, words=len(answer.split()), searches=searches)
            return {"question": question, "answer": answer, "searches": searches,
                    "labels": {k: {"doc": v["doc"], "text": v["text"]}
                               for k, v in labels.items()},
                    "usd": round(budget.usd, 4)}

        messages.append(msg.model_dump(exclude_none=True))   # ADAPT
        for call in calls:
            args = json.loads(call.function.arguments or "{}")
            query = args.get("query") or question
            hits = search(query, k=4)
            searches += 1
            trace("search", turn=turn, query=query, hits=[h["id"] for h in hits])
            block = "\n\n".join(f"[{label_for(h)}] ({h['doc']}) {h['text'][:600]}"
                                for h in hits)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": block or "No matching notes."})

        if searches >= MAX_SEARCHES:
            messages.append({"role": "user",
                             "content": "Search budget spent. Write the final answer now."})

    trace("gave_up", searches=searches)
    return {"question": question, "answer": "", "labels": {},
            "searches": searches, "usd": round(budget.usd, 4)}
```

- `the TOOLS list`: This is a JSON description of one tool: its name, a sentence telling the model when to use it, and the shape of its arguments. The model never sees your Python, only this description, so this text is your only lever on when it decides to search.
- `the if searches < MAX_SEARCHES branch`: Once the cap is hit, the tools list is simply left out of the call, so the model is not being politely asked to stop searching, it has no tool left to call. Removing the option in code beats asking for restraint in a prompt every time.
- `def label_for(hit):`: Each chunk id gets a short name the first time it turns up, and keeps that same name for the rest of the run. Short labels are easy for a model to copy without mangling and easy for a regular expression to find afterwards.
- `for turn in range(MAX_SEARCHES + 2):`: A for loop rather than a while loop means the function physically cannot spin forever, even if every other guard fails. The extra 2 leaves room for the final answer turn plus one wasted turn.
- `messages.append(msg.model_dump(exclude_none=True))   # ADAPT`: The model's own tool-call message has to go back into the conversation before the tool results do, or the next call sees answers to a request that was never made and the provider rejects it. exclude_none drops empty fields that some providers refuse to accept.
- `messages.append({"role": "tool", "tool_call_id": call.id,`: The tool_call_id ties this result back to the exact request the model made, which matters when it asks for two searches in a single turn. The content is plain text with the labels already stitched in, so the model reads each label right next to the text it belongs to.

**The maths, spelled out**

```
Why the cap is 4 and not 40, in tokens and dollars.

Every call resends the whole conversation, so the input grows on every turn. If each search adds S tokens of tool result, the input sizes look like this:

  call 1: B
  call 2: B + S
  call 3: B + 2S
  ...
  call n: B + (n-1)S
  total input over the whole run = n*B + S * (0 + 1 + 2 + ... + (n-1))
                                 = n*B + S * n * (n-1) / 2

Symbols: B is the fixed base, meaning the system prompt plus the question. S is how many tokens one tool result adds. The n*(n-1)/2 part is a triangular number, and it is the reason cost grows with the square of the number of turns rather than in a straight line.

Worked example with B = 200, S = 800, and 5 calls (4 searches, then the answer):
  total input = 5*200 + 800 * 5 * 4 / 2 = 1,000 + 8,000 = 9,000 tokens
  at 0.20 per million: 9,000 * 0.20 / 1,000,000 = 0.0018 dollars

Now double the cap to 8 searches, so 9 calls:
  total input = 9*200 + 800 * 9 * 8 / 2 = 1,800 + 28,800 = 30,600 tokens
  that is about 3.4 times the cost for twice the searches

Intuitively: doubling the search cap does not double your bill, it roughly quadruples it. That is why the cap is a small number written in code, and why the money ceiling sits behind it as a second net.
```

> **Watch out:** If you delete or break the ADAPT line that appends msg back into messages, the very next call fails with a complaint about a tool message that has no matching tool call, which reads like an API bug but is your message order.

### 6. Check the citations in code, then run one question

Now you verify the citations with ordinary code instead of trusting the prompt to have been obeyed. Three checks run, and all three are deterministic, meaning they give the same answer every time with no model involved. First, every label the answer uses must be one you actually handed over, which catches the model writing [S7] when only S1 to S4 ever existed. Second, every sentence longer than 40 characters must carry a label, which catches confident factual sentences with no source attached. Third, the sentence and the chunk it points at must share at least 20 percent of the sentence's content words, which catches an answer pointing at a real chunk about a completely different topic. Be honest about that third check: it is the weak one, and a careful-sounding wrong paraphrase of the right chunk will sail straight past it. Running python cite.py calls the agent once and prints the answer, then the cost and the search count, then a dictionary with either ok True or a list of problems.

```python
# ---- cite.py ----
import re

TAG = re.compile(r"\[(S\d+)\]")

def _content_words(s):
    return {w for w in re.findall(r"[a-z]{4,}", s.lower())}

def check(answer, labels):
    problems = []
    for tag in sorted(set(TAG.findall(answer))):
        if tag not in labels:
            problems.append(f"{tag} was never retrieved")

    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        sentence = sentence.strip()
        if len(sentence) < 40:
            continue
        tags = TAG.findall(sentence)
        if not tags:
            problems.append(f"no source label: {sentence[:60]}...")
            continue
        for tag in tags:
            if tag not in labels:
                continue
            words = _content_words(sentence) - {tag.lower()}
            shared = words & _content_words(labels[tag]["text"])
            if words and len(shared) / len(words) < 0.20:
                problems.append(f"{tag} looks unrelated to: {sentence[:60]}...")

    return {"ok": not problems, "problems": problems,
            "used": sorted(set(TAG.findall(answer)))}

if __name__ == "__main__":
    from agent import research
    r = research("What did I write about reranking?")
    print(r["answer"], "\n")
    print(f"${r['usd']}  searches={r['searches']}")
    print(check(r["answer"], r["labels"]))
```

- `TAG = re.compile(r"\[(S\d+)\]")`: The square brackets are escaped with backslashes because brackets already mean something else inside a regular expression. The round brackets capture just the S and its digits, so findall gives you S2 rather than [S2].
- `re.split(r"(?<=[.!?])\s+", answer)`: This cuts the answer into sentences at the whitespace that follows a full stop, question mark or exclamation mark. (?<=...) is a lookbehind, meaning it matches the position after those characters without eating them, so the punctuation stays attached to the sentence.
- `if len(sentence) < 40:`: Short fragments like "Here is what I found." are not factual claims, so demanding a source label on them would fill your problems list with noise. 40 characters is a rough dividing line you chose, and you can move it.
- `words = _content_words(sentence) - {tag.lower()}`: _content_words keeps only runs of 4 or more letters in lowercase, which quietly drops "the", "and", numbers and punctuation. Subtracting the tag stops the label itself, such as s2, from counting as a word the sentence and the chunk happen to share.
- `if words and len(shared) / len(words) < 0.20:`: The `words and` part guards against dividing by zero when a sentence has no long words at all. 0.20 is a threshold you picked rather than a law, so lowering it makes the check quieter and raising it produces more false alarms.
- `return {"ok": not problems, "problems": problems,`: An empty list is falsy in Python, so `not problems` is True exactly when nothing went wrong. Returning the verdict and the list together means the caller can print a reason instead of just a red light.

**The maths, spelled out**

```
The third check is one fraction.

  overlap = (content words shared with the cited chunk) / (content words in the sentence)
  flag a problem when overlap < 0.20

Symbols: a content word here is any run of 4 or more letters, lowercased. Short words such as "the" and "and" are dropped on purpose, because nearly every sentence contains them and they would inflate the score for free.

Worked example. Sentence: "Reranking pushes the best chunks to the top after retrieval [S2]."
  content words = {reranking, pushes, best, chunks, after, retrieval}  -> 6 words
  the chunk behind S2 contains reranking, chunks and retrieval          -> 3 shared
  overlap = 3 / 6 = 0.50, which is above 0.20, so no problem is reported

Same sentence, but pointed at a chunk about installing sqlite-vec that only shares "chunks":
  overlap = 1 / 6 = 0.167, which is below 0.20, so it is flagged as "looks unrelated"

Intuitively this measures topic overlap and nothing else. It is a smoke alarm, not a fact checker: a sentence that reuses the chunk's vocabulary while stating the exact opposite of what the chunk says scores high and passes cleanly.
```

> **Watch out:** Expect a few false alarms from the 0.20 rule on answers that paraphrase well, so read the problems list before you conclude the agent is broken.

### 7. Run eight questions, label them yourself, then let a judge disagree

Two files here. run_batch.py runs every question, checks the citations, prints a one-line summary per question and writes the whole lot to answers.json. Then you do the part most tutorials skip: open answers.json and set my_label to "pass" or "fail" on every row with your own eyes, because without your labels the judge's score is a number with nothing to be measured against. judge.py then reads that file, asks a model to grade each answer PASS or FAIL, and counts how often the model agreed with you. Before you run it, change LLM_PROVIDER in your .env to a different model family than the one that wrote the answers, because a model grades its own family generously by roughly 10 to 25 percent, and the script prints whoami() at the top so you can confirm the switch actually took. Read the two rates separately: a lazy judge that says PASS to everything scores 100 percent on one rate and 0 percent on the other, and a single agreement percentage would hide that completely. You should finish with one printed line per question showing your label beside the judge's, then the two rates underneath.

```python
# ---- run_batch.py ----
import json
from pathlib import Path
from agent import research
from cite import check
from guard import Budget, OverBudget

QUESTIONS = [
    "What did I write about reranking?",
    "Which vector store does the course use, and why that one?",
    "What is the cheapest way to cut retrieval failures?",
    "What is the population of Mars?",          # should refuse
    # add four more of your own, including at least one more it must refuse
]
OUT = Path(__file__).with_name("answers.json")

rows = []
for q in QUESTIONS:
    try:
        r = research(q, budget=Budget())
    except OverBudget as e:
        r = {"question": q, "answer": "", "labels": {}, "usd": None, "error": str(e)}
    r["citations"] = check(r["answer"], r.get("labels", {}))
    r["my_label"] = None            # you fill this in by hand: "pass" or "fail"
    rows.append(r)
    print(f"${str(r.get('usd')):7s} cites_ok={r['citations']['ok']}  {q[:55]}")

OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nNow open {OUT.name} and set my_label on every row.")

# ---- judge.py ----
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_shared"))
from llm import chat, whoami                    # ADAPT

RUBRIC = ("Grade one answer. Reply with one word: PASS or FAIL.\n"
          "PASS needs all three: it answers the question, every factual sentence "
          "ends with a source label, and it claims nothing missing from the sources.\n"
          "An honest 'the notes do not cover this' is a PASS.\n")

rows = json.loads(pathlib.Path(__file__).with_name("answers.json").read_text(encoding="utf-8"))
print("judging with:", whoami())
tp = fn = tn = fp = 0

for r in rows:
    if r["my_label"] not in ("pass", "fail"):
        raise SystemExit("Label every row by hand first.")
    sources = "\n".join(f"[{k}] {v['text'][:400]}" for k, v in r["labels"].items())
    prompt = (f"{RUBRIC}\nQUESTION: {r['question']}\n\nSOURCES:\n{sources or '(none)'}"
              f"\n\nANSWER:\n{r['answer'] or '(empty)'}\n\nPASS or FAIL:")
    verdict = "pass" if chat(prompt).strip().upper().startswith("PASS") else "fail"
    print(f"mine={r['my_label']:4s} judge={verdict:4s}  {r['question'][:50]}")
    if r["my_label"] == "pass":
        tp += verdict == "pass"
        fn += verdict == "fail"
    else:
        tn += verdict == "fail"
        fp += verdict == "pass"

print(f"\ntrue positive rate: {tp}/{tp + fn}   (of the ones I passed)")
print(f"true negative rate: {tn}/{tn + fp}   (of the ones I failed)")
```

- `"What is the population of Mars?",          # should refuse`: You need questions your documents cannot answer, because a system that never refuses is not honest, it has just been lucky about which questions you asked it. At least two of your eight should be things it must decline.
- `except OverBudget as e:`: One question hitting the ceiling should record a row and move on, not kill the whole batch. The row is kept with an empty answer, so the failure shows up in your results instead of quietly disappearing.
- `r["my_label"] = None            # you fill this in by hand: "pass" or "fail"`: The field exists but is deliberately empty, so the file itself reminds you that the hand labelling is still owed. judge.py refuses to run until every row has been filled in.
- `sources = "\n".join(f"[{k}] {v['text'][:400]}" for k, v in r["labels"].items())`: The judge is shown the same chunks the agent was shown, so it can tell whether a claim really appears in the sources. Without this the judge would only be marking whether the answer reads nicely.
- `verdict = "pass" if chat(prompt).strip().upper().startswith("PASS") else "fail"`: The rubric asks for one word but models add politeness anyway, so this reads only the start of the cleaned-up reply. Anything that is not clearly PASS counts as fail, which is the safer default for a grader.
- `tp += verdict == "pass"`: In Python True is 1 and False is 0, so adding a comparison straight onto a counter works and stays short. tp, fn, tn and fp are the four boxes every row falls into once your label is compared with the judge's.

**The maths, spelled out**

```
Four counters and two rates.

Every row lands in one of four boxes once you compare your label with the judge's:
  tp (true positive)  = you said pass, judge said pass
  fn (false negative) = you said pass, judge said fail
  tn (true negative)  = you said fail, judge said fail
  fp (false positive) = you said fail, judge said pass

  true positive rate = tp / (tp + fn)     how much of your good work the judge recognises
  true negative rate = tn / (tn + fp)     how much of your bad work the judge catches

Worked example. Eight questions. You passed 6 and failed 2. The judge passed all eight.
  tp = 6, fn = 0, tn = 0, fp = 2
  true positive rate = 6 / 6 = 1.00, which looks perfect
  true negative rate = 0 / 2 = 0.00, which is useless
  raw agreement      = (tp + tn) / 8 = (6 + 0) / 8 = 0.75

That 0.75 sounds respectable and tells you nothing. A judge that says PASS to everything scored 75 percent, purely because most of your answers were good. That is the whole reason the script prints two rates instead of one number.

One more number from the module notes: a judge marks answers from its own model family about 10 to 25 percent more generously. On 8 rows, 10 to 25 percent is one or two rows, which is enough to move both rates on a set this small, so switch LLM_PROVIDER before you judge anything.
```

> **Watch out:** judge.py exits with "Label every row by hand first" if any my_label is still null, and if the whoami() line prints the same provider that wrote the answers, both rates are flattering you rather than measuring you.

## You are done when

Running python run_batch.py prints one line per question, every line shows a cost under 0.25, cites_ok=True appears on every answer that has content, and the questions your documents do not cover come back as a plain refusal instead of a guess. traces/run.jsonl exists and holds one line per search, one per cost update and one per answer, all sharing the same run id. After you have hand labelled every row in answers.json, python judge.py runs with a different LLM_PROVIDER than the one that wrote the answers, prints your label next to the judge's for every question, and ends with a true positive rate and a true negative rate. You can point at one row where the judge disagreed with you and say out loud why it did.

---

## Mini-project: Ship it

Deploy the capstone to a free host and write the README that gets a stranger from a cold laptop to a cited answer. The artefact is my-work/labs/lab18/ship/RELEASE.json, a record of what you deployed, what it costs and what it cannot do.

- Make the folder my-work/labs/lab18/ship/ and copy config.py, guard.py, retrieve.py, agent.py and cite.py into it from the lab. Change config.py so both limits come from the environment, with tighter defaults than your laptop uses: USD_CEILING = float(os.environ.get("USD_CEILING", "0.05")) and MAX_SEARCHES = int(os.environ.get("MAX_SEARCHES", "2")). Nobody is watching the deployed copy, so it gets the smaller numbers.
- Deploy to one of Modal, Hugging Face Spaces, Cloudflare Workers AI or Render. Get a hello-world running there first, before your agent goes near it. Put your API key in the host's secret store, add a .env.example holding placeholder values, and leave no .env file in ship/.
- Decide what ships with it. Either include a small sample index you built, or make the first run build one from the user's own files. Record the choice as index_mode, either "sample" or "user_builds".
- Write ship/README.md with these seven headings, in this order: ## What this does, ## Install, ## Build the index, ## Run it, ## Example, ## Cost, ## What it cannot do. Under Example, paste a real question and the real answer your deployed copy gave, source labels included.
- Run the stranger test. Hand the URL and the README to someone who has never seen the project, on a machine you have not touched, and time them. Fix the README everywhere they stopped, and keep the list of fixes.
- Write ship/RELEASE.json with exactly this shape: {"host": "Modal", "public_url": "https://...", "index_mode": "sample", "deployed_usd_ceiling": 0.05, "deployed_max_searches": 2, "cost_per_question_usd": 0.0015, "example": {"question": "...", "answer": "... [S1].", "labels": {"S1": {"doc": "rag-notes.md"}}}, "cannot_do": ["...", "...", "..."], "stranger_test": {"tested_by": "Priya", "minutes": 7, "readme_fixes": ["..."]}}. Each cannot_do entry needs at least 15 characters, so "no web" will not pass.

### Check it

`check.py` is in this folder. Run it:

```bash
cd my-work/labs/lab18/ship && python check.py
```


**You are done when** python check.py prints 25 PASS lines and ends with ALL CHECKS PASSED and exit code 0. It also prints one "not checked automatically" line covering the three things a program cannot see: whether your URL is live, whether the README reads well, and whether the host really enforces the ceiling. Open your own URL, ask your example question, and settle those three yourself.

**If you want more:** Add one tool that reaches outside your own documents, such as fetching a web page. You have now completed Simon Willison's lethal trifecta: private data, untrusted content, and a way to send data out. Remove one leg in code, not in the prompt, then add a trifecta_leg_removed field to RELEASE.json naming the leg and the exact line that removes it.
