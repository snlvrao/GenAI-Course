# Lab 08: Pack a context window

**Module 8: Context engineering**

You are going to build a context packer: it takes six competing sources of text, a fixed token budget and a priority order, then decides what goes into the prompt and what gets cut. The part that matters most is the report, because it prints exactly what was dropped and why, with the numbers behind each decision. You will run the same question at three budgets and watch a correct answer turn into "I do not know", purely because your packing threw away the one document that held the answer. Before you start, make sure python llm.py works, see setup.html.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Open the folder and check your helper

This step proves your shared helper is reachable before you write any real code. The helper lives at my-work/labs/_shared/llm.py and it is the only file in this course that talks to a language model, so if Python cannot import it, nothing later in the lab will run. The one-line command adds that shared folder to Python's import search path, imports whoami, and prints it. You should see a single line naming your provider, your model, the endpoint address and whether your key is set, something like provider=groq  model=llama-3.1-8b-instant  endpoint=https://api.groq.com/openai/v1  key=set. If it says key=MISSING instead, stop and fix your .env file now, because every model call in step 6 will fail the same way. Doing this check first means that when something breaks later, you already know it is your packing code and not your setup. Note that this lab, like every other module, switches provider by editing LLM_PROVIDER in .env, never by editing code.

```python
cd my-work\labs\lab08
python -c "import sys; sys.path.append('../_shared'); from llm import whoami; print(whoami())"
```

- `cd my-work\labs\lab08`: Moves you into the folder for this lab, which the course already ships. Every command in this lab assumes you are standing here, and the relative path '../_shared' on the next line only points at the right place from inside it.
- `sys.path.append('../_shared')`: Python only imports from folders it already knows about. This adds the shared folder one level up to that list, so the import on the same line finds my-work/labs/_shared/llm.py without you copying that file into every lab folder.
- `from llm import whoami; print(whoami())`: whoami reports which provider, model and endpoint you are about to use and whether your key is present. It makes no network call and costs nothing, so it is a safe check to run whenever you are confused about what is configured.

> **Watch out:** If you get ModuleNotFoundError: No module named 'llm' you are not standing inside the new folder, and if the line ends with key=MISSING your .env was not found, so fix both here rather than discovering them in step 6.

### 2. Write the six sources

Here you write the raw material, six separate layers of text that all want space in the same context window. On purpose there is more material here than the smallest budget can hold, because this lab is about choosing, and nothing forces a choice when everything fits. Three traps are planted deliberately, and you should leave all three exactly as they are. First, the document that actually answers the question sits third in DOCS, not first, so a packer that keeps only the top retrieval hits loses the answer. Second, two memory notes are near copies of each other, which is what a real memory file looks like after a few months, and packing both wastes tokens for no gain. Third, the note dated 2025-11-02 is stale: it is still on topic and it is no longer true, which is the failure the module calls temporal drift. Nothing runs yet, this file only holds data, and step 3 imports it.

```python
# my-work/labs/lab08/sources.py

SYSTEM = (
    "You are a support assistant for the Northwind billing system. "
    "Answer only from the context you are given. If the answer is not in the "
    "context, say exactly: I do not know from the context I was given."
)

QUESTION = (
    "A customer paid twice for invoice INV-4471. How many days do I have to "
    "issue the refund, and do I need an approval code?"
)

TOOLS = [
    "lookup_invoice(invoice_id: str) -> dict. Returns the invoice record including status, amount_cents, currency and the customer id.",
    "list_payments(customer_id: str) -> list[dict]. Returns every payment the customer has made, newest first, with charge dates.",
    "refund(invoice_id: str, amount_cents: int, approval_code: str = '') -> dict. Issues a refund against an invoice.",
    "search_kb(query: str) -> list[str]. Full text search over the internal support knowledge base. Returns up to ten passages.",
    "send_email(to: str, subject: str, body: str) -> dict. Sends an email to the customer from the support address.",
    "open_dispute(invoice_id: str, reason: str) -> dict. Moves an invoice into the disputes queue and assigns it to finance.",
]

DOCS = [  # retrieved passages, best match first
    "Refund policy overview. Refunds are returned to the original payment method used for the charge. Card refunds appear on the customer statement in three to five business days. Bank transfer refunds take five to ten business days. The customer does not need to return anything for a billing-only refund. Partial refunds are allowed on any invoice that has been paid in full. Refund records are written to the audit log and cannot be edited afterwards.",
    "Duplicate payment handling. A duplicate payment is two successful charges against the same invoice number inside a rolling seven day period. The billing system does not block these automatically, because some customers pay part of an invoice twice on purpose. Confirm with the customer that the second charge was not intended before you take any action. Always check list_payments first.",
    "Refund window and approval. A refund for a duplicate payment must be issued within 30 days of the date of the second charge. After 30 days the case moves to the finance team and support can no longer issue it. Any refund over 500 USD requires a supervisor approval code entered at the time of the refund. Refunds of 500 USD or less need no approval code.",
    "Chargebacks. If the customer's bank has already opened a chargeback, stop. Do not refund a charge that is under dispute, because the account will be debited twice. Chargeback cases belong to the disputes queue. Add a note to the ticket and reassign it.",
]

HISTORY = [  # oldest first
    "user: Hi, I think one of my customers has been charged twice.",
    "assistant: I can help with that. Do you have the invoice number?",
    "user: It is INV-4471. The amount was 240 USD each time.",
    "assistant: Thanks. I looked it up. There are two successful charges on INV-4471, on the 3rd and the 4th.",
    "user: The customer says the second one was a mistake, they clicked pay twice.",
    "assistant: Understood. I will check what the refund process is for a duplicate charge.",
]

MEMORY = [  # notes carried from earlier sessions, with the date each was written
    "2026-02-11: This agent handles the EU billing region. Amounts are shown in the invoice currency, not converted.",
    "2026-02-11: This agent handles the EU billing region. Amounts are shown in the invoice currency, not converted. (re-saved)",
    "2025-11-02: The staging billing API is at 10.0.0.4. Do not use it for real refunds.",
    "2026-06-30: Supervisor approval codes are requested in the #billing-approvals channel and expire after one hour.",
]
```

- `say exactly: I do not know from the context I was given.`: Gives the model one fixed sentence to produce when the context does not contain the answer. Because the wording is fixed, you can check for it in the output, and that is how you will spot the exact moment your packing starved the model.
- `DOCS = [  # retrieved passages, best match first`: The list arrives in relevance order from a retriever, so index 0 is the best scoring match. That ordering is what makes the planted trap work, because a packer that trusts the ranking blindly keeps the top passages and loses the useful one.
- `Refund window and approval. A refund for a duplicate payment must be issued within 30 days`: This third passage is the only one that contains both numbers the question asks for, the 30 day window and the 500 USD approval threshold. Keep your eye on this exact item, because it is the one you will watch disappear from the DROP lines later.
- `the two 2026-02-11 memory notes`: They are the same sentence apart from a trailing (re-saved) marker. They exist so your duplicate check in step 4 has something realistic to catch, because memory files genuinely fill up with near copies over time.
- `2025-11-02: The staging billing API is at 10.0.0.4.`: This note is old, still on topic, and probably no longer true. It is here so you can see that the date stamp is the only thing separating a useful note from a misleading one, since the model cannot tell them apart on its own.
- `HISTORY = [  # oldest first`: The conversation is stored in normal reading order. Step 3 reverses it when packing so the newest turns are protected first, and step 5 flips it back for display, so the order changes twice on purpose.

**The maths, spelled out**

```
Formula: layer_tokens = the sum of est_tokens(item) for every item in that layer, counting roughly four characters per token (the rule you write in step 3).

What the symbols mean: est_tokens(item) is the estimated token cost of one string. layer_tokens is what that whole layer costs if you pack all of it.

Worked example, the four retrieved documents. Their lengths are 449, 387, 354 and 252 characters. Divide each by four and round up: 449 becomes 113, 387 becomes 97, 354 becomes 89, 252 becomes 63. Add them: 113 + 97 + 89 + 63 = 362 tokens.

Doing the same for every layer gives system 50, question 30, tools 181, docs 362, history 114, memory 108. Total = 50 + 30 + 181 + 362 + 114 + 108 = 845 tokens.

What it means: docs are 362 of the 845 tokens, which is 100 * 362 / 845 = 43 percent of everything you have, so retrieval is the layer most likely to eat your budget. The passage holding the answer is 89 tokens, which costs about the same as four old chat turns, so this is a genuine trade and not an obvious one.
```

> **Watch out:** If you save sources.py anywhere other than next to packer.py, step 3 fails with ModuleNotFoundError: No module named 'sources', which reads like a code bug and is really a file location problem.

### 3. Count tokens and define one item

Now you start packer.py, and the first thing it needs is a way to measure size. Models charge and limit by tokens, which are the sub-word pieces a model chops text into, not by characters and not by words. The exact count depends on which model's tokeniser you use, and installing one just for this lab is not worth it, so you approximate at four characters per token. Write the reason in the docstring, because a number that looks exact but is not causes bugs later when someone trusts it as a billing figure. The Item dataclass is the second half of this step: it holds one packable thing, which layer it came from, its text, its priority where 0 is most important, and whether it is pinned. Pinned means it goes in no matter what, which you use for the system prompt and the question, because a run missing either of those is meaningless. build_items flattens the six lists into one list of 22 Items, and once step 6 prints the total you should see 845 tokens of material available.

```python
# my-work/labs/lab08/packer.py
import re
from dataclasses import dataclass
from sources import SYSTEM, QUESTION, TOOLS, DOCS, HISTORY, MEMORY


def est_tokens(text: str) -> int:
    """Rough estimate. Real tokenisers differ per model. This is close enough
    to reason about budgets and it needs no extra install."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class Item:
    source: str        # one of: system, question, tools, docs, history, memory
    text: str
    priority: int      # 0 is most important
    pinned: bool = False

    @property
    def tokens(self) -> int:
        return est_tokens(self.text)


def build_items():
    items = [Item("system", SYSTEM, 0, pinned=True),
             Item("question", QUESTION, 0, pinned=True)]
    items += [Item("tools", t, 1) for t in TOOLS]
    items += [Item("docs", d, 2) for d in DOCS]
    items += [Item("history", h, 3) for h in reversed(HISTORY)]  # newest first
    items += [Item("memory", m, 4) for m in MEMORY]
    return items
```

- `return max(1, (len(text) + 3) // 4)`: Counts the characters and divides by four, adding 3 first so the division rounds up rather than down. The max(1, ...) makes sure nothing is ever counted as zero tokens, which would otherwise let empty strings into the prompt for free.
- `@dataclass`: Generates the constructor and the printable representation from the field names below it, so you write four lines instead of fifteen. It is a convenience only, and the class behaves like any ordinary Python class.
- `the tokens property`: Marking it with @property means you read it.tokens like an attribute, but the value is recomputed from the current text every time. That way you can edit an item's text and the count stays correct, with no stale stored number to forget about.
- `Item("system", SYSTEM, 0, pinned=True)`: Priority 0 and pinned=True both matter, and they do different jobs. Priority decides where it sits in the sort, while pinned makes the packer skip every rejection check, so the system prompt and the question are guaranteed to be in the prompt at any budget.
- `items += [Item("history", h, 3) for h in reversed(HISTORY)]  # newest first`: Reversing before packing means the most recent turns are considered first. If the budget runs out part way through, it is the oldest turns that get cut, which is almost always the right thing to lose in a conversation.

**The maths, spelled out**

```
Formula: tokens = max(1, (characters + 3) // 4)

What the symbols mean: characters is len(text), the number of characters in the string. // is integer division in Python, which divides and throws the remainder away. Adding 3 before dividing by 4 turns rounding down into rounding up. max(1, ...) means the result is never below 1.

Worked example one, the SYSTEM string, which is 199 characters long. 199 + 3 = 202. 202 // 4 = 50, because 4 times 50 is 200 and the leftover 2 is discarded. So 50 tokens. Checking it by hand: 199 / 4 = 49.75, and rounded up that is 50, the same answer.

Worked example two, showing why the +3 is there. Take a 5 character string. Without the +3 you get 5 // 4 = 1, which undercounts. With it you get (5 + 3) // 4 = 8 // 4 = 2, which is right.

What it means: ordinary English text costs roughly one token per four characters, so dividing the character count by four is close enough to plan a budget with.

Being honest about the simplification: real tokenisers split text into learned sub-word pieces, so the true count depends on which model you call. Code, JSON, long numbers and non-English text pack worse, sometimes nearer two or three characters per token, so this estimate can be low by half on that kind of input. Use it to reason about budgets, never to predict a bill.
```

> **Watch out:** Because tokens is a property you write it.tokens with no brackets, and writing it.tokens() gives TypeError: 'int' object is not callable, which is confusing until you remember the @property line.

### 4. Write the packer

This is the decision logic, the part that actually chooses what survives. It walks the items in a fixed order, pinned first and then by priority, and asks four questions of each one before letting it in. A source cap limits each layer to a fixed share of the budget, which stops one greedy layer swallowing everything, and retrieval is the usual offender because it will happily hand you fifty passages. A duplicate check stops the same text going in twice, which is where the module's clash and confusion failures come from, since two copies of a fact look like two independent confirmations to the model. Be clear about how crude this duplicate check is: it compares the first 80 characters after lowercasing and squashing whitespace, so it catches near identical copies and nothing else, while real deduplication compares meaning using the embeddings from the retrieval module. Every rejection is recorded with a reason code and the five numbers behind it, and that record is exactly what the mini-project later turns into plain English. Nothing prints yet, so run nothing at this point, the payoff arrives in step 5.

```python
CAPS = {"docs": 0.40, "history": 0.25, "memory": 0.10}  # share of budget allowed


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())[:80]


def pack(items, budget):
    order = sorted(items, key=lambda i: (not i.pinned, i.priority))
    kept, dropped, seen = [], [], set()
    used = 0
    per_source = {}

    for it in order:
        cap = int(budget * CAPS[it.source]) if it.source in CAPS else budget
        so_far = per_source.get(it.source, 0)

        if it.pinned:
            why = None
        elif _key(it.text) in seen:
            why = "duplicate"
        elif it.tokens > cap:
            why = "bigger_than_source_cap"
        elif so_far + it.tokens > cap:
            why = "source_cap_reached"
        elif used + it.tokens > budget:
            why = "no_room_left"
        else:
            why = None

        if why:
            dropped.append({"item": it, "reason": why, "budget": budget,
                            "used": used, "cap": cap, "source_used": so_far,
                            "needed": it.tokens})
        else:
            kept.append(it)
            used += it.tokens
            per_source[it.source] = so_far + it.tokens
            seen.add(_key(it.text))

    return kept, dropped, used
```

- `CAPS = {"docs": 0.40, "history": 0.25, "memory": 0.10}`: Each number is a share of the whole budget, not a token count, so the caps rescale automatically whenever you change the budget. Layers missing from this dictionary get no cap at all, and that choice has real consequences at small budgets.
- `re.sub(r"\s+", " ", text.strip().lower())[:80]`: Collapses every run of whitespace into a single space, lowercases everything, and keeps the first 80 characters. Two texts that begin the same way therefore produce the same key, which is how the near copy of the memory note gets caught.
- `sorted(items, key=lambda i: (not i.pinned, i.priority))`: Sorting on a tuple sorts on the first value and breaks ties with the second. `not i.pinned` is False for pinned items and False counts as 0 in Python, so pinned items come first, and because Python's sort is stable, items sharing a priority keep the order build_items gave them.
- `the if/elif chain`: The checks run in a fixed order and the first match wins, so an item that is both a duplicate and too big is reported only as a duplicate. Pinned is checked first, and that single line is what makes pinning absolute rather than merely strong.
- `the dropped.append record`: It stores five numbers alongside the reason code: the budget, how much was already used, the cap for that layer, how much that layer had already spent, and how much the item needed. The mini-project turns exactly these numbers into a readable sentence, so capturing them now saves you rewriting the packer later.
- `the three running totals in the else branch`: used, per_source and seen are updated only when an item is genuinely kept. If you update them before the checks run, the packer starts counting tokens it never packed and the report stops describing the prompt you actually sent.

**The maths, spelled out**

```
Formula one, the cap for a layer: cap = int(budget * CAPS[source])

What the symbols mean: budget is the total tokens allowed for the whole prompt. CAPS[source] is that layer's allowed share written as a decimal fraction, so 0.40 means 40 percent. int() throws away the decimal part, so it always rounds down.

Worked example at budget 700: docs cap = int(0.40 * 700) = int(280.0) = 280 tokens. history cap = int(0.25 * 700) = 175. memory cap = int(0.10 * 700) = 70. Any layer not in CAPS gets cap = budget, so it is effectively uncapped.

Formula two, the running check: keep the item only if (source_used_so_far + item_tokens) is less than or equal to cap, and (used + item_tokens) is less than or equal to budget.

Worked example, the four documents at budget 700 with a docs cap of 280:
doc 1 needs 113. 0 + 113 = 113, under 280, so it is kept and so_far becomes 113.
doc 2 needs 97. 113 + 97 = 210, under 280, so it is kept and so_far becomes 210.
doc 3 needs 89. 210 + 89 = 299, which is over 280, so it is dropped with reason source_cap_reached. This is the passage holding the answer.
doc 4 needs 63. 210 + 63 = 273, under 280, so it is kept.

What it means: the cap is tested one item at a time in list order, so a smaller lower ranked passage can slip in after a bigger higher ranked one has already been refused. That is greedy first fit packing, and it is why your best passage is never automatically safe.

One more number worth seeing: the caps add up to 0.40 + 0.25 + 0.10 = 0.75, which leaves 25 percent of the budget for system, question and tools, none of which have a cap. At budget 400 those three need 50 + 30 + 181 = 261 tokens, and 100 * 261 / 400 = 65 percent. They overrun the quarter that was left for them, which is why history and memory starve at small budgets before their own caps ever matter.
```

> **Watch out:** Only docs, history and memory appear in CAPS, so tools are uncapped and quietly take 181 tokens, which is 45 percent of a 400 token budget, and if your tight runs look starved you should read the tools line before blaming the caps.

### 5. Assemble the prompt and print the report

Packing decided what goes in. Assembling decides what order it appears in, and those are two separate jobs that people often mash together. You put the layers that rarely change first (the system prompt and the tool descriptions) and the layers that change on every call last, because prompt caching matches on the prefix, and one changed character throws away the discount for everything below it. History is flipped back into reading order, oldest first, because you packed it newest first to protect the recent turns, but a conversation read backwards confuses a human reader and the model equally. The report function is the real deliverable of this lab and not a debug aid, because if you cannot see what was cut you cannot tell a wrong answer from a starved one. It prints one KEPT line per layer with its token total and its share of the budget, then one DROP line per rejected item with a reason code and the first 52 characters of the text. You should see blocks like KEPT docs 273 tok 39.0% of budget followed by DROP docs 89 tok source_cap_reached.

```python
LABEL = {"system": "SYSTEM", "tools": "AVAILABLE TOOLS",
         "memory": "NOTES FROM PAST SESSIONS", "docs": "RETRIEVED DOCUMENTS",
         "history": "EARLIER CONVERSATION", "question": "USER QUESTION"}
ORDER = ["system", "tools", "memory", "docs", "history", "question"]  # stable to volatile


def build_prompt(kept):
    blocks = []
    for s in ORDER:
        chunk = [i.text for i in kept if i.source == s]
        if s == "history":
            chunk = list(reversed(chunk))  # packed newest first, read oldest first
        if chunk:
            blocks.append(f"## {LABEL[s]}\n" + "\n\n".join(chunk))
    return "\n\n".join(blocks)


def report(kept, dropped, used, budget):
    print(f"\n=== BUDGET {budget} | used {used} | free {budget - used} ===")
    by_source = {}
    for i in kept:
        by_source[i.source] = by_source.get(i.source, 0) + i.tokens
    for s, t in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  KEPT {s:<9} {t:>5} tok  {100 * t / budget:5.1f}% of budget")
    for d in dropped:
        it = d["item"]
        head = " ".join(it.text.split())[:52]
        print(f"  DROP {it.source:<9} {it.tokens:>5} tok  {d['reason']:<22} {head}...")
```

- `ORDER = ["system", "tools", "memory", "docs", "history", "question"]  # stable to volatile`: The list runs from text that is identical on every call to text that changes on every call. That makes the unchanged opening stretch of your prompt as long as possible, and the length of that stretch is what prompt caching charges you less for.
- `chunk = list(reversed(chunk))  # packed newest first, read oldest first`: History was packed newest first so the recent turns were protected when the budget ran out. This line puts it back into oldest first order for the prompt only, so the conversation reads forwards the way a conversation should.
- `blocks.append(f"## {LABEL[s]}\n" + "\n\n".join(chunk))`: Adds a plain heading above each layer so the model can tell a retrieved document from an old chat turn from a saved note. Without these headings the whole prompt is one flat wall of text with nothing marking where anything came from.
- `by_source[i.source] = by_source.get(i.source, 0) + i.tokens`: Adds up the kept tokens for each layer. Using .get with a default of 0 means the first item from a layer starts the total at zero instead of raising KeyError, which is shorter than testing whether the key exists.
- `print(f"  KEPT {s:<9} {t:>5} tok  {100 * t / budget:5.1f}% of budget")`: The format codes line the columns up: <9 pads the name to nine characters on the left, >5 right aligns the number in five characters, and 5.1f shows one decimal place in five characters. Aligned columns are what let you scan three reports and spot the difference instantly.

**The maths, spelled out**

```
Formula one, the share printed on each KEPT line: percent = 100 * layer_tokens / budget

What the symbols mean: layer_tokens is the number of tokens kept from that layer, and budget is the total allowed for the whole prompt.

Worked example: at budget 700 the packer kept 273 tokens of documents. 100 * 273 / 700 = 39.0, which prints as "39.0% of budget".

Formula two, why ORDER runs from stable to volatile. Prompt caching bills a repeated opening stretch at roughly one tenth of the normal input price, so:
cost_in_token_equivalents = cached_tokens / 10 + fresh_tokens

What the symbols mean: cached_tokens is the length of the opening stretch that is character for character identical to your previous call. fresh_tokens is everything from the first differing character onwards.

Worked example: you send 200,000 input tokens and the first 180,000 are unchanged from last time. Cost = 180,000 / 10 + 20,000 = 18,000 + 20,000 = 38,000 token equivalents instead of 200,000, which is about 81 percent off. Now move a timestamp to the very top of the prompt. The first character now differs, cached_tokens becomes 0, and you pay the full 200,000.

What it means: caching matches from the start of the prompt forward, so the earlier in the prompt something changes, the more of your discount it destroys.

Being honest about the simplification: the one tenth figure is a round number, and the real discount, the minimum prefix length and how long the cache lives all vary by provider. Check your provider's pricing page before you build a cost model on it.
```

> **Watch out:** The system text is inside the assembled prompt because "system" is listed in ORDER, so if you later switch to chat(prompt, system=SYSTEM) you must remove it from ORDER or the model receives the same instructions twice.

### 6. Run it at three budgets and watch the answer break

Now you add the entry point and run python packer.py from inside the lab folder. You get three passes over the same question at budgets of 1200, 700 and 400 tokens, with the full report and the model's answer printed after each one. At 1200 tokens everything fits except the duplicated memory note, the refund window document survives, and the model should answer with the real numbers, 30 days and an approval code for anything over 500 USD. At 700 the docs cap is 280 tokens, the first two passages use 210 of it, and the third one, the one holding the answer, needs 89 more and does not fit, so it is dropped with reason source_cap_reached and the model should switch to the exact refusal line from your system prompt. That is the whole lesson: the model did not get worse between those two runs, your packing did. At 400 the report grows to twelve DROP lines and almost nothing survives except the tools, which is a warning sign in itself. If the break does not land where you expect, change a budget or a cap and watch the report move.

```python
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
    from llm import chat, whoami

    print("provider:", whoami())
    items = build_items()
    print("material available:", sum(i.tokens for i in items), "tokens")

    for budget in (1200, 700, 400):
        kept, dropped, used = pack(items, budget)
        report(kept, dropped, used, budget)
        answer = chat(build_prompt(kept))          # if your llm.py takes a system
        print("  ANSWER:", " ".join(answer.split())[:300])   # prompt separately, pass it there
```

- `sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))`: Builds the path to my-work/labs/_shared from the location of this file rather than from wherever you happen to be standing. That means python packer.py works from any directory, unlike the '../_shared' shortcut you used in step 1.
- `print("material available:", sum(i.tokens for i in items), "tokens")`: Prints the total size of everything before any packing happens, which should read 845. Comparing that one number against each budget tells you straight away whether a run is a comfortable fit or a squeeze.
- `for budget in (1200, 700, 400):`: Runs the same question three times with only the budget changed. Holding the question, the sources and the caps fixed is what lets you attribute the change in the answer to packing and to nothing else.
- `answer = chat(build_prompt(kept))`: Sends the whole assembled block as one user message. The system text is already inside that string, so if you prefer to pass it separately use chat(prompt, system=SYSTEM) and remove "system" from ORDER so it is not sent twice.
- `" ".join(answer.split())[:300]`: split() followed by join() squashes newlines and repeated spaces into single spaces, and [:300] trims the result. It keeps each answer on one readable line so the three runs sit side by side and are easy to compare.

**The maths, spelled out**

```
Formula: free = budget - used, where used is the sum of the tokens of every item the packer kept.

Worked example, budget 1200. The total material is 845 tokens, comfortably under 1200, so almost everything fits. The only rejection is the duplicated memory note at 31 tokens, so used = 845 - 31 = 814 and free = 1200 - 814 = 386. The document with the answer survives.

Worked example, budget 700. The docs cap is int(0.40 * 700) = 280. Documents one and two use 113 + 97 = 210, and the third needs 89, but 210 + 89 = 299 is over 280, so it is dropped. The memory cap is int(0.10 * 700) = 70, and after two kept notes totalling 49 the last note needs 28, so 49 + 28 = 77 is over 70 and it goes too. used = 697 and free = 3.

The detail that matters most: at the moment the answer document was refused, only 471 of the 700 tokens had been used (50 for system, 30 for question, 181 for tools, 113 and 97 for the first two documents). There were 229 tokens free and the item needed 89. The overall budget had plenty of room. The 280 token docs cap is what refused it.

Worked example, budget 400. The docs cap is int(0.40 * 400) = 160, and only the first document fits, because 113 + 97 = 210 is already over 160. Twelve items are dropped in total, used = 396 and free = 4.

What it means: the answer breaks somewhere between 1200 and 700 tokens, and it breaks because of a percentage cap you chose, not because the window filled up.
```

> **Watch out:** At the tighter budgets a model may still confidently say "30 days" from its own training data rather than giving your refusal line, which looks like the lab failing but is really the model ignoring your system prompt, so check the DROP lines first to confirm the document really was cut and then treat the invented answer as the finding it is.

## You are done when

You are done when you can run python packer.py and get three reports in a row. At budget 1200 you see 845 tokens of material available, 814 used, and only the duplicated memory note dropped, with the model answering 30 days and an approval code over 500 USD. At budget 700 you see a line reading DROP docs 89 tok source_cap_reached against the "Refund window and approval" passage, and the model switches to the exact refusal sentence from your system prompt. At budget 400 you see twelve DROP lines and the answer is still a refusal. Every dropped item carries a reason code you can name out loud and a number you can point at.

---

## Mini-project: The honest context packer

Your report prints codes like source_cap_reached, which means something to you today and nothing to anyone else in three weeks. Rewrite every dropped item as two plain sentences that name the deciding number, and write them to drops.json so a program can check your work.

- In my-work/labs/lab08 create explain.py. It imports build_items and pack from packer.py and calls pack() directly, so it makes no model call and needs no API key.
- Write why_sentence(rec): one sentence of ordinary English that names at least two of the record's own numbers (budget, cap, source_used, used, needed) and never prints the raw reason code. For the budget 700 docs drop it should read like "Retrieved documents may use at most 280 of the 700 token budget. 210 were already spent and this document needed 89 more."
- Write fix_sentence(rec): one sentence naming the number that would have kept the item, such as the smallest budget that fits it or the cap percentage that fits it.
- Loop over budgets 1200, 700 and 400. For each, pack the items and build one run object {"budget": 700, "used": 697, "drops": [...]}. Each drop is {"source", "reason", "budget", "needed", "cap", "source_used", "used", "text_head", "why", "fix"}: the first seven come straight from the packer's drop record, text_head is the first 60 characters of the item text with whitespace squashed, and why and fix are your two sentences.
- Write {"budgets": [run1200, run700, run400]} to drops.json with json.dump(data, f, indent=2). Print the same sentences to the terminal too, grouped by source under a WHY THINGS WERE DROPPED heading.
- Save check.py next to explain.py and run python check.py from inside my-work/labs/lab08.

### Check it

`check.py` is in this folder. Run it:

```bash
python check.py     (run it inside my-work/labs/lab08, after explain.py has written drops.json)
```


**You are done when** python check.py prints 16 PASS lines, then ALL CHECKS PASSED, and exits 0. It checks the drop counts (1, 3 and 12), the tokens used (814, 697 and 396), that every why sentence names at least two of its own numbers and leaks no reason code, that every fix names a number, that each reason is consistent with its own arithmetic, and that the 89 token "Refund window and approval" document is dropped at budget 700 and kept at 1200. Whether your sentences read clearly to a stranger is not checked automatically, and the script says so at the end.

**If you want more:** Run the same question at ten budgets from 300 to 2000 and record which items were dropped at each. If one document is dropped at every budget except the largest, its priority in build_items is wrong, not your budget. Change the priority, rerun, and show the before and after drop lists.
