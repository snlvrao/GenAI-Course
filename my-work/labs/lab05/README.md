# Lab 05: Base model vs reasoning model

**Module 5: How an LLM is trained**

You are going to send the same two questions to two different models, a cheap fast one and a slower reasoning one, and measure exactly what happens. For every call you will record the answer, the input tokens, the output tokens, the hidden thinking tokens, the wall clock seconds and the cost in dollars. A reasoning model is one trained to write a long private working-out before it answers, and you are billed for that working even though the provider usually hides the text from you. By the end you will have four rows of real numbers that tell you when the expensive model earns its price on your own work, instead of a benchmark score that somebody else ran on somebody else's tasks.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Point your .env at two models

The course helper llm.py decides which company to call by reading LLM_PROVIDER out of your .env file, so you switch providers by editing text and never by editing Python. A .env file is a plain text file of NAME=value lines that gets loaded into your environment variables when the program starts, which keeps model choices and API keys out of your source code. Here you add two more lines naming the cheap fast model and the reasoning model, because this lab has to call both of them in the same run. Use the exact model ids listed on your provider's own model page, since providers rename and retire models often and the ids below are only what the tiers happened to be called in August 2026. Check the helper is alive by running python -c "import sys; sys.path.append('my-work/labs/_shared'); from llm import whoami; print(whoami())", and you should get one line back like provider=openai  model=gpt-5.6-luna  endpoint=https://api.openai.com/v1  key=set. If that line says key=MISSING, either the .env was not found or the key line is not in it, and every call later in this lab will fail with an authentication error.

```python
# my-work/labs/.env
LLM_PROVIDER=openai
MODEL_FAST=gpt-5.6-luna
MODEL_REASONING=gpt-5.6-sol
```

- `# my-work/labs/.env`: The leading # makes this a comment, and it is there to tell you where to save the file. The file must sit inside the labs folder and its name is literally .env, starting with a dot and with nothing after the word env.
- `LLM_PROVIDER=openai`: This one word picks which company's endpoint llm.py talks to, because the helper keeps a registry of providers and looks yours up by this name. Change this single line and every lab in the course points somewhere else, including a model running on your own laptop.
- `MODEL_FAST=gpt-5.6-luna`: The cheap, quick model you want as your baseline. Your lab script reads this name itself, so llm.py never sees it and the value can be any model id your chosen provider accepts.
- `MODEL_REASONING=gpt-5.6-sol`: The model trained to write out its working before answering. Both models have to belong to the same provider you named on the first line, because there is only one endpoint and one API key in play for the whole run.

> **Watch out:** On Windows, Notepad and File Explorer quietly save the file as .env.txt, and a file called .env.txt is never loaded, so check the real name with `dir /a` inside the labs folder.

### 2. Create the file and the two questions

Create my-work/labs/lab05/compare_reasoning.py and put this block at the top of it. It does four separate jobs: it teaches Python where to find the shared helper, it reads your two model names out of the environment, it writes down what each model costs, and it holds the two questions you are about to compare. The hard question is a small constraint puzzle with a costing sum bolted on, so it needs several dependent steps and it has exactly one right answer that you can check by hand. The easy question is ordinary rewriting work, which is what most real production traffic actually is, and it is here as the control case. The price table is list price in dollars per million tokens, checked on 5 August 2026, so replace the numbers with today's if you are reading this later. Running the file at this point prints nothing at all, because there is no main() yet, and that silence is the expected result rather than a failure.

```python
# my-work/labs/lab05/compare_reasoning.py
import os, sys, time, json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "_shared"))
from llm import chat_raw, whoami

FAST = os.getenv("MODEL_FAST", "")
REASONING = os.getenv("MODEL_REASONING", "")

# Dollars per million tokens: (input, output). Checked 5 Aug 2026.
PRICES = {
    "gpt-5.6-sol":      (5.00, 30.00),
    "gpt-5.6-terra":    (2.00, 12.00),
    "gpt-5.6-luna":     (0.20,  1.20),
    "gemini-3.1-pro":   (2.00, 12.00),
    "gemini-3.6-flash": (1.50,  7.50),
}

HARD = """Four servers (A, B, C, D) each run exactly one nightly job: backup, index, report, sync.
- A runs neither backup nor sync.
- The server running index comes immediately before the server running report in alphabetical order.
- D runs either backup or index.
- C does not run report.
Job costs per night: backup $12, index $7, report $3, sync $9.
Servers A and C are billed at double rate.
Give each server's job and the total nightly cost."""

EASY = """Rewrite this as one calm sentence for a public status page:
'db failover done 04:12 UTC, 6 min read-only, no data loss'."""
```

- `sys.path.append(str(Path(__file__).resolve().parents[1] / "_shared"))`: Path(__file__).resolve() is the full path of this script, and parents[1] steps up two folders to reach labs. Adding that folder plus _shared to sys.path is how `import llm` finds a file that lives in a sibling directory, with no pip install and no package setup.
- `from llm import chat_raw, whoami`: This import also loads your .env, because llm.py calls load_dotenv the first time it is imported. That is why this line has to sit above the os.getenv lines below it, otherwise your two model names would still be unset when you read them.
- `FAST = os.getenv("MODEL_FAST", "")`: os.getenv reads an environment variable, and the second argument is the fallback value used when the variable is missing. Defaulting to an empty string lets main() print a clear instruction later instead of sending an empty model name to the provider and getting a confusing 400 error.
- `PRICES = { "gpt-5.6-sol": (5.00, 30.00), ... }`: A plain dictionary mapping a model id to a two item tuple of (input price, output price) in dollars per million tokens. Output is always the dearer of the two, because generating each new token costs the provider a full pass through the model while reading your input can be done in bulk.
- `HARD = """Four servers (A, B, C, D) ..."""`: Triple quotes let a string run over several lines without escaping anything. The four bullet constraints narrow to exactly one arrangement, which is the point: you can grade the answer as right or wrong rather than judging whether it feels good.
- `EASY = """Rewrite this as one calm sentence ..."""`: A rewrite with no hidden steps and no single correct answer. It is the control case, and its job is to show you what you are paying for when everyday shaping work gets sent to an expensive model.

**The maths, spelled out**

```
Converting a price list into the cost of one call.

formula: price per token = price per million tokens / 1,000,000

what the symbols mean:
  price per million tokens = the number in the PRICES table, in US dollars
  1,000,000 = the block size providers quote against, chosen so the numbers are readable

worked example, using gpt-5.6-luna whose output price is 1.20 dollars per million:
  price per token = 1.20 / 1,000,000 = 0.0000012 dollars
  500 output tokens = 500 x 0.0000012 = 0.0006 dollars, which is 0.06 of a cent

now the gap between the two tiers in this table:
  output: 30.00 / 1.20 = 25, so each output token from gpt-5.6-sol costs 25 times as much
  input:  5.00 / 0.20  = 25, the same multiple on the input side

what it means: the price list alone is a 25 times gap before the reasoning model has written a single extra token. The difference in how many tokens it writes then multiplies on top of that 25, which is why a final bill can end up more than a hundred times larger for one answer.
```

> **Watch out:** If your LLM_PROVIDER is not openai, the ids in PRICES will not match the models you actually call, PRICES.get falls back to (0.0, 0.0) and every cost prints as 0.00000, so add your own model ids and their real prices to the table.

### 3. Read the token counts off the response

These three small helpers pull the token counts off the provider's reply and turn them into a dollar figure. A token is a small chunk of text, roughly three to four characters of English, and every provider bills you by counting them. You get four counts that matter here: input is what you sent, output is everything the model wrote, reasoning is the hidden working it did before answering, and cached is the part of your input the provider had already stored from an earlier call and charges less for. Providers agree on what these four numbers mean but not on what to call them, so read_usage tries both common spellings and returns 0 rather than crashing when a field is simply absent. The one thing to hold on to is that reasoning tokens are counted inside output tokens and not added on top, so if you add them again you will double count your own bill. Without these helpers your script would raise AttributeError the moment you switched provider, because the response objects genuinely do use different attribute names. Nothing prints yet at this stage.

```python
def _int(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if isinstance(v, int):
            return v
    return 0

def read_usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input": 0, "output": 0, "reasoning": 0, "cached": 0}
    out_d = getattr(u, "completion_tokens_details", None) or getattr(u, "output_tokens_details", None)
    in_d  = getattr(u, "prompt_tokens_details", None) or getattr(u, "input_tokens_details", None)
    return {
        "input":     _int(u, "prompt_tokens", "input_tokens"),
        "output":    _int(u, "completion_tokens", "output_tokens"),
        "reasoning": _int(out_d, "reasoning_tokens"),
        "cached":    _int(in_d, "cached_tokens"),
    }

def cost_usd(model, use):
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    fresh_in = max(use["input"] - use["cached"], 0)
    dollars = fresh_in * price_in + use["cached"] * price_in * 0.10 + use["output"] * price_out
    return dollars / 1_000_000
```

- `def _int(obj, *names):`: Takes any object plus however many possible attribute names you care to pass, and returns the first one that holds a whole number, otherwise 0. The star in *names collects the extra arguments into a tuple, which is what lets you write _int(u, "prompt_tokens", "input_tokens") and cover two providers with one call.
- `v = getattr(obj, n, None)`: getattr reads an attribute whose name you only have as a string, and the third argument is what to hand back when that attribute does not exist. That third argument is the whole reason this never raises AttributeError on a provider that does not report the field.
- `out_d = getattr(u, "completion_tokens_details", None) or getattr(u, "output_tokens_details", None)`: The reasoning and cached counts are not on the usage object directly, they sit one level down inside a details object. The `or` picks whichever of the two spellings your provider used, because a missing attribute comes back as None and None is falsy in Python.
- `"reasoning": _int(out_d, "reasoning_tokens"),`: This is the number the whole lab exists to show you, the hidden thinking you paid for and never read. If your provider does not expose it you will see 0, which is not proof that no thinking happened, only proof that you were not told about it.
- `fresh_in = max(use["input"] - use["cached"], 0)`: Cached tokens are already counted inside the input total, so you subtract them out before charging full price or you would be billing the same tokens twice. The max(..., 0) guards against a provider reporting a cached count larger than input, which would otherwise produce a negative bill.
- `return dollars / 1_000_000`: The prices are quoted per million tokens, so everything above this line is measured in millionths of a dollar and this single division brings it back to real money. The underscores in 1_000_000 are ignored by Python and exist only so your eye can see it is a million and not ten million.

**The maths, spelled out**

```
The full cost formula, which is the only equation in this lab.

formula:
  fresh_in = max(input - cached, 0)
  dollars  = (fresh_in x price_in + cached x price_in x 0.10 + output x price_out) / 1,000,000

what every symbol means:
  input    = total input tokens, everything you sent including the system text
  cached   = the part of input the provider had already seen and stored
  fresh_in = the part of input it had to read for the first time, so full price
  output   = every token the model produced, and this already includes the hidden thinking
  price_in = dollars per million input tokens, from the PRICES table
  price_out= dollars per million output tokens, from the PRICES table
  0.10     = the cache discount, cached input costs roughly one tenth of fresh input

worked example, the hard question on gpt-5.6-sol at 5.00 in and 30.00 out:
  input = 180, cached = 0, output = 1400, of which reasoning = 1100
  fresh_in    = 180 - 0 = 180
  input part  = 180 x 5.00  = 900
  cached part = 0 x 5.00 x 0.10 = 0
  output part = 1400 x 30.00 = 42000
  total       = 900 + 0 + 42000 = 42900
  dollars     = 42900 / 1,000,000 = 0.0429, about 4.3 cents

the same question on gpt-5.6-luna at 0.20 in and 1.20 out:
  input = 180, cached = 0, output = 220, reasoning = 0
  180 x 0.20 = 36, and 220 x 1.20 = 264, so 36 + 264 = 300
  dollars = 300 / 1,000,000 = 0.0003, about 0.03 of a cent

and what the cache is worth, if 128 of those 180 input tokens had been cached:
  fresh_in = 180 - 128 = 52
  52 x 5.00 = 260, plus 128 x 5.00 x 0.10 = 64, so the input part is 324 instead of 900

what it means: on a reasoning model almost your entire bill is the output line, and almost all of that output is text the provider never showed you. Note carefully that the 1400 output tokens already contain the 1100 thinking tokens, so you must not add them again.
```

> **Watch out:** If every reasoning number prints as 0, the likely cause is that your provider does not report the field rather than that the model did no thinking, so sanity check it by comparing the output token count against how long the visible answer actually is.

### 4. Ask one model and time it

This is the one function in the file that actually talks to a model. It starts a stopwatch, sends the question to one named model, stops the stopwatch, pulls out the answer text and the four token counts, then hands it all back as a single flat dictionary. It uses chat_raw() rather than chat() because chat_raw returns the whole response object from the provider while chat returns only a string, and the usage numbers live on the object. Chat APIs take their input as a message list, which is a list of dictionaries each holding a role ("user", "assistant" or "system") and the content, which is why the question is wrapped rather than passed as a bare string. The module page shows this call as chat_raw(question, model=model), but the shared my-work/labs/_shared/llm.py expects a message list as its first argument, so the wrapping is corrected here; if your copy of llm.py differs again, this is the only line in the whole file you need to change. Keeping the wall clock timing is worth the two extra lines, because slowness is often what kills a reasoning model inside a real product even when the price is perfectly acceptable. Still nothing on screen, since nothing calls ask() yet.

```python
def ask(model, question):
    started = time.perf_counter()
    resp = chat_raw([{"role": "user", "content": question}], model=model)
    seconds = time.perf_counter() - started
    answer = (resp.choices[0].message.content or "").strip()
    use = read_usage(resp)
    return {"model": model, "seconds": round(seconds, 1),
            "answer": answer, "cost": cost_usd(model, use), **use}
```

- `started = time.perf_counter()`: perf_counter is a high resolution stopwatch rather than a clock. Its raw value has no meaning as a date, only the difference between two readings does, which is exactly what you want when measuring how long a user waited.
- `resp = chat_raw([{"role": "user", "content": question}], model=model)`: The shared helper expects a list of message dictionaries, so the question is wrapped in a single user message. The model=model keyword is forwarded straight into the provider call and overrides the default model from your .env, and that override is the mechanism that lets one script hit two different models in one run.
- `answer = (resp.choices[0].message.content or "").strip()`: Chat APIs can return several candidate answers, so choices[0] takes the first one. The `or ""` is not decoration: a reasoning model that spends its entire output budget on hidden thinking returns content of None, and calling .strip() on None crashes the script.
- `use = read_usage(resp)`: Pulls the four token counts off the same response object while you still have it. It has to happen here, because the response is thrown away the moment this function returns its dictionary.
- `**use`: The two stars unpack the four keys of use into the same flat dictionary as model, seconds, answer and cost. Flat is what the printing loop in the next step needs, because it reads r['input'] directly rather than r['use']['input'].
- `round(seconds, 1)`: One decimal place is the honest amount of precision here. Network jitter between two calls is worth far more than a hundredth of a second, so printing extra digits would imply an accuracy the measurement does not have.

**The maths, spelled out**

```
Turning the seconds column into something you can reason about.

formula: tokens per second = output tokens / seconds

what the symbols mean:
  output tokens = every token the model wrote, hidden thinking included
  seconds       = wall clock time from just before the call to just after it, so it includes network time

worked example with the same two calls as before:
  reasoning model: 1400 tokens in 22.0 seconds  = 1400 / 22.0 = 63.6 tokens per second
  fast model:       220 tokens in  1.8 seconds  =  220 /  1.8 = 122.2 tokens per second

now split the wait into its two causes:
  total slowdown  = 22.0 / 1.8   = 12.2 times slower end to end
  per token speed = 122.2 / 63.6 = 1.9 times slower per token
  quantity        = 1400 / 220   = 6.4 times more tokens written
  and 1.9 x 6.4 = 12.2, which matches the total

what it means: only about a factor of 2 of the extra wait comes from the reasoning model being a slower machine. The other factor of 6 is simply that it chose to write six times more text, nearly all of it hidden working you never see.
```

> **Watch out:** If you get a TypeError mentioning messages, your copy of llm.py has a different chat_raw signature, so match the wrapping to whatever it expects; this is the only line in the file that touches the provider.

### 5. Run both questions through both models

main() runs the two questions through the two models, four calls in total, prints each answer, prints one table and saves everything to results.json. The guard at the top stops you with a readable one line message if MODEL_FAST or MODEL_REASONING is missing, instead of making you wait for a confusing error from the provider. whoami() prints the provider, model, endpoint and whether a key was found, which is the fastest way to spot a .env that did not load. The table exists so that all four rows sit in front of you at once, because a comparison you have to scroll between is a comparison you will read wrong. results.json is the file you reuse in the mini-project, so you do not pay for the same four calls twice. Grade the hard answers yourself before you look at the cost column: the one correct solution is A index, B report, C sync, D backup, total $47 per night. Expect roughly 30 to 90 seconds of waiting, nearly all of it on the two reasoning calls, then four answer blocks followed by the table and the line wrote results.json.

```python
def main():
    if not FAST or not REASONING:
        sys.exit("Set MODEL_FAST and MODEL_REASONING in your .env first.")
    print(whoami())

    rows = []
    for label, question in (("hard", HARD), ("easy", EASY)):
        for model in (FAST, REASONING):
            r = ask(model, question)
            r["question"] = label
            rows.append(r)
            print(f"\n=== {label} / {model} ===")
            print(r["answer"][:600])

    print(f"\n{'q':6}{'model':24}{'in':>7}{'out':>7}{'think':>7}{'sec':>7}{'usd':>10}")
    for r in rows:
        print(f"{r['question']:6}{r['model']:24}{r['input']:7d}{r['output']:7d}"
              f"{r['reasoning']:7d}{r['seconds']:7.1f}{r['cost']:10.5f}")

    Path("results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote results.json")

if __name__ == "__main__":
    main()
```

- `if not FAST or not REASONING:`: An empty string is falsy in Python, so this single condition catches both a variable that was never set and one set to nothing at all. sys.exit with a string prints that string and stops the program with a non zero exit code, which is the polite way to fail.
- `for label, question in (("hard", HARD), ("easy", EASY)):`: The outer loop walks the two questions and the inner loop walks the two models, which gives you exactly four calls. Writing it as nested loops rather than four separate calls means adding a third model later is a one word change.
- `print(r["answer"][:600])`: Slices the answer down to its first 600 characters so a long reasoning reply does not push the table off the top of your terminal. Nothing is lost, because the full text is still saved in results.json.
- `{'in':>7}{'out':>7}{'think':>7}`: Inside an f-string the number after the colon is the column width and > means right align. Right aligning the numeric columns while leaving the text columns left aligned is what makes the digits line up so you can compare straight down a column.
- `{r['cost']:10.5f}`: Ten characters wide, five decimal places, fixed point notation. Five decimals is deliberate, because a single cheap call can cost less than a hundredth of a cent and Python's default float printing would show that in scientific notation like 3e-04.
- `Path("results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")`: json.dumps turns the list of dictionaries into text and indent=2 makes that text readable by a human. The encoding="utf-8" is not optional on Windows, where the default encoding will throw UnicodeEncodeError on any accented letter or curly quote the model happened to write.

**The maths, spelled out**

```
The cost multiple, which is the single number to carry out of this lab.

formula: cost multiple = cost of the reasoning model / cost of the fast model, on the same question

what the symbols mean:
  both costs come straight out of cost_usd() for the same question text
  the multiple is a plain ratio with no units, so 143 means one hundred and forty three times more expensive

worked example, using the two hard question calls from step 3:
  reasoning model = 0.0429 dollars
  fast model      = 0.0003 dollars
  cost multiple   = 0.0429 / 0.0003 = 143

where that 143 comes from, since it is not one effect but two multiplied together:
  price per output token is 25 times higher: 30.00 / 1.20 = 25
  it wrote about 6.4 times more output:      1400 / 220   = 6.4
  25 x 6.4 = 160, and you land near 143 once the much cheaper input tokens are folded back in

what this looks like at real volume, for a task you run 10,000 times a month:
  fast model:      10,000 x 0.0003 = 3.00 dollars
  reasoning model: 10,000 x 0.0429 = 429.00 dollars

what it means: the price list gap and the token count gap multiply, they do not add. That is why a tier that is only 25 times dearer on paper can produce a bill that is well over a hundred times larger.
```

> **Watch out:** Run the script from inside my-work/labs/lab05, because both the .env lookup and the results.json write follow the folder you ran the command from and not the folder the script lives in.

### 6. Read what actually happened

Look at the think column first. On the hard question the reasoning model probably spent hundreds or thousands of tokens you never saw, and that hidden text is most of what you were charged for. Now check the easy question: if the reasoning model still burned thinking tokens to produce one calm sentence, you paid for deliberation on a task that needed none, and that is the most common way money quietly leaks out of a production system. If the fast model also got the puzzle right, that is a real finding and not a failed lab, because it means this particular question was not hard enough to separate the two models. Finding a question that does separate them is exactly what turns your mini-project into evidence rather than a guess. Be honest with yourself about how much one run proves: four calls is an anecdote, which is why the mini-project makes you repeat the interesting ones three times each. What you should be looking at is the reasoning row on the hard question holding the largest think, sec and usd numbers of all four rows.

**The maths, spelled out**

```
Two numbers worth working out by hand before you decide anything.

FIRST, how much of your output bill was invisible.

formula: thinking share = reasoning tokens / output tokens

what the symbols mean:
  reasoning tokens = the hidden working, the think column in your table
  output tokens    = everything the model produced, which already contains the thinking

worked example: 1100 / 1400 = 0.786, which is 78.6 percent
so roughly four out of every five tokens you paid for on the output side were text you never got to read.

SECOND, the break even error rate, which is the honest way to decide which model to buy.

formula: use the fast model when  cost_fast + (p x W) < cost_reasoning
         rearranged, break even p = (cost_reasoning - cost_fast) / W

what every symbol means:
  cost_fast      = dollars for one call to the cheap model
  cost_reasoning = dollars for one call to the expensive model
  p              = the fraction of the time the cheap model gets the answer wrong
  W              = what one wrong answer costs you to catch and put right, in dollars

worked example with the numbers from this lab:
  cost_reasoning - cost_fast = 0.0429 - 0.0003 = 0.0426 dollars
  say a wrong answer takes a person 5 minutes to catch and redo, and that person costs 60 dollars an hour
  W = 60 x (5 / 60) = 5.00 dollars
  break even p = 0.0426 / 5.00 = 0.0085, which is 0.85 percent, roughly 1 in 120

so if the cheap model gets this kind of question wrong more often than about 1 in 120, the reasoning model is the cheaper choice overall, even though it costs 143 times more per call.

be clear that this is a simplification: it assumes you always notice the error, that the reasoning model is never wrong, and that every wrong answer costs the same to fix. Real systems break all three of those assumptions, so treat the number as a starting point for the argument rather than the end of it.

what it means: the per call price gap looks enormous as a multiple but is tiny in absolute dollars, so what usually decides the answer is the cost of being wrong, not the cost of the call.
```

> **Watch out:** The biggest mistake at this step is grading the hard answers after you have seen the cost column, because knowing which call was expensive will quietly make you more forgiving towards it.

## You are done when

You can print a four row table (two questions by two models) showing input tokens, output tokens, thinking tokens, seconds and dollars; results.json exists on disk with the same four rows in it; you have marked each hard answer right or wrong against the one correct solution (A index, B report, C sync, D backup, $47 total per night); and you can say in one sentence which model you would pay for on each question and why.

---

## Mini-project: Where reasoning pays

Find one question where the reasoning model is repeatably better and one where it costs many times more for the same quality. You record every run in my-work/labs/lab05/verdict.json, and check.py verifies the counts, the arithmetic and your pass/fail discipline.

- Write five candidate questions before you run anything, and next to each write the expected answer or the property a good answer must have. Deciding what counts as correct after you have seen the output is how people fool themselves.
- Run all five through both models with your lab script and grade every answer pass or fail against what you wrote down. Do not score 1 to 5: you will drift and cluster in the middle, and the checker rejects anything that is not true or false.
- Pick a winner (reasoning right, fast model wrong) and a waste case (both right, reasoning much dearer), then run those two three times each on each model. That is 12 calls. Keep the token counts from every one of them.
- Save the numbers as my-work/labs/lab05/verdict.json. Top-level keys: fast_model and reasoning_model (the two ids), prices (a map from model id to [input_price, output_price] per million tokens), cases (a list of two objects) and rule (a string). Each case has role ("winner" or "waste"), question, expected, cost_multiple (mean reasoning cost divided by mean fast cost) and runs, a list of 6 objects each holding model, pass (true or false), input, output, reasoning, cached, seconds and cost. Copy the token counts straight out of results.json; the checker recomputes every cost from your own price table and rejects mismatches.
- Write the rule field: at least 25 words saying when you would spend the money, in terms a teammate could apply to a question you have never seen. Name what the winning question has that the waste case does not, such as several dependent steps, or one wrong step ruining the answer.
- Save check.py in my-work/labs/lab05 next to verdict.json and run it. Fix whatever it reports and run it again. It makes no API calls, so rerunning is free.

### Check it

`check.py` is in this folder. Run it:

```bash
python my-work/labs/lab05/check.py
```


**You are done when** python my-work/labs/lab05/check.py prints 21 lines, one per check, and ends with ALL CHECKS PASSED and exit code 0. It fails you if the winner case is not 3/3 reasoning against 0/3 fast, if any recorded cost disagrees with your own price table, if you graded on a scale instead of true/false, if a case has fewer than 3 runs per model, if reasoning tokens exceed output tokens, or if cost_multiple does not match the runs. It also prints one NOTE saying that the quality of your written rule is not checked automatically, because it cannot be.

**If you want more:** Send your winning question to the fast model three more times with "work through this step by step, then give your final answer" appended, and record those runs in a separate my-work/labs/lab05/prompted.json (the checker expects exactly two cases in verdict.json, so keep them out of it). Compare pass rate and cost against both models. Sometimes prompted step-by-step working closes the gap for a fraction of the price, and sometimes it stays worse than a model actually trained to do it.
