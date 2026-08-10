# Lab 06: Build a cost model

**Module 6: Picking a model, and what it costs**

You will run one realistic task, read the exact token counts back out of the provider's response, and turn those counts into a dollar figure for five different models. Then you will move a single line in the prompt and watch the cached-token count change, which is the cheapest way to see why the order of your prompt costs you money. Everything runs through the shared helper at my-work/labs/_shared/llm.py, so you need no new key and no new account, and you switch model provider by editing LLM_PROVIDER in your .env file, never by editing lab code. Before you start, make sure python llm.py works, see setup.html.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Make the folder and check your provider

Create my-work/labs/lab06 next to your other labs, then run this one small file before anything else. whoami() comes from the shared helper and prints four facts on one line: which provider you are pointed at, which model name will be sent, which address the call goes to, and whether your API key was found. You should see a line ending in key=set, or key=none needed if you chose a local model that runs on your own machine. If it ends in key=MISSING, every later step in this lab will fail on its first call with an error that looks like a code bug but is really a missing line in .env. Nothing here calls the model, so this costs you nothing and takes about a second. Run it first, because a wrong provider name here would waste the next twenty minutes.

```python
# my-work/labs/lab06/check.py
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from llm import whoami

print(whoami())
```

- `sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))`: __file__ is the path of this script and .resolve() turns it into a full path from the drive root, so it works no matter which folder you run the command from. parents[0] would be lab06, so parents[1] is labs, and joining _shared onto it points at the shared helper folder. Adding that folder to sys.path is what makes the next line able to find llm.py.
- `from llm import whoami`: This pulls in just one function from the shared helper instead of the whole module. It only works because the line above put the shared folder on the import path first, which is why the import sits below the sys.path line rather than at the top with the others.
- `print(whoami())`: Prints the provider, model, endpoint and key status as a single readable line. It makes no network call, so it tells you what you are about to use without spending anything or waiting for a server.

> **Watch out:** If the line ends in key=MISSING, your .env file was not found from the folder you ran the command in, because the helper starts looking for .env in your current directory and works upward.

### 2. Write the task you are going to measure

This step writes the one real task you will measure, and the shape of the prompt is the lesson rather than the words in it. A prompt cache is a store the provider keeps of the model's working-out for text it has already read, and it charges you roughly a tenth of the normal price when you send that same text again. The catch is that the reuse has to start at the very first character and run forward without a break, so the long unchanging policy block goes first and the short changing question goes last. The usage_of function then reads the real token counts out of the usage block that the provider attaches to its response, and falls back to a rough character count when there is no such block, so this file works on free tiers and on a model running on your own laptop. It hands back four things: input tokens, output tokens, cached input tokens, and a label saying whether those came from the provider or from the estimate, and that label matters more than the numbers do. Running this file prints the model's answer plus four counts, and writes them to usage.json so the next step can price them.

```python
# my-work/labs/lab06/measure.py
"""Measure what one real task costs in tokens. Writes usage.json."""
import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from llm import chat_raw, whoami  # noqa: E402

RULES = """\
- Returns are accepted within 30 days of delivery with proof of purchase.
- Faulty items carry a 24 month warranty starting on the delivery date.
- A warranty claim needs the order number and a photo of the fault.
- Refunds go back to the original payment method within 5 working days.
- Delivery charges are refunded only if the item arrived faulty.
- Never promise a replacement before the warranty check is done.
"""

CATEGORIES = ["blenders", "kettles", "toasters", "vacuum cleaners",
              "fans", "heaters", "microwaves", "coffee machines"]

# The long, unchanging part goes FIRST. That is the bit a cache can reuse.
SYSTEM = "You are the support assistant for a home appliance shop.\n\n" + "\n".join(
    f"Policy for {c}:\n{RULES}" for c in CATEGORIES
)

QUESTION = ("A customer bought a blender 40 days ago and it has stopped "
            "working. What do we do? Answer in under 60 words.")


def usage_of(resp, sent: str, got: str):
    """Real numbers if the provider reports them, a rough count if it does not."""
    u = getattr(resp, "usage", None)
    if u and getattr(u, "prompt_tokens", None):
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return u.prompt_tokens, u.completion_tokens, cached, "reported"
    # Four characters per token is a rough English rule, usually within ~20%.
    return len(sent) // 4, len(got) // 4, 0, "estimated"


if __name__ == "__main__":
    print(whoami())
    resp = chat_raw(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": QUESTION}],
        temperature=0, max_tokens=200,
    )
    answer = (resp.choices[0].message.content or "").strip()
    i, o, cached, how = usage_of(resp, SYSTEM + QUESTION, answer)

    pathlib.Path("usage.json").write_text(json.dumps(
        {"model": getattr(resp, "model", "unknown"), "counts": how,
         "input_tokens": i, "output_tokens": o, "cached_input_tokens": cached},
        indent=2), encoding="utf-8")

    print("\nanswer:", answer)
    print(f"\ninput tokens : {i}  ({how})")
    print(f"output tokens: {o}")
    print(f"cached input : {cached}")
    print(f"ratio        : {i / max(o, 1):.1f} input tokens per output token")
```

- `SYSTEM = "You are the support assistant for a home appliance shop.\n\n" + "\n".join(f"Policy for {c}:\n{RULES}" for c in CATEGORIES)`: This glues one copy of the same rules onto the prompt for each of the eight categories, which is a realistic way that real system prompts get long. It is written as a loop so you get roughly 900 tokens of stable prefix without typing 3,500 characters by hand. That length is deliberate, because a prompt too short to cache would make step 6 pointless.
- `u = getattr(resp, "usage", None)`: getattr with a default reads an attribute that may not be there and hands back None instead of crashing. Some OpenAI-compatible servers, especially local ones, return no usage block at all, and a plain resp.usage would raise AttributeError and end the lab.
- `cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0`: The cached count lives one level down, inside prompt_tokens_details, so this reads it only if that object exists. The or 0 converts a reported None into a real zero, so the division in the last print line cannot fail later.
- `return len(sent) // 4, len(got) // 4, 0, "estimated"`: This is the fallback when the provider reports nothing, using the rough rule of four characters per token. // is integer division, so you get a whole number of tokens rather than a decimal, and the label estimated travels with the numbers so you can never mistake a guess for a measurement.
- `temperature=0, max_tokens=200`: Temperature controls how much randomness the model uses when picking each next word, and 0 makes the answer as repeatable as it can be, so two runs are comparable. max_tokens caps how much the model may write, which stops one rambling answer from distorting your output count and your bill.
- `if __name__ == "__main__":`: Everything that actually calls the model sits behind this guard, so the call only happens when you run the file directly. That is what lets step 6 import SYSTEM, QUESTION and usage_of from this file without firing an extra paid request.

**The maths, spelled out**

```
The fallback estimate uses one formula:

  estimated_tokens = number_of_characters / 4

Symbols:
  number_of_characters = every character in the text, including spaces, punctuation and newlines
  4 = the average number of characters per token for ordinary English text
  a token = a chunk of text the model reads as one unit, often a word or part of a word

Worked example, using the exact strings in this file:
  RULES is 416 characters.
  Each category block is "Policy for <name>:" plus a newline plus RULES, so about 429 characters plus the length of the name.
  Eight blocks plus the opening sentence gives SYSTEM = 3,571 characters.
  QUESTION = 108 characters.
  Total sent = 3,571 + 108 = 3,679 characters.
  3,679 / 4 = 919 estimated tokens.
  The provider in the sample run reported 912 tokens.
  Difference = 919 - 912 = 7 tokens, which is 7 / 912 = 0.008, so 0.8% off.

Intuition: the four-characters rule is close enough to sketch a budget with, and wrong enough that you should never quote it as if you had measured it.
```

> **Watch out:** pathlib.Path("usage.json") writes to the folder you are standing in, not next to the script, so run this from inside my-work/labs/lab06 or step 4 will not find the file.

### 3. Run it and look at the ratio

Run python measure.py and read the four numbers it prints. Expect input in the high hundreds and output well under a hundred, giving a ratio of roughly fifteen input tokens for every output token. That ratio is the most useful number in this lab, because it tells you which of the two price columns actually decides your bill. Output tokens cost about six times more each than input tokens, so any ratio above six means the input column is the bigger half, and yours is around fifteen. Your exact numbers will differ from the ones shown here, because different providers split text into tokens differently and the model writes a slightly different answer each time. If the label says estimated, your provider returned no usage block, so the counts came from the four-characters-per-token approximation and you should say so every time you quote them.

```python
python measure.py

input tokens : 912  (reported)
output tokens: 58
cached input : 0
ratio        : 15.7 input tokens per output token
```

- `input tokens : 912  (reported)`: This counts the whole system prompt, the question, and the small amount of formatting the provider wraps around your messages. The word reported means these came from the provider's own counter, so you can quote them as facts rather than estimates.
- `output tokens: 58`: Only the words the model wrote back are counted here. You asked for under 60 words and got 58 tokens, so words and tokens land close together in short English answers, but they are not the same thing.
- `cached input : 0`: Zero is the expected answer on a first run, because nothing identical has been sent yet, so there is nothing stored to reuse. Step 6 is where this number is supposed to move.
- `ratio        : 15.7 input tokens per output token`: This is simply input divided by output, printed to one decimal place by the :.1f format. It is the single number you carry into the pricing step, because it tells you whether input price or output price is the one to shop on.

**The maths, spelled out**

```
Two pieces of arithmetic sit behind this line.

1) The ratio
  ratio = input_tokens / output_tokens
  912 / 58 = 15.7 input tokens for every output token.

2) Which price column dominates
  input_share_of_bill = (input_tokens x price_in) / (input_tokens x price_in + output_tokens x price_out)

Symbols:
  price_in = dollars per million input tokens
  price_out = dollars per million output tokens

Worked example with GPT-5.6 Sol at $5 in and $30 out per million:
  input part  = 912 x 5  = 4,560
  output part = 58 x 30   = 1,740
  total       = 6,300
  input share = 4,560 / 6,300 = 0.724, so 72% of the bill is input.

The break-even rule:
  input dominates whenever ratio > price_out / price_in
  For Sol that is 30 / 5 = 6. Your ratio is 15.7, which is above 6, so input wins.

Intuition: each output token costs more, but you send far more input than you get back, so the side you send is usually the side you pay for.
```

> **Watch out:** If the label says estimated, your ratio is a guess divided by a guess, so do not put it into a budget without writing the word estimated next to it.

### 4. Write the price table

This step turns tokens into money, and it is short because the cost model really is one line of arithmetic. Prices are always quoted per million tokens, so every figure has to be divided by 1,000,000 at the end, and forgetting that is the most common mistake in this whole lab. PRICES holds five models with three numbers each: the input price, the output price, and the cached input price, all in dollars per million tokens. These are list prices checked on 5 August 2026, and large customers negotiate below list, so treat the whole table as a worked example rather than a live quote. There is no context window column, because only the GPT-5.6 figure (one million tokens for all three tiers) is confirmed here, and you should read each provider's own page for the rest. Be honest about one thing when you use this: only the GPT-5.6 Sol cached price is published, and the other four cached figures apply the rough one tenth rule, so they are estimates.

```python
# my-work/labs/lab06/price.py
"""Turn tokens into money."""
import json
import pathlib

# Dollars per MILLION tokens, list prices checked 5 August 2026.
# The GPT-5.6 Sol cached price is published. The other cached figures use the
# rough "cached input costs about a tenth" rule, so treat them as estimates.
PRICES = {
    "GPT-5.6 Sol":      {"in": 5.00, "out": 30.00, "cached_in": 0.50},
    "GPT-5.6 Terra":    {"in": 2.00, "out": 12.00, "cached_in": 0.20},
    "GPT-5.6 Luna":     {"in": 0.20, "out": 1.20,  "cached_in": 0.02},
    "Gemini 3.1 Pro":   {"in": 2.00, "out": 12.00, "cached_in": 0.20},
    "Gemini 3.6 Flash": {"in": 1.50, "out": 7.50,  "cached_in": 0.15},
}


def cost(in_tok, out_tok, p, hit=0.0):
    """Cost of one call in dollars. hit = share of input served from cache."""
    fresh = in_tok * (1 - hit)
    reused = in_tok * hit
    return (fresh * p["in"] + reused * p["cached_in"] + out_tok * p["out"]) / 1e6


def table(title, in_tok, out_tok, runs=1000, hit=0.0):
    print(f"\n{title}: {in_tok:,} in / {out_tok:,} out, "
          f"{runs:,} runs, cache hit {hit:.0%}")
    print(f"{'model':<18}{'per call':>12}{'per 1k runs':>14}")
    for name, p in PRICES.items():
        one = cost(in_tok, out_tok, p, hit)
        print(f"{name:<18}{'$' + format(one, '.5f'):>12}"
              f"{'$' + format(one * runs, '.2f'):>14}")


path = pathlib.Path("usage.json")
if not path.exists():
    raise SystemExit("Run measure.py first - it writes usage.json.")
u = json.loads(path.read_text(encoding="utf-8"))
i, o = u["input_tokens"], u["output_tokens"]

table("Your measured task", i, o)
table("Same task, 90% of the input cached", i, o, hit=0.90)
# An agent re-sends its whole history and its tool results on every turn.
table("The same job inside an agent loop", 60_000, o)
table("Agent loop with a stable prefix cached", 60_000, o, hit=0.90)
```

- `"GPT-5.6 Sol":      {"in": 5.00, "out": 30.00, "cached_in": 0.50},`: Three prices per model, all in dollars per million tokens. Notice the pattern that holds across every row: output is about six times input, and cached input is about a tenth of input, which is where the two rules of thumb in this module come from.
- `fresh = in_tok * (1 - hit)  /  reused = in_tok * hit`: hit is your cache hit rate written as a fraction between 0 and 1, so 0.90 means 90% of your input was already stored. These two lines split the input tokens into the part billed at full price and the part billed at the cheap cached price, which is exactly how providers bill it.
- `return (fresh * p["in"] + reused * p["cached_in"] + out_tok * p["out"]) / 1e6`: This single line is the whole cost model. The three products are the three things you are charged for, and the / 1e6 converts from per-million prices into actual dollars for this one call.
- `print(f"{'model':<18}{'per call':>12}{'per 1k runs':>14}")`: &lt;18 pads a value to 18 characters and lines it up on the left, while &gt;12 pads to 12 and lines it up on the right. Numbers only compare easily when their last digits sit in the same column, which is the only reason this formatting exists.
- `if not path.exists(): raise SystemExit("Run measure.py first - it writes usage.json.")`: SystemExit stops the program with a plain sentence instead of a stack trace. It matters because the most likely failure here is skipping step 3, and a readable message points you straight at the fix.
- `table("The same job inside an agent loop", 60_000, o)`: This keeps your own measured output count but swaps the input for 60,000 tokens, which is what one step of a modest agent carries once it re-sends its history and its tool results. The underscore in 60_000 is only a digit separator that Python ignores, there to make the number readable.

**The maths, spelled out**

```
The full cost model, in ordinary characters:

  cost_dollars = ( in_tok x (1 - hit) x price_in
                 + in_tok x hit x price_cached
                 + out_tok x price_out ) / 1,000,000

Symbols:
  in_tok      = input tokens you send on this call
  out_tok     = output tokens the model writes back
  hit         = share of input served from cache, 0 to 1 (0.90 means 90%)
  price_in    = dollars per million fresh input tokens
  price_cached= dollars per million cached input tokens
  price_out   = dollars per million output tokens
  1,000,000   = because every published price is per million tokens

Worked example, your measured 912 in and 58 out, no cache, on GPT-5.6 Sol:
  fresh input  = 912 x (1 - 0) = 912, and 912 x 5 = 4,560
  reused input = 912 x 0 = 0, and 0 x 0.50 = 0
  output       = 58 x 30 = 1,740
  sum          = 4,560 + 0 + 1,740 = 6,300
  cost         = 6,300 / 1,000,000 = $0.00630 per call
  per 1,000 runs = $0.00630 x 1,000 = $6.30

Same numbers on GPT-5.6 Luna at $0.20 in and $1.20 out:
  912 x 0.20 = 182.4, and 58 x 1.20 = 69.6, sum 252
  252 / 1,000,000 = $0.000252 per call, so $0.25 per 1,000 runs

Spread between them = 6,300 / 252 = 25.0, so exactly twenty-five to one.

Intuition: only five numbers decide your bill, two of which you measured yourself, so this is arithmetic rather than guesswork.
```

> **Watch out:** Prices are per million, so if a per call figure comes out in whole dollars instead of fractions of a cent, you dropped the / 1e6 and every number is a million times too big.

### 5. Read the four tables

Run python price.py and read the four tables in order, because each one changes exactly one thing from the one before it. The first is your real job at list price, and the gap between the dearest and cheapest row is twenty-five to one for the identical 912 tokens. The second changes nothing except assuming 90% of the input is served from cache, and the total drops by more than half. The third replaces your input count with 60,000 tokens, which is roughly what one step of a modest agent carries once it re-sends its conversation history and its tool results, and the per-thousand figure jumps from single dollars into the hundreds. The fourth applies the cache to that agent, and on Sol it saves more money than downgrading Sol to Terra would have saved you. One warning before you quote any of it: this table prices five named models no matter which model you actually called, so the top row is not a receipt for what you just spent.

```python
Your measured task: 912 in / 58 out, 1,000 runs, cache hit 0%
model                 per call   per 1k runs
GPT-5.6 Sol           $0.00630         $6.30
GPT-5.6 Luna          $0.00025         $0.25

The same job inside an agent loop: 60,000 in / 58 out, 1,000 runs, cache hit 0%
GPT-5.6 Sol           $0.30174       $301.74
GPT-5.6 Luna          $0.01207        $12.07
```

- `Your measured task: 912 in / 58 out, 1,000 runs, cache hit 0%`: The header repeats every assumption behind the rows below it. That is deliberate, because a cost table pasted into a document without its assumptions is a number nobody can check, including you in three months.
- `GPT-5.6 Sol           $0.00630         $6.30`: Two columns for the same fact: what one call costs, and what a thousand calls cost. The per call figure is too small for anyone to reason about, which is exactly why the second column is there.
- `GPT-5.6 Luna          $0.00025         $0.25`: Identical task, identical token counts, twenty-five times cheaper. This is the row that shows why measuring first is worth doing, because the choice between these two is worth real money only once you know your token counts.
- `The same job inside an agent loop: 60,000 in / 58 out`: Only the input number changed, from 912 to 60,000, and the output stayed at your measured 58. That is what makes the comparison honest, since the model is doing the same size of thinking with a much bigger pile of context in front of it.
- `GPT-5.6 Sol           $0.30174       $301.74`: The same shape of task now costs about 48 times more, purely because of input. This is the number that explains why agent builders shop on input price and cache hit rate rather than on benchmark scores.

**The maths, spelled out**

```
Compare the four tables with the same cost formula from step 4.

1) Your task on Sol, no cache
  912 x 5 = 4,560 plus 58 x 30 = 1,740, total 6,300, so $6.30 per 1,000 runs.
  Input share = 4,560 / 6,300 = 72%.

2) Your task on Sol at 90% cache
  fresh  = 912 x 0.10 = 91.2, and 91.2 x 5 = 456
  reused = 912 x 0.90 = 820.8, and 820.8 x 0.50 = 410.4
  output = 58 x 30 = 1,740
  total  = 2,606.4, so $2.61 per 1,000 runs
  Saving = 1 - (2,606.4 / 6,300) = 0.586, so 59% off.

3) Agent loop on Sol, no cache
  60,000 x 5 = 300,000 plus 1,740 = 301,740, so $301.74 per 1,000 runs.
  Input share = 300,000 / 301,740 = 99.4%, so output is now a rounding error.
  That is 301.74 / 6.30 = 48 times your original task.

4) Agent loop on Sol at 90% cache
  fresh  = 6,000 x 5 = 30,000
  reused = 54,000 x 0.50 = 27,000
  output = 1,740
  total  = 58,740, so $58.74 per 1,000 runs, an 80.5% cut.

Now the comparison that matters. The same agent moved down to Terra with no caching:
  60,000 x 2 = 120,000 plus 58 x 12 = 696, total 120,696, so $120.70 per 1,000 runs.
  Caching on the expensive model ($58.74) beats downgrading to the mid model ($120.70).

The general rule behind that, for any model:
  fraction saved on input = hit x (1 - price_cached / price_in)
  at 90% hit and a one tenth cached price: 0.9 x (1 - 0.1) = 0.81, so 81% off the input part.

Intuition: changing how you send the prompt often saves more than changing which model reads it.
```

> **Watch out:** The model column is a fixed list of five names, so if you ran this on a free Groq key or a local model your real spend was zero and none of these rows describes what you actually paid.

### 6. Prove the prefix rule with your own money

This is the step where you test the prefix rule instead of taking it on trust. The script sends the same text four times and changes exactly one thing: whether a timestamp sits at the end of the system prompt or at the start. With the timestamp at the end, the second call should report cached tokens, because everything before the timestamp matched the previous call character for character. With it at the start, the very first characters differ on every call, so nothing after them can be reused and the cached count should stay at zero, even though the text is the same length and says the same things. The system prompt is doubled first, because most providers only cache prefixes above roughly 1,000 tokens and your original prompt sits just under that line at about 900. If every number comes back zero, do not conclude the rule is wrong, because your provider may not report cached tokens at all or may not cache at this size, so check its documentation before deciding anything.

```python
# my-work/labs/lab06/cache_test.py
"""Show that a cache only reuses a matching prefix."""
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from llm import chat_raw  # noqa: E402
from measure import QUESTION, SYSTEM, usage_of  # noqa: E402

BIG = SYSTEM * 2  # most providers only cache prefixes above roughly 1000 tokens


def run(label: str, system: str) -> None:
    resp = chat_raw([{"role": "system", "content": system},
                     {"role": "user", "content": QUESTION}],
                    temperature=0, max_tokens=120)
    got = (resp.choices[0].message.content or "").strip()
    i, _o, cached, _how = usage_of(resp, system + QUESTION, got)
    print(f"{label:<22} input={i:<7} cached={cached:<7} hit={cached / max(i, 1):.0%}")


for n in (1, 2):  # first call warms the cache, second call should reuse it
    stamp = f"Request time: {time.time()}"
    run(f"stamp at END   #{n}", BIG + "\n" + stamp)

for n in (1, 2):
    stamp = f"Request time: {time.time()}"
    run(f"stamp at START #{n}", stamp + "\n" + BIG)

print("\nSame text, same length, one difference: where the changing line sits.")
print("If every number is 0, your provider does not report cached tokens or")
print("does not cache at this size. Check its docs before concluding anything.")
```

- `BIG = SYSTEM * 2`: Multiplying a string by 2 in Python just repeats it, so this turns 3,571 characters into 7,142. That pushes the prefix from about 900 tokens to about 1,785, which clears the roughly 1,000 token minimum most providers require before they will cache anything at all.
- `from measure import QUESTION, SYSTEM, usage_of`: This reuses the exact prompt and the exact counting code from step 2, so nothing can quietly differ between the two experiments. It costs you no extra API call only because measure.py keeps its own run behind the if __name__ == "__main__": guard.
- `for n in (1, 2):`: Two calls per arrangement, on purpose. The first call is what puts the text into the cache, and the second is the only one that can possibly hit it, so a single call would always show zero and prove nothing.
- `stamp = f"Request time: {time.time()}"`: time.time() returns the seconds since 1970 as a decimal number with fractions, so this line is guaranteed to be different on every single call. That is the whole point, because it stands in for the real-world timestamps, session IDs and user names that people put at the top of system prompts without thinking.
- `BIG + "\n" + stamp  versus  stamp + "\n" + BIG`: Same characters, same total length, same meaning to the model. The only difference is which end the changing line sits at, and that difference is the entire experiment.
- `hit={cached / max(i, 1):.0%}`: Cached tokens divided by input tokens, formatted as a whole percentage by :.0%. max(i, 1) guards against dividing by zero if a provider reports an input count of 0, which would crash the script rather than print a useful line.

**The maths, spelled out**

```
The number this step prints:

  cache_hit_rate = cached_input_tokens / input_tokens

Symbols:
  cached_input_tokens = the part of your input the provider served from its stored working-out
  input_tokens        = all input tokens on that call, cached and fresh together

Size check first, so you know the test can work at all:
  SYSTEM is 3,571 characters, so BIG is 7,142 characters.
  7,142 / 4 = about 1,785 tokens, comfortably above the usual 1,024 token minimum.
  Your original 3,571 character prompt was about 893 tokens, which is below it, and that is why the doubling exists.

Worked example, using plausible reported numbers:
  input reported  = 1,820
  cached reported = 1,664
  hit rate = 1,664 / 1,820 = 0.914, so 91%

What that saves on GPT-5.6 Sol at $5 in and $0.50 cached in per million:
  no cache: 1,820 x 5 = 9,100, so $0.00910 per call
  with cache: fresh 156 x 5 = 780, plus cached 1,664 x 0.50 = 832, total 1,612, so $0.001612 per call
  saving = 1 - (1,612 / 9,100) = 0.82, so 82% off the input bill

Why 1,664 and not the exact prefix length: many providers count cached tokens in blocks of 128 and round down, so a 1,790 token prefix reports as 13 blocks, and 13 x 128 = 1,664. That block size is common but not universal, so check your provider's documentation rather than assuming it.

Intuition: watch the percentage, not the raw count, because the percentage is what multiplies your input bill.
```

> **Watch out:** If several minutes pass between the two calls, the stored prefix may have expired and both lines show zero for a reason that has nothing to do with where you put the timestamp.

### 7. Write the decision down with a date

This is the step people skip and later regret, and it takes about two minutes. Add a short README.md to my-work/labs/lab06 with four lines: the model you would pick for this task, the cost per thousand runs, the cache hit rate you assumed, and what would make you change your mind. Put the date on it, because list prices move every few months and an undated number is a trap for whoever reads it next, including you. The last line is the important one, because a named trigger and a named fallback turn a number into a decision someone else can re-check, while a vague note to revisit later helps nobody. Notice that the assumption line records what you did not do as well as what you did, in this case no caching, because the policy block is only about 900 tokens and sits under the usual cache minimum. If you cannot fill in all four lines from your own output, go back to the step that produced the missing one.

```python
# my-work/labs/lab06/README.md
Decision, 5 Aug 2026
Model:        GPT-5.6 Luna (measured 912 in / 58 out)
Cost:         $0.25 per 1,000 runs at 0% cache hit
Assumed:      no caching yet, policy block is only ~900 tokens
Would change: if answers start missing the 30 day rule, retest on Terra
```

- `Decision, 5 Aug 2026`: The date is the load-bearing part of this file. Model prices change every few months, so a cost figure without a date cannot be checked by anyone, and a reader has no way to know whether to trust it or redo it.
- `Model:        GPT-5.6 Luna (measured 912 in / 58 out)`: It names the model and the measurement the choice came from, in the same line. That means someone with new prices can redo the arithmetic in thirty seconds instead of re-running the whole lab.
- `Assumed:      no caching yet, policy block is only ~900 tokens`: This records the assumption behind the cost, and the reason for it. The same job at 90% cache is a completely different number, so a cost quoted without its cache assumption is not reproducible.
- `Would change: if answers start missing the 30 day rule, retest on Terra`: A named symptom and a named next step. It is the difference between a decision and an opinion, because it tells the next reader exactly what evidence would overturn it.

**The maths, spelled out**

```
Scale your per-call number up until it is big enough to feel:

  cost_per_year = (cost_per_1000_runs / 1,000) x runs_per_day x 365

Symbols:
  cost_per_1000_runs = the second column from your price table
  runs_per_day = how many times this task really runs in a day
  365 = days in a year

Worked example at 10,000 runs a day:
  Luna: 0.25 / 1,000 = $0.00025 per run
        $0.00025 x 10,000 = $2.50 a day
        $2.50 x 365 = $912.50 a year
  Sol:  6.30 / 1,000 = $0.00630 per run
        $0.00630 x 10,000 = $63.00 a day
        $63.00 x 365 = $22,995 a year
  Difference = 22,995 - 912.50 = about $22,000 a year, for the same 912 tokens.

Intuition: $0.0063 per call is too small to argue about, and $22,000 a year is not, so always write the decision down at the scale you will actually run it.
```

> **Watch out:** Writing the cost line without the cache assumption beside it makes the figure impossible to reproduce, because the same task at 90% cache costs a different amount on every row of the table.

## You are done when

You can state, from measured numbers rather than guesses, what one thousand runs of your task would cost on five named models. Concretely: my-work/labs/lab06/usage.json exists and holds your own input and output token counts with the reported or estimated label attached, python price.py prints four tables built from those counts, and python cache_test.py has shown you a cached-token count that differs between the stamp-at-end runs and the stamp-at-start runs. If every cached number was zero, you are still done, provided you can point at the line in your provider's documentation that explains why it does not cache or does not report caching. Your README.md has all four lines filled in, with a date on the top one.

---

## Mini-project: A router that saves money

Build a router that sends easy questions to a cheap model and hard ones to a strong one, then prove in numbers what it saved and what it cost you in quality. It produces my-work/labs/lab06/router.py and my-work/labs/lab06/router_report.json, and check.py re-runs your router against that report so the numbers cannot be typed in by hand.

- Write 20 questions from work you actually care about, numbered id 1 to 20, ten you expect to be easy and ten you expect to be hard. Label each one "easy" or "hard" by hand before you write any code. That hand label is your only ground truth, and check.py requires exactly ten of each.
- Write my-work/labs/lab06/router.py with route(question: str) -> str returning "cheap" or "strong". Use rules you can say out loud, like question length or the presence of "why", "compare", "step by step". Keep any run code behind if __name__ == "__main__":, because check.py imports this file and must not trigger an API call.
- Run each question twice: once through route() to the model it picked, once through the strong model alone. Read the in and out token counts for every call with usage_of from measure.py, so the counts are the provider's rather than your guess.
- Escalate. If a cheap answer comes back empty, hedging, or failing a check you write yourself, redo it on the strong model, set "escalated": true, and append that strong call to router_calls so the double payment lands in the total.
- Read every cheap answer and set "cheap_correct": true or false. For questions routed straight to strong, set it to null. Then write my-work/labs/lab06/router_report.json in exactly this shape: {"date": "2026-08-05", "prices_used": {"cheap": {"in": 0.20, "out": 1.20}, "strong": {"in": 5.00, "out": 30.00}}, "questions": [{"id": 1, "text": "...", "label": "easy", "routed_to": "cheap", "escalated": false, "cheap_correct": true, "router_calls": [{"model": "cheap", "in": 912, "out": 58}], "strong_only_call": {"model": "strong", "in": 912, "out": 58}}], "totals": {"cost_routed": 0.072324, "cost_strong_only": 0.126000, "percent_saved": 42.6, "misrouted_count": 3}}. Token counts are whole numbers, costs are dollars rounded to 6 decimal places, percent_saved to 1.
- Save check.py next to those files and run python check.py from inside my-work/labs/lab06.

### Check it

`check.py` is in this folder. Run it:

```bash
python check.py   (run it from inside my-work/labs/lab06)
```


**You are done when** Ten PASS lines and a final ALL CHECKS PASSED, with exit code 0. Two of them do the real work. "route() reproduces every recorded decision" re-runs your own route() on all 20 questions and fails with the mismatched ids if you edited a routing decision by hand. The three cost lines recompute cost_routed, cost_strong_only and percent_saved from your token counts and your price table, so a total you typed in yourself will not match. The checker also prints one line naming what it did not check: whether your easy/hard labels are fair, and whether your cheap_correct flags are right.

**If you want more:** Replace the rule based router with the cheapest model itself: one short call that must reply with only the word easy or hard. Add that routing call to router_calls so check.py bills you for it, then see whether the saving survives. On short questions it often does not.
