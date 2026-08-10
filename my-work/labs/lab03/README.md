# Lab 03: Look inside the tokenizer

**Module 3: How text becomes numbers**

You will look at your own sentences three ways: as characters, as tokens (chunks of text taken from a fixed list the model was built with), and as points in space. Along the way you will price a single request in real dollars and see the exact arithmetic behind a similarity score, so nothing in this module stays magic. By the end you will have a script that takes a question and ranks five sentences by closeness in meaning, with no API key and no paid service. Everything runs on your own machine on the CPU, and the only download is a model of about 90 MB that is cached after the first run, so before you start make sure python llm.py works (see setup.html).

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Open the lab folder and install the tools

Work inside my-work/labs/lab03, which the course already ships. These three lines are shell commands, not Python, so type them at your terminal prompt and not inside a .py file. You are reusing the one virtual environment you built on the setup page, so there is nothing new to create here, only to switch on. Installing sentence-transformers also pulls in transformers and torch underneath, which is exactly what you want, so do not install those separately. On macOS or Linux the first line is source my-work/.venv/bin/activate instead. Nothing here needs an API key, and the first time you run a script later it downloads a model of about 90 MB once and caches it in your home folder. You should see your prompt start with (.venv), and pip should finish with a line beginning Successfully installed.

```python
.\my-work\.venv\Scripts\Activate.ps1
cd my-work\labs\lab03
pip install "sentence-transformers==5.6.1"
```

- `.\my-work\.venv\Scripts\Activate.ps1`: Points this terminal window at the copy of Python inside the course's .venv, so that python and pip now mean the private ones. It only affects the window you type it in, so you have to run it again in every new terminal. Run it from the top of the GenAI-Course folder.
- `cd my-work\labs\lab03`: Moves you into the folder for this lab, which already exists. Everything you write here stays inside my-work.
- `pip install "sentence-transformers==5.6.1"`: Installs one package at one exact version. Pinning the version means the numbers you see later match the numbers described here, and the quotes stop the shell from treating the == as something of its own.

**The maths, spelled out**

```
How the ~90 MB download is made up.

Formula:
bytes on disk = number of parameters x bytes per parameter

What the symbols mean:
- number of parameters = how many learned numbers the model holds
- bytes per parameter = how much space one number takes, 4 bytes for the usual float32 format

Worked example for all-MiniLM-L6-v2:
about 22,700,000 parameters
22,700,000 x 4 = 90,800,000 bytes
90,800,000 / 1,000,000 = about 91 MB

What it means:
The download is not code, it is the numbers the model learned during training, one number per parameter and four bytes for each. That is also why a bigger model is a bigger download in a fairly straight line.
```

> **Watch out:** If your prompt does not show (.venv) after the activate line, activation did not happen and pip will install into your system Python instead, and on Windows PowerShell the give-away is an error mentioning execution policy. On Windows the activate line is refused the first time, with "running scripts is disabled on this system". That is the default on a fresh install, not something you broke. Run Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned once, which needs no administrator rights, then run the activate line again. If you would rather not change that setting, skip activation entirely and call the environment by its path instead: .\my-work\.venv\Scripts\python.exe rank.py.

### 2. See your own text as tokens

Save this as tokens.py and run it. A tokenizer is the piece of code that cuts your text into chunks from a fixed list and swaps each chunk for its row number in that list, and this step makes that cutting visible. You need to see it because token counts drive your bill, fill your context window, and explain why models miscount the letters in a word: they never saw the letters, only the chunks. Two different models chop the same sentence differently, so expect two blocks of output with two different token counts for identical text. The MiniLM tokenizer uses ## to mark a piece that continues a word, so antidisestablishmentarianism comes back as several pieces, and the Qwen one instead uses a special mark for a leading space. The second call downloads only the tokenizer files (a few MB), not the model weights, so it is fast, and if that download fails you can delete the second show line and carry on. Then replace the text with your own: try a URL, a code snippet, an emoji, a long number like 8675309, and a sentence in another language, and watch the characters-per-token number move.

```python
# tokens.py
from transformers import AutoTokenizer

text = "Tokenization splits text into pieces. Antidisestablishmentarianism is expensive."

def show(model_name, text):
    tok = AutoTokenizer.from_pretrained(model_name)
    ids = tok.encode(text, add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    print(f"\n{model_name}")
    print(f"  characters: {len(text)}   tokens: {len(ids)}")
    print("  " + " | ".join(pieces))
    print(f"  first 10 IDs: {ids[:10]}")

show("sentence-transformers/all-MiniLM-L6-v2", text)
show("Qwen/Qwen3-Embedding-0.6B", text)
```

- `from transformers import AutoTokenizer`: AutoTokenizer is a loader that reads the model's own config file and picks the matching tokenizer for you. That is why the same two lines work for MiniLM and for Qwen even though they cut text using different schemes.
- `tok = AutoTokenizer.from_pretrained(model_name)`: Downloads that model's chunk list and its joining rules the first time, then caches them in your home folder so later runs are instant. It fetches only the tokenizer files, a few MB, not the model weights.
- `ids = tok.encode(text, add_special_tokens=False)`: Turns your text into a list of whole numbers, one per chunk. add_special_tokens=False leaves out the extra bookkeeping markers the model normally wraps around a sentence, so the count you print is the count of your text and nothing else.
- `pieces = tok.convert_ids_to_tokens(ids)`: Goes the other way, from row numbers back to the visible chunks, purely so you can read them. Nothing downstream uses pieces, it is there for your eyes.
- `print(f"  characters: {len(text)}   tokens: {len(ids)}")`: Prints the two counts side by side. The ratio between them is the number you actually care about, because it is what changes when you swap in a URL or a code snippet.

**The maths, spelled out**

```
Characters per token, and tokens per word.

Formula:
characters per token = number of characters / number of tokens
tokens per word = number of tokens / number of words

What the symbols mean:
- number of characters = len(text), every letter, space and full stop counted
- number of tokens = len(ids), how many chunks the tokenizer produced
- number of words = how many space-separated words you wrote

Worked example with the lab sentence:
The sample text is 80 characters and 8 words.
Suppose your run prints 17 tokens for the MiniLM model.
80 / 17 = 4.7 characters per token
17 / 8 = 2.1 tokens per word

The common rule of thumb for ordinary English is about 4 characters per token and about 1.3 tokens per word. This sentence runs higher than 1.3 because one very long rare word gets cut into many pieces.

Scaling it up with the rule of thumb:
1,000 words of ordinary English x 1.3 = about 1,300 tokens
4,000 words x 1.3 = about 5,300 tokens

What it means:
More common your text, fewer tokens per character. Rare words, URLs, code and non-English text all break into more, smaller pieces, and each extra piece is something you pay for and something that eats your context window.
```

> **Watch out:** The second show line reaches out to the internet for the Qwen tokenizer, so a network or access error there is about that download and not about your code, and deleting that one line lets the rest of the step finish.

### 3. Turn a token count into money

Save this as cost.py. It counts the tokens in its own source file, then prices that as one request against four published price lists. You need this step because prices are quoted per million tokens, which is impossible to feel until you do the division yourself. The count is approximate, because each provider uses its own chunk list and you are counting with MiniLM's, so treat it as accurate to within about 20 percent. Expect a few hundred input tokens and totals in the thousandths of a dollar, which looks harmless on one request. Now change n_input to 200,000 and imagine 30 turns of an agent loop, and you will see why the input price matters far more than the output price. The same token count also fills the context window (the maximum text a model can hold in one request), and although the GPT-5.6 tiers all advertise 1M tokens, accuracy drops well before that limit, which you will meet later as context rot.

```python
# cost.py
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

with open("cost.py", "r", encoding="utf-8") as f:
    document = f.read()

n_input = len(tok.encode(document, add_special_tokens=False))
n_output = 400

# US dollars per million tokens, list prices checked 5 August 2026
prices = {
    "GPT-5.6 Sol":    (5.00, 30.00),
    "GPT-5.6 Terra":  (2.00, 12.00),
    "GPT-5.6 Luna":   (0.20,  1.20),
    "Gemini 3.1 Pro": (2.00, 12.00),
}

print(f"input tokens: {n_input}, output tokens assumed: {n_output}")
for name, (p_in, p_out) in prices.items():
    cost = n_input / 1_000_000 * p_in + n_output / 1_000_000 * p_out
    print(f"{name:>16}  ${cost:.5f}")
```

- `with open("cost.py", "r", encoding="utf-8") as f:`: The script reads its own file, so you always have a real document to measure with nothing to download. with closes the file for you afterwards, and encoding="utf-8" stops Windows from decoding the file with an older regional character set and mangling anything unusual.
- `n_input = len(tok.encode(document, add_special_tokens=False))`: Only the length of the ID list matters here, so the IDs themselves are thrown away immediately. This is the honest way to count, as opposed to guessing from word count.
- `n_output = 400`: This is an assumption, not a measurement, because you cannot know how long a reply will be before you ask for it. The print line says "assumed" out loud so you never mistake it for a real number.
- `prices = { "GPT-5.6 Sol": (5.00, 30.00), ... }`: Each entry stores input price and output price together as a pair, in dollars per million tokens. Keeping them in one pair makes it much harder to accidentally use the output price for input, which is the classic estimating mistake.
- `cost = n_input / 1_000_000 * p_in + n_output / 1_000_000 * p_out`: Dividing by a million first converts a token count into millions of tokens, which is the unit the price is quoted in. The underscores in 1_000_000 are digit separators that Python ignores, they only make the number readable.
- `print(f"{name:>16}  ${cost:.5f}")`: &gt;16 right-aligns the name in a 16 character column so the prices line up in a readable table. .5f shows five decimal places, which you need because a single request costs a fraction of a cent.

**The maths, spelled out**

```
The cost formula, in full.

Formula:
cost = (n_input / 1,000,000) x price_in + (n_output / 1,000,000) x price_out

What the symbols mean:
- n_input = number of tokens you send, including the whole conversation so far
- n_output = number of tokens the model writes back
- price_in = dollars per million input tokens, from the table
- price_out = dollars per million output tokens, from the table

Worked example, one request of 350 input tokens and 400 output tokens.

On GPT-5.6 Sol at 5.00 and 30.00:
350 / 1,000,000 = 0.00035
0.00035 x 5.00 = 0.00175 dollars for input
400 / 1,000,000 = 0.0004
0.0004 x 30.00 = 0.01200 dollars for output
total = 0.01375 dollars

The same request on GPT-5.6 Luna at 0.20 and 1.20:
0.00035 x 0.20 = 0.00007
0.0004 x 1.20 = 0.00048
total = 0.00055 dollars, which is 25 times cheaper

Now the agent loop, 200,000 input tokens per turn for 30 turns, on Sol:
input tokens = 200,000 x 30 = 6,000,000
6,000,000 / 1,000,000 = 6
6 x 5.00 = 30.00 dollars of input
output tokens = 400 x 30 = 12,000
12,000 / 1,000,000 = 0.012
0.012 x 30.00 = 0.36 dollars of output
30.00 against 0.36 is about 83 times more spent on input than output.

Prompt caching, roughly:
Providers charge about one tenth for text they have already processed at the start of a prompt. If nearly all of that 6,000,000 input tokens is an unchanged prefix, the 30.00 dollars falls towards 3.00 dollars. Caching matches on prefixes only, so one changed character near the top throws away everything after it, which is why timestamps and session IDs belong at the end of a prompt.

What it means:
In agent work you pay mostly for what you send, not for what you get back, because every turn resends the whole conversation and every tool result.
```

> **Watch out:** Run the script from the same folder that holds cost.py, otherwise open() raises FileNotFoundError because it looks in whatever folder your terminal happens to be sitting in.

### 4. Turn sentences into vectors

Save this as embed.py. An embedding is a list of numbers that stands in for the meaning of a piece of text, and this step makes five of them. The shape line prints (5, 384), which tells you five sentences came back as five rows of 384 numbers each, and that 384 does not change if you make the sentences ten times longer, because the model averages the per-token vectors into a single row. That averaging is a real simplification and it has a limit: all-MiniLM-L6-v2 reads only about the first 256 tokens and quietly drops the rest, with no error and no warning. Setting normalize_embeddings=True scales every row to length one, which is why the last line prints 1.0 and why the next step can use a plain multiply-and-add instead of a similarity library. Look at the first eight numbers and notice they mean nothing on their own, nobody labeled them, and only the distance between whole vectors carries information. Without this step you have text, and text cannot be compared with arithmetic.

```python
# embed.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "I forgot my login details and cannot get in.",
    "The refund took nine days to arrive.",
    "Boil the rice for twelve minutes.",
    "Our office is closed on Monday.",
]

vectors = model.encode(sentences, normalize_embeddings=True)
print("shape:", vectors.shape)
print("first 8 numbers of sentence 0:", vectors[0][:8].round(3))
print("length of vector 0:", float((vectors[0] ** 2).sum()) ** 0.5)
```

- `model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`: Loads the actual model weights, about 90 MB the first time and instant afterwards from cache. Do this once at the top and never inside a loop, because loading takes seconds while encoding a sentence takes milliseconds.
- `vectors = model.encode(sentences, normalize_embeddings=True)`: Passing the whole list in one call lets the library process all five sentences together, which is much faster than five separate calls. normalize_embeddings=True divides each row by its own length for you, so every row comes out at length one.
- `print("shape:", vectors.shape)`: shape reports (rows, columns) for the array that came back. Printing it early is worth the one line, because a surprising shape is the most common reason a later multiply fails.
- `float((vectors[0] ** 2).sum()) ** 0.5`: Squares all 384 numbers, adds them up, then takes the square root, which is the length of the vector. It exists only as proof that normalization really happened, so you can trust the plain dot product in the next step.

**The maths, spelled out**

```
Vector length, and what normalizing does.

Formula:
length of v = square root of (v1 x v1 + v2 x v2 + ... + vn x vn)
normalized vector u = v divided by (length of v), every number divided by the same amount

What the symbols mean:
- v1 to vn are the individual numbers in the vector
- n is how many numbers there are, 384 for this model
- length is also called the magnitude or the norm

Worked example with 3 numbers instead of 384, so you can follow it by hand:
v = [3, 0, 4]
squares: 9, 0, 16
sum: 9 + 0 + 16 = 25
length: square root of 25 = 5
normalized: [3/5, 0/5, 4/5] = [0.6, 0, 0.8]
check: 0.6 x 0.6 + 0 + 0.8 x 0.8 = 0.36 + 0.64 = 1.0, square root of 1.0 = 1.0

How much space these take:
384 numbers x 4 bytes each (float32) = 1,536 bytes per sentence
10,000 documents x 1,536 = 15,360,000 bytes, about 15 MB

What it means:
Normalizing keeps the direction of the vector and throws away its size. Direction is where the meaning lives, while size mostly tracks incidental things like how long the sentence was, so throwing size away makes comparisons fairer. The storage sum shows why keeping embeddings for tens of thousands of documents is cheap.
```

> **Watch out:** The length may print as 0.99999994 rather than exactly 1.0, which is ordinary floating point rounding and not a sign that normalization failed.

### 5. Rank five sentences against a question

Save this as rank.py and run it. This is the payoff of the whole module: you embed five sentences and one question, then score the question against every sentence with a single multiply. Because every vector was scaled to length one in the previous step, the dot product (multiply matching positions, add the results) is exactly cosine similarity, so no similarity library is needed. Expect the two login sentences on top with scores around 0.5 to 0.7, and the rice and office sentences near 0.0 to 0.2. Look carefully at the query, "I can't sign in to my account": it shares no meaningful word with either winner, so a keyword search would return nothing, and that gap is the entire reason embeddings exist. The library also offers model.similarity(vectors, q), which does the same arithmetic if you prefer a named method. Swap in your own five sentences and your own question before moving on, because seeing it work on your own text is what makes it stick.

```python
# rank.py
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "I forgot my login details and cannot get in.",
    "The refund took nine days to arrive.",
    "Boil the rice for twelve minutes.",
    "Our office is closed on Monday.",
]
query = "I can't sign in to my account"

vectors = model.encode(sentences, normalize_embeddings=True)
q = model.encode(query, normalize_embeddings=True)

scores = vectors @ q
for i in np.argsort(-scores):
    print(f"{scores[i]: .3f}  {sentences[i]}")
```

- `q = model.encode(query, normalize_embeddings=True)`: query is a plain string and not a list, so this returns one flat vector of 384 numbers with shape (384,). That flat shape is exactly what the multiply on the next line expects.
- `scores = vectors @ q`: @ is numpy's matrix multiply. A (5, 384) array times a (384,) vector gives 5 numbers, one score per sentence, and each of those 5 numbers is 384 multiplies added together.
- `np.argsort(-scores)`: argsort gives you the positions that would put the array in order from smallest to largest, and negating the scores flips that into largest first. It hands back positions rather than values, so the same i reaches both the score and the matching sentence.
- `print(f"{scores[i]: .3f}  {sentences[i]}")`: The space right after the colon reserves a column for a minus sign, so positive and negative scores stay lined up under each other. Three decimal places is enough, since anything beyond that is noise you cannot act on.

**The maths, spelled out**

```
Dot product and cosine similarity, in full.

Formulas:
dot product: a . b = a1 x b1 + a2 x b2 + ... + an x bn
cosine similarity: cos = (a . b) / (length of a x length of b)
if both vectors already have length one: cos = a . b, because the bottom is 1 x 1 = 1

What the symbols mean:
- a and b are two vectors, here 384 numbers each
- a1, b1 and so on are the numbers at matching positions
- cos is the cosine of the angle between the two vectors, running from -1 to 1

Worked example with 3 numbers so you can check it by hand:
a = [0.6, 0.8, 0]
b = [1, 0, 0]
dot: 0.6 x 1 + 0.8 x 0 + 0 x 0 = 0.6
length of a: square root of (0.36 + 0.64 + 0) = 1
length of b: square root of (1 + 0 + 0) = 1
cos = 0.6 / (1 x 1) = 0.6, which is an angle of about 53 degrees

Now a third vector c = [0, 1, 0]:
a . c = 0.6 x 0 + 0.8 x 1 + 0 x 0 = 0.8
0.8 beats 0.6, so c is closer in meaning to a than b is.

What the range means in practice:
-1 is opposite direction, 0 is unrelated, 1 is the same direction. These models rarely give negatives. Unrelated pairs usually land around 0.0 to 0.2 and good matches around 0.5 to 0.9. There is no universal cut-off you can copy from anywhere.

How much work this is:
384 multiplies and 383 additions per sentence, so 5 sentences is 1,920 multiplies, which a laptop does instantly. One million documents is 384 million multiplies per query, which is why real systems add a search index later instead of comparing against everything.

What it means:
Cosine similarity asks "are these two vectors pointing the same way" and ignores how long they are. Normalizing first turns that question into one cheap multiply-and-add.
```

> **Watch out:** If you write model.encode([query], normalize_embeddings=True) with square brackets, q comes back with shape (1, 384) and vectors @ q fails with a shapes-not-aligned ValueError.

### 6. Find where similarity misleads you

Save this as limits.py. This step is about learning what a high score does not tell you, which matters more than any of the wins in step 5. Before you run it, write down your guess for each of the four scores, because being wrong on paper is what makes the lesson land. The love and hate pair usually scores above 0.8, and the 9am against 5pm pair also scores high, which shows that similarity means "about the same thing" rather than "says the same thing", and definitely not "is correct". The river bank pair usually lands lower than people expect, often near 0.3, because the model adjusts each token's numbers using the rest of the sentence, so bank beside interest rates ends up somewhere different from bank beside river. The Friday pair is the one honest match in the list, and it scores high for the right reason. Finish by noting the score that separates your good matches from your bad ones, because that number is your threshold, and it belongs to this model on this data and will not transfer to another model.

```python
# limits.py
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pairs = [
    ("I love cats.", "I hate cats."),
    ("The bank raised interest rates.", "We sat on the river bank."),
    ("Ship it on Friday.", "Send the package on Friday."),
    ("The meeting is at 9am.", "The meeting is at 5pm."),
]

for a, b in pairs:
    va, vb = model.encode([a, b], normalize_embeddings=True)
    print(f"{float(np.dot(va, vb)): .3f}   {a!r} vs {b!r}")
```

- `pairs = [("I love cats.", "I hate cats."), ...]`: A list of two-item tuples, so each entry is one comparison. Keeping the test cases in a list separate from the loop means you can add your own awkward pairs without touching any logic.
- `for a, b in pairs:`: Python unpacks each tuple straight into two names, so a is the first sentence and b the second. That saves writing pair[0] and pair[1] everywhere.
- `va, vb = model.encode([a, b], normalize_embeddings=True)`: Encodes both sentences in one call, which is faster than two calls, and then unpacks the two returned rows into two names. This works because the result is a 2-row array and Python hands out one row per name.
- `float(np.dot(va, vb))`: np.dot on two flat arrays gives the dot product, which equals cosine similarity here because both were normalized to length one. Wrapping it in float() turns a numpy value into a plain Python number so the number formatting behaves predictably.
- `{a!r} vs {b!r}`: !r prints the value the way Python would write it, so the quotation marks stay visible around each sentence. That makes trailing spaces and odd characters obvious instead of invisible.

**The maths, spelled out**

```
Why "I love cats" scores high against "I hate cats".

This is a deliberate simplification to build intuition. The model does not literally store a topic part and a sentiment part, but splitting the vector this way predicts the number you will see.

Setup:
Imagine each unit vector is made of two pieces at right angles to each other, a shared topic piece t (cats) and a sentiment piece s (love versus hate). Because they are at right angles, t . s = 0 and the weights must satisfy weight_t squared + weight_s squared = 1.

Pick weight_t = 0.95 and weight_s = 0.312, since:
0.95 x 0.95 = 0.9025
0.312 x 0.312 = 0.0973
0.9025 + 0.0973 = 0.9998, near enough to 1

Vector A (love) = 0.95 t + 0.312 s
Vector B (hate) = 0.95 t - 0.312 s

Dot product:
(0.95 x 0.95) + (0.312 x -0.312) = 0.9025 - 0.0973 = 0.805

So the score is about 0.81 even though the sentiment is exactly reversed.

Choosing your threshold:
Suppose on your own data the good matches score 0.62, 0.58 and 0.55, and the bad ones score 0.31, 0.24 and 0.18. The gap sits between 0.55 and 0.31, so a sensible cut is the midpoint:
(0.55 + 0.31) / 2 = 0.43
Anything at or above 0.43 you keep, anything below you drop.

What it means:
If most of a vector is spent describing the topic, flipping the small part that carries the opinion barely moves the score. A high score tells you two texts are about the same subject, and nothing at all about whether they agree or whether either one is true.
```

> **Watch out:** Exact scores shift a little between library versions, machines and rounding, so never hardcode a number you saw once, and recompute your threshold whenever you change embedding model.

## You are done when

You are done when rank.py prints a ranked list where the two login sentences beat the other three even though the query shares no words with them, embed.py prints shape: (5, 384) with a vector length of 1.0, and you can state how many tokens your own paragraph is and what one request containing it would cost on two different models from the table in cost.py.

---

## Mini-project: Find the odd one out

Build a tool that takes five sentences, four about one thing and one not, and names the odd one using embeddings and arithmetic only, with no LLM call and no keyword rules. It must write its results to odd_one_out.json, which check.py then verifies for you.

- Make a folder for the mini-project, save check.py (below) into it, and write your own script there as odd_one_out.py. Use the same model as the lab, sentence-transformers/all-MiniLM-L6-v2, so the model is already cached and nothing needs the internet.
- Write two sets of five sentences each. The easy set is four sentences on one topic plus one clearly unrelated. The hard set is four on one topic plus one that shares words with them but not meaning, for example four about apples the fruit and one about Apple the company. For each set, record expected_odd_index, the position you think is the odd one, counting from 0.
- Embed each set in one model.encode(sentences, normalize_embeddings=True) call, then build the 5x5 score matrix with V @ V.T. Every entry on the diagonal comes out at 1.0, because each sentence is compared with itself.
- Fit score for sentence i is the mean of its four scores against the others. Exclude the diagonal: a 1.0 in every row would flatten the differences and hide the odd one. Then predicted_odd_index is the position of the lowest fit score, and confidence_gap is the second lowest score minus the lowest.
- Write odd_one_out.json in that same folder, every number rounded to 4 decimals, in exactly this shape: {"model": "sentence-transformers/all-MiniLM-L6-v2", "sets": [{"name": "easy", "sentences": [five strings], "expected_odd_index": 4, "similarity_matrix": [five rows of five numbers], "fit_scores": [five numbers], "predicted_odd_index": 4, "confidence_gap": 0.2043, "correct": true}, {"name": "hard", ...the same eight keys...}]}. The correct field is simply predicted_odd_index == expected_odd_index.
- Run python check.py in that folder and fix whatever it reports.

### Check it

`check.py` is in this folder. Run it:

```bash
python check.py
```


**You are done when** check.py prints 21 PASS lines and ends with ALL CHECKS PASSED, and its exit code is 0. It re-embeds your own sentences offline and compares them against your saved matrix, so the numbers have to be real ones you computed. If the hard set fooled your tool, its correct field is false and the checker still passes: it prints a NOTE saying that outcome is not judged, because a tool that fails on hard input is a finding, not a mistake.

**If you want more:** Print "not confident" when confidence_gap is under 0.05, and see how often that fires on hard sets. Then change the model field to Qwen/Qwen3-Embedding-0.6B, regenerate the file and run check.py again: it re-embeds with whichever model you name, so every check still applies. Raw scores from two models are not comparable, but the ordering usually is.
