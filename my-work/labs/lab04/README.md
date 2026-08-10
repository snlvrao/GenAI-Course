# Lab 04: Watch attention happen

**Module 4: The transformer, and what attention means**

In this lab you load a real language model onto your own computer, feed it one sentence, and pull the actual attention numbers out of it. Attention is the step where every word looks at the other words and decides which ones matter, and in a running model it is just a square grid of fractions, one grid per head per layer. You will print one row of that grid, watch it add up to 1, and see the exact zeros that prove a GPT-style model cannot look forward at words it has not reached yet. Nothing here needs an API key, a GPU or a paid account, and before you start you should have already confirmed that `python llm.py` works (see `setup.html`), even though this is the one lab that will not use it.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Check what you already have

This step is a check rather than real work. You are confirming that the two libraries this lab needs are installed in the same Python you are about to run the lab in. torch (PyTorch) is the numerical library that does the actual multiplications inside the model, and transformers is the Hugging Face library that downloads model files and wires them up for you. Every other lab in this course talks to a hosted model through my-work/labs/_shared/llm.py, but a hosted API only ever sends you finished text, so it can never hand you the internal attention numbers you want here. That is why the model has to run locally, and why this is the only lab that ignores llm.py. If you installed sentence-transformers in an earlier module then both libraries came along with it, and you should see two version numbers print, something like a 2.x torch and a 5.x transformers. If either import raises ModuleNotFoundError, run the pip line in the comment and try again.

```python
# if these fail: pip install transformers torch
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
```

- `import torch`: torch is PyTorch, the library that stores the model's numbers as tensors (multi-dimensional arrays) and multiplies them. If this line fails, nothing else in the lab can run, so it is deliberately the first thing you test.
- `import transformers`: transformers is the Hugging Face library that knows how to fetch a model from the internet, load its weights, and run it. It sits on top of torch and cannot work without it.
- `print("torch:", torch.__version__)`: Printing the version proves the import came from the environment you think it did, not some other Python on your machine. It also gives you something concrete to quote if you need help later.

> **Watch out:** If torch imports but transformers does not, you are almost certainly in a different virtual environment from the one where you installed things earlier, so check which python you are running before installing anything twice.

### 2. Load a small model in eager mode

Here you download the model once and load it into memory. distilgpt2 is a shrunken copy of GPT-2 that was trained to imitate the bigger model, about 82 million parameters (a parameter is one learned number inside the model) and roughly 350 MB on disk. The first run downloads it and every later run reads it from a local cache folder, so only the first run is slow. The argument attn_implementation="eager" is the part that matters most in this whole lab: recent versions of transformers default to a faster attention routine that never builds the full weight grid in memory, and if you let that happen, output_attentions in the next step quietly returns None and everything after it fails. "eager" means do it the plain, slower way and keep every intermediate number, which is exactly what you want when those intermediate numbers are the whole point. model.eval() switches off training-only behaviour such as dropout (randomly ignoring parts of the network), so your numbers come out identical every run. You should see the line "6 layers, 12 heads per layer" print.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"          # 6 layers, 12 heads per layer, ~82M parameters

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
model.eval()

print(model.config.n_layer, "layers,", model.config.n_head, "heads per layer")
```

- `MODEL = "distilgpt2"`: This is a name on the Hugging Face hub, not a file path. Keeping it in one variable means the mini-project can swap in "gpt2" by editing a single line.
- `tokenizer = AutoTokenizer.from_pretrained(MODEL)`: The tokenizer is the piece that turns your text into the integer IDs the model actually eats, and back again. It must come from the same model name, because a different model numbers its vocabulary differently and the IDs would mean the wrong words.
- `AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")`: CausalLM means a model that predicts the next token and is only allowed to look backwards. The eager argument forces the slow reference implementation that materialises the full attention grid, which is the only way you get real numbers instead of None.
- `model.eval()`: This puts the model in evaluation mode, turning off dropout and other training-time randomness. Without it your attention numbers would wobble slightly from run to run for no useful reason.
- `print(model.config.n_layer, "layers,", model.config.n_head, "heads per layer")`: config holds the model's own description of itself, so this reads the real shape rather than trusting the comment. Those two numbers, 6 and 12, are the sizes of the loops you write in step 6.

**The maths, spelled out**

```
How 768 numbers per word get split across 12 heads.

Formula: d_head = d_model / n_heads
  d_model is the length of the vector the model carries for each word (768 in distilgpt2).
  n_heads is how many attention heads run side by side inside one layer (12).
  d_head is how many of those 768 numbers each head gets to work with.

Worked example: 768 / 12 = 64. Head 0 works on numbers 1 to 64, head 1 on 65 to 128, and so on up to head 11 on 705 to 768. Every head sees every word, but only a 64-number slice of each word.

How many attention grids one sentence produces: layers x heads = 6 x 12 = 72. That is 72 separate square grids you could inspect, which is why steps 4 and 5 average them and step 6 does not.

Where the roughly 82 million parameters sit:
  word embedding table: 50257 vocabulary entries x 768 = 38,597,376
  position table: 1024 positions x 768 = 786,432
  per layer, the four attention matrices Q, K, V and output: 4 x 768 x 768 = 2,359,296
  per layer, the two feed-forward matrices: (768 x 3072) + (3072 x 768) = 4,718,592
  per layer total: 2,359,296 + 4,718,592 = 7,077,888
  six layers: 6 x 7,077,888 = 42,467,328
  grand total: 38,597,376 + 786,432 + 42,467,328 = 81,851,136

That is the "~82M" in the comment, and the small gap to the real published figure is the bias terms and layer-norm values, which are tiny by comparison.

Intuitively: about two thirds of each layer is the feed-forward part rather than attention, and nearly half the whole model is just the lookup table that turns tokens into vectors. Attention is the interesting part, not the big part.
```

> **Watch out:** If you skip attn_implementation="eager" you will not get an error here, you will get a warning you probably scroll past and then a confusing crash in step 4 because out.attentions is None.

### 3. Run one sentence and look at the shape

Now you push one sentence through the model and catch the attention numbers on the way out. The sentence is a Winograd-style ambiguity test, meaning a sentence where a pronoun has two possible referents and one word decides between them: "it" could be the trophy or the suitcase, and only "big" settles it. Print the token list before anything else, because a token is not the same thing as a word, and GPT-2's vocabulary will most likely chop "suitcase" into two pieces such as "suit" and "case". out.attentions is a tuple with one entry per layer, and each entry is a 4-dimensional tensor shaped (batch, heads, tokens, tokens), so att[0] is layer 0 for all 12 heads. You should see "layers returned: 6", a shape like torch.Size([1, 12, 15, 15]), and a printed list of tokens. Read your own token count off that shape rather than trusting any number written here, because it changes the moment you edit the sentence.

```python
sentence = "The trophy did not fit in the suitcase because it was too big."

enc = tokenizer(sentence, return_tensors="pt")
ids = enc["input_ids"][0].tolist()
tokens = [tokenizer.decode([i]).strip() for i in ids]

with torch.no_grad():
    out = model(**enc, output_attentions=True)

att = out.attentions
print("layers returned:", len(att))
print("shape of one layer:", att[0].shape)   # (batch, heads, tokens, tokens)
print("tokens:", tokens)
```

- `enc = tokenizer(sentence, return_tensors="pt")`: This turns your sentence into integer IDs plus an attention mask. return_tensors="pt" asks for PyTorch tensors rather than plain Python lists, because that is what the model expects to be fed.
- `tokens = [tokenizer.decode([i]).strip() for i in ids]`: Decoding each ID on its own gives you the readable text of that single token, so you can see exactly where the tokenizer split words. .strip() removes the leading space GPT-2 attaches to most tokens, which is what lets tokens.index("it") work in the next step.
- `with torch.no_grad():`: This tells PyTorch not to record the operations it would need for training. You are only reading the model, never updating it, so this saves memory and time and cannot change the answers.
- `out = model(**enc, output_attentions=True)`: output_attentions=True is the switch that makes the model hand back its internal attention grids instead of throwing them away. Without it out.attentions is None, and this is where the eager setting from step 2 pays off.
- `print("shape of one layer:", att[0].shape)`: The four numbers are batch, heads, query tokens and key tokens. Batch is 1 because you sent one sentence, and the last two are equal because every token is compared against every token.

**The maths, spelled out**

```
Size of what you just pulled out, and why long inputs cost so much.

Formula: numbers returned = layers x heads x n x n
  n is the number of tokens the tokenizer produced (read it from your own printed shape).
  layers = 6 and heads = 12 for distilgpt2.

Worked example with n = 15:
  one head, one layer: 15 x 15 = 225 numbers
  all 12 heads, one layer: 12 x 225 = 2,700
  all 6 layers: 6 x 2,700 = 16,200 numbers
  at 4 bytes each that is 64,800 bytes, about 63 KB. Nothing at all.

Now the same sentence stretched to n = 1000 tokens:
  1000 x 1000 = 1,000,000 per head per layer
  x 12 heads x 6 layers = 72,000,000 numbers
  x 4 bytes = 288,000,000 bytes, about 288 MB

The n x n term is why doubling your prompt roughly quadruples the attention work: (2n) x (2n) = 4 x (n x n). Doubling 2,000 tokens to 4,000 multiplies this step by about four, not two.

Intuitively: attention compares every token against every token, so the bill grows with the square of the length. That single fact explains a large part of why long context windows are priced the way they are, and why models get slower and vaguer deep into a long document.
```

> **Watch out:** Do not assume the token count matches the word count, because "suitcase" almost certainly arrives as two tokens and every index you use later shifts if you forget that.

### 4. Read one row and find the zeros

The attention grid has one row per word doing the looking and one column per word being looked at, and every row is forced to add up to exactly 1. This step pulls out the single row belonging to "it" and prints its weight against every other token. Two things should jump out. First, the sum prints as 1.0 or something like 0.999 or 1.001 from rounding, which confirms these are fractions of a fixed budget rather than free-floating scores. Second, every token that comes after "it" prints exactly 0.000, and that is the causal mask seen directly instead of described: a GPT-style model is blocked from looking forward, so those weights are driven to zero before any blending happens. Averaging the 12 heads with .mean(dim=0) gives you one readable grid instead of twelve, at the cost of hiding the interesting disagreements, which is what step 6 goes back for. If you change the sentence, read the printed token list and fix the lookup, because a capitalised "It" is a completely different token from "it".

```python
LAYER = 3                              # 0-indexed, so the 4th of 6 layers
A = att[LAYER][0].mean(dim=0)          # average the 12 heads -> (tokens, tokens)

q = tokens.index("it")
row = A[q]

print("this row adds up to:", round(float(row.sum()), 3))
for j, w in enumerate(tokens):
    print(f"   it -> {w:<10} {float(row[j]):.3f}")
```

- `LAYER = 3`: Layers are numbered from 0, so this is the fourth of six. Middle layers are a reasonable place to look because early layers tend to do local, positional work and the last layer is already busy shaping the next-token prediction.
- `A = att[LAYER][0].mean(dim=0)`: The [0] drops the batch dimension, since you only sent one sentence. mean(dim=0) then averages across the head dimension, turning (12, tokens, tokens) into a single (tokens, tokens) grid you can print.
- `q = tokens.index("it")`: This finds which row belongs to the pronoun by searching the decoded token list. It works only because you stripped the leading spaces in step 3, and it returns the first match if the word appears twice.
- `print("this row adds up to:", round(float(row.sum()), 3))`: This is your proof that softmax did its job and the row is a genuine probability distribution. If it does not print close to 1, something is wrong with your indexing, not with the model.
- `print(f"   it -> {w:<10} {float(row[j]):.3f}")`: The {w:<10} pads each token to 10 characters so the numbers line up in a column, and .3f shows three decimals, which is enough to make masked entries read as a clean 0.000.

**The maths, spelled out**

```
What actually produced the numbers in this row, in three stages.

1. score(i, j) = (q_i dot k_j) / sqrt(d_head)
2. if j > i then score(i, j) = minus infinity        (the causal mask)
3. weight(i, j) = exp(score(i, j)) / sum over all j of exp(score(i, j))   (softmax)

Symbols:
  i is the token doing the looking (the row), j is the token being looked at (the column).
  q_i is token i's query vector, k_j is token j's key vector, each 64 numbers long here.
  "dot" is the dot product: multiply the two vectors position by position and add up all 64 results. A large answer means the two vectors point in a similar direction, which the model reads as a good match.
  d_head = 64, so sqrt(d_head) = 8.
  exp(x) means e to the power x, where e is about 2.71828.

Worked example with three visible columns and raw dot products of 8.0, 4.0 and 6.0:
  divide by 8:      1.00, 0.50, 0.75
  exponentiate:     exp(1.00) = 2.718, exp(0.50) = 1.649, exp(0.75) = 2.117
  add them up:      2.718 + 1.649 + 2.117 = 6.484
  divide each:      2.718 / 6.484 = 0.419
                    1.649 / 6.484 = 0.254
                    2.117 / 6.484 = 0.327
  check:            0.419 + 0.254 + 0.327 = 1.000

That 1.000 is the number your script prints.

The mask in the same arithmetic: exp(minus infinity) = 0, so a masked column adds nothing to the total and comes out as exactly 0.000, not merely small. That is why the tokens after "it" print 0.000 with no rounding fuzz.

Why divide by 8 at all: adding 64 products together can easily give a score in the tens, and exp(30) is about 1.07e13, which would swamp every other column and collapse the row to a single 1.000 with the rest at 0.000. Dividing by sqrt(d_head) keeps scores in a range where softmax still spreads weight around, which during training also keeps the gradients from shrinking to nothing.

Intuitively: softmax turns any list of scores into fractions that add to 1, so each row of attention is a budget of exactly 100 percent that one word must spend across the words it is allowed to see.
```

> **Watch out:** If you edited the sentence, tokens.index("it") raises ValueError: 'it' is not in list, which means your pronoun was tokenised differently (often capitalised, or glued to punctuation), so print tokens and pick the index by eye.

### 5. Deal with the first-token sink

Run step 4 and you will notice a large weight sitting on the very first token, often larger than anything that looks meaningful. That is not the model deciding "The" is important. Because every row is forced to add up to 1, a head that has genuinely nothing useful to say still has to spend its whole budget somewhere, and GPT-style models learn to dump the leftover on position 0 where it does the least harm. This is a known and well documented effect called an attention sink, and if you do not remove it you will spend the rest of the lab admiring an artefact. This step zeroes the sink and each word's attention to itself (which is also usually large and also tells you nothing), then prints the top 3 remaining sources for every token. Expect fairly dull results, mostly the previous word or nearby punctuation, and expect the first couple of rows to be near-empty because there is barely anything behind them to look at.

```python
def top_sources(A, tokens, k=3):
    for i, w in enumerate(tokens):
        if i < 2:
            continue                   # nothing meaningful behind these yet
        r = A[i].clone()
        r[i] = 0.0                     # drop the word looking at itself
        r[0] = 0.0                     # drop the first-token sink
        vals, idx = torch.topk(r, k)
        picks = ", ".join(f"{tokens[j]} {v:.2f}" for v, j in zip(vals.tolist(), idx.tolist()))
        print(f"{w:<10} -> {picks}")

top_sources(A, tokens)
```

- `if i < 2:
    continue`: Token 0 can only look at itself and token 1 can only look at token 0 and itself, so once you remove the sink and the self-weight there is nothing left to report. Skipping them avoids printing rows made entirely of zeros.
- `r = A[i].clone()`: clone() makes a separate copy of the row before you edit it. Without it, the next two lines would permanently overwrite values inside A, and step 6 or a re-run would silently read corrupted data.
- `r[i] = 0.0`: Position i is the word looking at itself, which is usually one of the biggest numbers in the row and carries no information about which other word matters. Zeroing it stops it from filling your top 3.
- `r[0] = 0.0`: Position 0 is the attention sink, the parking spot for leftover weight. Removing it is the difference between a readable result and a list where "The" wins every single row.
- `vals, idx = torch.topk(r, k)`: topk returns the k largest values and, separately, the positions where they were found. You need idx because a value on its own does not tell you which token it belongs to.
- `", ".join(f"{tokens[j]} {v:.2f}" for v, j in zip(vals.tolist(), idx.tolist()))`: zip pairs each value with its position so you can look the word back up in tokens. .tolist() converts the PyTorch tensors into ordinary Python numbers so the f-string formats them cleanly.

**The maths, spelled out**

```
What zeroing two entries does to a row that was built to add up to 1.

Formula: after r[0] = 0 and r[i] = 0, the row now sums to
  S = 1 - w_sink - w_self
  w_sink is the weight that was on the first token, w_self is the weight the word gave itself.

Worked example for the row belonging to "it":
  first token "The": 0.55
  itself "it":       0.18
  "trophy":          0.09
  "suit":            0.07
  everything else:   0.11 spread thinly
  total:             1.00

  After zeroing: S = 1 - 0.55 - 0.18 = 0.27
  So "trophy" at 0.09 is really 0.09 / 0.27 = 0.33, a third of the attention that was actually about the sentence rather than about housekeeping.

The code does not perform that division, and for this lab that is fine, because dividing every surviving number by the same 0.27 cannot change which one is largest. It only changes how small the printed numbers look. Be careful when comparing two different rows though, since each row has its own S, so 0.09 in one row and 0.09 in another are not the same share.

Intuitively: softmax has no way to output "none of these are relevant". A head with nothing to say still has to spend 100 percent of its budget, so it learns to park the remainder on position 0, and that parked weight is bookkeeping, not meaning.
```

> **Watch out:** For the earliest rows, topk(r, 3) will pad its answer with 0.00 entries because there simply are not three visible tokens left after the mask and the two zeroings, so read those lines as empty rather than as findings.

### 6. Check whether the heads agree

Averaging 12 heads into one grid, which is what step 4 did, is exactly what hides the interesting behaviour. If one head points hard at "trophy" and the other eleven ignore it, the average buries that head under the eleven boring ones. This step keeps the heads separate and prints, for every layer and every head, the single token that head weighted most heavily for the word "it". You will see the heads disagree loudly with each other, and most picks will be dull, usually the previous token or a piece of punctuation, which is the honest normal result. If any head picks "trophy" or a piece of "suitcase", write down the layer number and head number, because the mini-project asks you to test whether that head is tracking meaning or just got lucky. There are only 6 x 12 = 72 head positions in distilgpt2, so this whole search finishes instantly and prints six lines.

```python
q = tokens.index("it")

for layer in range(len(att)):
    picks = []
    for head in range(att[layer].shape[1]):
        r = att[layer][0, head, q].clone()
        r[q] = 0.0
        r[0] = 0.0
        picks.append(tokens[int(torch.argmax(r))])
    print(f"layer {layer}: {picks}")
```

- `for layer in range(len(att)):`: len(att) is the number of layers, 6 here, because out.attentions is a tuple with one entry per layer. Reading it from the data means the loop still works if you switch MODEL to gpt2 with its 12 layers.
- `for head in range(att[layer].shape[1]):`: shape[1] is the head count, taken from the tensor rather than hard-coded as 12. Same reason: swapping models in the mini-project should not require editing this loop.
- `att[layer][0, head, q]`: This indexes straight into the 4-dimensional tensor: batch 0, this head, row q for the pronoun, leaving one vector of weights across all tokens. Crucially there is no .mean() here, so you are seeing one head's raw opinion.
- `r[q] = 0.0
r[0] = 0.0`: The same two clean-up steps as before, self-attention and the sink, applied per head this time. Without them almost every head would report "The" and the whole print would be useless.
- `tokens[int(torch.argmax(r))]`: argmax returns the position of the largest weight, not the weight itself, so you index back into tokens to get a readable word. int() converts the 0-dimensional tensor into a plain Python integer that list indexing accepts.

**The maths, spelled out**

```
Why averaging hides heads, and how much of what you see is luck.

Averaging formula: A_mean(i, j) = (1 / H) x sum over h of A_h(i, j), with H = 12 heads.

Worked example. Suppose head 4 puts 0.90 on "trophy" and the other 11 heads put only 0.02 each on it:
  sum  = 0.90 + (11 x 0.02) = 0.90 + 0.22 = 1.12
  mean = 1.12 / 12 = 0.093
Now a dull column, the previous word, that every head gives 0.30:
  mean = (12 x 0.30) / 12 = 0.30
In the averaged grid from step 4 the dull column wins 0.30 against 0.093, even though one head was pointing hard at "trophy". Averaging divides a single strong signal by 12 while leaving a consistent weak one untouched, which is precisely why this step prints heads separately.

How big the search is: layers x heads = 6 x 12 = 72 head positions in distilgpt2, and 12 x 12 = 144 in gpt2.

How much is chance. Suppose "it" sits at position 10, so it can see positions 0 to 10, which is 11 columns. Remove itself and position 0 and 9 candidates remain.
  chance of a random pick landing on the right noun = 1 / 9 = 0.111
  expected lucky hits across 72 positions = 72 x 0.111 = about 8

So finding eight heads that "pick trophy" proves nothing at all. That is exactly why the mini-project demands a head whose pick flips when you flip "big" to "small": matching once is cheap, matching the change is not.

Intuitively: argmax tells you where a head put its largest weight, but not whether that weight was 0.6 or 0.04. A confident head and a head that is barely distinguishable from noise look identical in this printout.
```

> **Watch out:** argmax always returns something, so a head whose top weight is 0.03 will still be printed as a firm-looking pick, and the fix is to print the weight alongside the token before you believe any of it.

## You are done when

Your script runs top to bottom with no errors and prints all of the following: the line "6 layers, 12 heads per layer"; a shape like torch.Size([1, 12, N, N]) where N is your token count; a row for "it" that sums to 1.0 within rounding; exact 0.000 entries for every token that comes after "it", which you can point at as the causal mask; a top-3 source list per token with the first-token sink and self-attention removed; and six lines, one per layer, each listing 12 per-head picks for "it". You should also be able to say out loud why the per-head picks in step 6 differ from the averaged picks in step 5.

---

## Mini-project: Attention detective

Find out whether any attention head really tracks which noun a pronoun refers to, or whether the ones that look right are luck. You record every head's pick for both halves of a minimal pair in findings.json, and check.py re-runs the flip test against your own data.

- Write a minimal pair: two sentences identical except for one word, where that word flips what the pronoun refers to. The standard pair is "The trophy did not fit in the suitcase because it was too big." against the same sentence ending "too small." Record the correct noun for each half as the token the lab prints, so "suitcase" is usually the token "suit".
- Run the step 6 loop from the lab on both sentences and collect the per-head picks into two grids, one row per layer and one entry per head. distilgpt2 gives 6 x 12 = 72 entries per sentence, gpt2 gives 144.
- Look for a position where grid A picks sentence A's noun and grid B picks sentence B's noun. A head that picks the same noun in both has learned nothing about the sentence in front of it.
- Save findings.json next to check.py in my-work/labs/lab04/ with exactly these eight keys: model (string), n_layers and n_heads (ints), pronoun (string, "it"), head_positions_checked (int, must equal n_layers x n_heads), pair (list of two objects, each {"sentence": ..., "answer": ...}), picks ({"A": grid, "B": grid}, each grid n_layers lists of n_heads token strings), claim ({"layer": L, "head": H} or null).
- Set claim to the layer and head you found, or to null if none did. null is the common and honest result, so do not round up.
- Run python check.py from my-work/labs/lab04/.

### Check it

`check.py` is in this folder. Run it:

```bash
cd my-work/labs/lab04, then run: python check.py
```


**You are done when** check.py prints nine PASS lines and then ALL CHECKS PASSED, exiting 0. The last check is the one that bites: it recomputes the flip test over your own grids, so a claimed head whose picks do not actually flip fails, and a claim of null fails if some head in your data did flip. It also prints two lines naming what it cannot check.

**If you want more:** Take the head you found and run it on five unrelated pronoun sentences. If it only works on the sentence you found it with, you have a coincidence, not a mechanism. check.py cannot judge that for you.
