# Lab 20: Build TinyGPT

**Module 20: B2 · Build the transformer**

You are building and inspecting, not training. Everything here runs in seconds on a laptop processor with no graphics card. Keep my-work/labs/_shared/tinygpt.py open beside you and put your scratch scripts in my-work/labs/lab20/.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Run it before you read it

The file has a shape check at the bottom under `if __name__ == "__main__"`. Run it and you get the parameter count, the input and output shapes, and the loss on random data. The loss is the number to sit with: 4.174 is what you score by guessing uniformly over 65 characters, and an untrained model scores about that or slightly worse, because random weights are worse than an honest shrug. Measured runs printed 4.495 and 4.249, and it moves every run because the weights are freshly random each time. If your number is near 4.2 to 4.5, the model is wired correctly and knows nothing, which is exactly the right starting point.

```python
python my-work/labs/_shared/tinygpt.py

# parameters      824,897
# input  (2, 16)  ->  logits (2, 16, 65)
# loss on random data 4.249 (a fair coin over 65 options would be 4.174)
# generated 20 tokens from an empty prompt
```

- `input (2, 16) -> logits (2, 16, 65)`: 2 sequences of 16 tokens, and for every one of the 32 token positions a score for all 65 vocabulary entries. The model predicts at every position at once, not just the last one.
- `m(x, x)`: Passing the input as its own targets is deliberate nonsense that still gives a valid loss. It only checks that the shapes line up for cross-entropy.

**The maths, spelled out**

```
Uniform guessing over V options costs ln(V) nats of cross-entropy. ln(65) = 4.174. A model scoring 4.174 has learned nothing, and 4.495 means its random weights are slightly worse than a shrug. After training in B3 this drops to 1.674.
```

> **Watch out:** If your loss prints far below 4.1 on random data, something is leaking the answer. That is the failure the causal mask exists to prevent, and you test for it directly in step 4.

### 2. Attention is ten lines

Open `Head.forward` in my-work/labs/_shared/tinygpt.py. It is ten lines including its comments and six actual statements, and it is the whole idea of a transformer. Build one head on its own and print the shapes so you see what shrinks and what does not. The input is 128 wide, query, key and value are each 32 wide, and the score matrix is 6 by 6, one number per pair of token positions. That score matrix is the only place in the model where tokens look at each other; everything else works on one token at a time.

```python
# my-work/labs/lab20/peek.py
import sys, pathlib, torch
from torch.nn import functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from tinygpt import Config, Head

torch.manual_seed(0)
h = Head(Config(), head_size=32)
x = torch.randn(1, 6, 128)
k, q, v = h.key(x), h.query(x), h.value(x)
print("x", tuple(x.shape), "q", tuple(q.shape), "k", tuple(k.shape), "v", tuple(v.shape))

att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
print("att", tuple(att.shape))
print("row 2 before mask", [round(float(a), 2) for a in att[0, 2]])

# x (1, 6, 128) q (1, 6, 32) k (1, 6, 32) v (1, 6, 32)
# att (1, 6, 6)
# row 2 before mask [-0.16, 0.07, -0.24, -0.47, 0.09, 0.29]
```

- `k.transpose(-2, -1)`: Flips the last two axes so (1, 6, 32) becomes (1, 32, 6), which lets the matrix multiply pair every query with every key. The leading batch axis is left alone.
- `row 2 still has six numbers`: The scores for positions 3, 4 and 5 exist and are non-zero at this point. Step 4 is where they get destroyed.

**The maths, spelled out**

```
att[i][j] = (q[i] . k[j]) / sqrt(32), a plain dot product of 32 numbers divided by 5.657. Row i is how much token i cares about each token j. The matrix is T by T, so doubling the context quadruples this cost. That is why block_size is 128 and not 128,000.
```

> **Watch out:** `torch.randn(1, 6, 128)` stands in for real embeddings so you can inspect one head alone. Do not read anything into the values, only the shapes and the pattern of zeros.

### 3. Why divide by the square root of the head size

The line `att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5` has a factor most people copy without testing. Test it. A dot product of 32 roughly standard-normal numbers has a standard deviation near sqrt(32), about 5.66, and on random matrices it measured 5.14. Feed numbers that size into softmax and one token takes 99.6% of the weight while the other seven share 0.4%, so the head reads one token and the gradient through the others is close to nothing. With the scale applied the same row spreads 0.424, 0.160, 0.093 and so on, which is a preference rather than a hard pick.

```python
# my-work/labs/lab20/scale.py
import torch
from torch.nn import functional as F

torch.manual_seed(1)
q, k = torch.randn(8, 32), torch.randn(8, 32)
raw = q @ k.T
scaled = raw * 32 ** -0.5
print("std  raw", round(float(raw.std()), 2), " scaled", round(float(scaled.std()), 2))
print("raw   ", [round(float(w), 3) for w in F.softmax(raw[0], -1)])
print("scaled", [round(float(w), 3) for w in F.softmax(scaled[0], -1)])

# std  raw 5.14  scaled 0.91
# raw    [0.996, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# scaled [0.424, 0.16, 0.05, 0.082, 0.044, 0.079, 0.067, 0.093]
```

- `k.shape[-1] ** -0.5`: Reads the head size off the tensor rather than hard-coding 32, so the same line stays correct if you change n_head or n_embd in Config.
- `0.996 versus 0.424`: Softmax is exponential. A gap of 5 in the scores is a factor of 150 in the weights; a gap of 1 is a factor of 2.7.

**The maths, spelled out**

```
For independent q and k with mean 0 and variance 1, the dot product over d terms has variance d, so standard deviation sqrt(d). Multiplying by d ** -0.5 = 32 ** -0.5 = 0.1768 brings the variance back to 1. Measured: 5.14 becomes 0.91.
```

> **Watch out:** The scale is by head size (32 here), not by embedding width (128) and not by context length. Taking it from the wrong dimension is a silent bug: the model still trains, just worse.

### 4. The causal mask, and why a model that sees the future learns nothing

`tril` is registered as a buffer, not a parameter, so it moves to the right device and is saved with the model but never receives a gradient. `masked_fill` writes negative infinity into every position above the diagonal, and softmax maps negative infinity to exactly 0, so those tokens contribute nothing to the blend. Add these lines to peek.py and look at row 2: three real weights that sum to 1.000, and three hard zeros. Now think about what happens without it. The target for position 2 is the token at position 3, and position 2 can read position 3, so the model learns the identity function, the loss collapses toward zero, and at generation time (when there is no position 3 yet) it produces garbage.

```python
# append to my-work/labs/lab20/peek.py
print(h.tril[:4, :4])
masked = att.masked_fill(h.tril[:6, :6] == 0, float("-inf"))
w = F.softmax(masked, dim=-1)
print("row 2 after mask", [f"{float(a):.2f}" if a > -1e9 else "-inf" for a in masked[0, 2]])
print("weights row 2   ", [round(float(a), 3) for a in w[0, 2]], "sum", float(w[0, 2].sum()))
print("out", tuple((w @ v).shape))

# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
# row 2 after mask ['-0.16', '0.07', '-0.24', '-inf', '-inf', '-inf']
# weights row 2    [0.313, 0.396, 0.291, 0.0, 0.0, 0.0] sum 1.0
# out (1, 6, 32)
```

- `self.tril[:T, :T]`: The buffer is built at full block_size (128 by 128) but sliced to the actual sequence length, so shorter inputs work without rebuilding anything.
- `register_buffer, not nn.Parameter`: It is a fixed fact about time, not something to learn. Buffers appear in state_dict and move with .to(device), but no optimiser touches them.
- `att @ v`: Weights (1, 6, 6) times values (1, 6, 32) gives (1, 6, 32): each position's output is a weighted average of the values at and before it.

**The maths, spelled out**

```
softmax(x)[j] = exp(x[j]) / sum(exp(x)). exp(-inf) = 0 exactly, so masked positions get weight 0 and the visible ones renormalise to sum to 1. Row 2 above: exp of (-0.16, 0.07, -0.24) is (0.852, 1.073, 0.787), total 2.712, giving 0.313, 0.396, 0.291.
```

> **Watch out:** The mask goes on before softmax, never after. Zeroing the weights after softmax leaves the survivors no longer summing to 1, which quietly shrinks the signal at early positions.

### 5. Four heads, one block, and the edits that make it a stack

`Block` is attention plus a small feed-forward network, with layer norm before each and an addition around each. Each of the four heads returns 32 numbers, they concatenate back to 128, and `self.proj` mixes them. Both lines are `x = x + ...`, so the block edits the running vector rather than replacing it, which is what lets you stack four of them (or ninety-six) without destroying the signal on the way. Measure the edit sizes on the untrained model: each block adds about 13.5 units of change to a vector about 65 units long, roughly a fifth. The feed-forward is `Linear(128, 512), ReLU, Linear(512, 128)`, the standard 4x widening, and it is where most of your parameters are hiding.

```python
# my-work/labs/lab20/residual.py
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from tinygpt import Config, TinyGPT, Block

b = Block(Config())
xs = torch.randn(1, 6, 128)
outs = [head(xs) for head in b.heads]
print("one head", tuple(outs[0].shape), "-> concat", tuple(torch.cat(outs, dim=-1).shape))

torch.manual_seed(0)
m = TinyGPT(65, Config()).eval()
x = torch.randint(0, 65, (1, 16))
with torch.no_grad():
    h = m.tok(x) + m.pos(torch.arange(16))
    for i, block in enumerate(m.blocks):
        out = block(h)
        print(f"block {i}: ||x||={h.norm():.2f}  ||edit||={(out - h).norm():.2f}")
        h = out
    print("before final layer norm: mean %.3f std %.3f" % (h.mean(), h.std()))
    print("after  final layer norm: mean %.3f std %.3f" % (m.lnf(h).mean(), m.lnf(h).std()))

# one head (1, 6, 32) -> concat (1, 6, 128)
# block 0: ||x||=63.09  ||edit||=13.46
# block 1: ||x||=64.53  ||edit||=14.69
# block 2: ||x||=65.56  ||edit||=13.13
# block 3: ||x||=66.58  ||edit||=13.82
# before final layer norm: mean 0.005 std 1.523
# after  final layer norm: mean 0.000 std 1.000
```

- `h = self.ln1(x), then x = x + self.proj(...)`: The norm is applied to the copy that goes into attention, not to the copy that is added back. The residual path from input to output stays untouched, and that is what gradients travel down.
- `return x + self.ff(self.ln2(x))`: The second edit. It uses the already-edited x from the attention line, so the two edits are sequential, not parallel.
- `4 * cfg.n_embd`: The feed-forward widens to 512, applies ReLU, and comes back to 128. This is where the model does per-token thinking, as opposed to per-token mixing.

**The maths, spelled out**

```
head_size = n_embd // n_head = 128 // 4 = 32, and 4 heads of 32 concatenate back to exactly 128. If n_head does not divide n_embd the concatenation is the wrong width and the projection fails, so keep them divisible.
```

> **Watch out:** The vector growing from 63.09 to 66.58 across four blocks is normal on an untrained model. If the length doubles per block you have lost a layer norm somewhere, and training will diverge in the first hundred steps.

### 6. Take the position embedding away

`x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))` is the only thing telling the model what order the tokens arrived in. Prove it by zeroing the position table and feeding the same tokens in two different orders. On a one-layer model the final prediction moves by 5.96e-07, which is float rounding: attention is permutation-equivariant, so with no position signal the model sees an unordered bag. With the position embedding intact the same swap moves the prediction by 0.0288. Use `n_layer=1` for this test, because on the four-layer model the causal mask smuggles a little order information back in through the earlier positions, and you get a smaller but non-zero difference that muddies the point.

```python
# my-work/labs/lab20/no_position.py
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from tinygpt import Config, TinyGPT

torch.manual_seed(0)
cfg = Config(n_layer=1)
a = torch.tensor([[5, 9, 20, 7]])
b = torch.tensor([[20, 9, 5, 7]])          # same tokens, first and third swapped

m = TinyGPT(65, cfg).eval()
with torch.no_grad():
    m.pos.weight.zero_()
    print("no position:  ", float((m(a)[0][0, -1] - m(b)[0][0, -1]).abs().max()))

m2 = TinyGPT(65, cfg).eval()
with torch.no_grad():
    print("with position:", round(float((m2(a)[0][0, -1] - m2(b)[0][0, -1]).abs().max()), 4))

# no position:   5.960464477539062e-07
# with position: 0.0288
```

- `torch.arange(T)`: Positions 0 to T-1, so slot 0 always gets row 0 of the table. The model learns what 'third token in the window' means, not what 'third token in the document' means.
- `Embedding(block_size, n_embd)`: One row per slot, and only block_size rows exist. Feed 129 tokens to a model with block_size 128 and you get IndexError: index out of range in self.

**The maths, spelled out**

```
pos is an Embedding(block_size, n_embd) = 128 by 128 = 16,384 parameters, 2.0% of the model. That is the entire cost of knowing what order words come in.
```

> **Watch out:** This model learns its position vectors from scratch, so position 127 is trained on far fewer examples than position 0 and works worse. Larger models often use rotary or sinusoidal positions instead, which extend past the training length. This one does not.

### 7. Count the parameters and find where they actually live

824,897 parameters sounds like a lot until you ask where they are. Group them by module and the answer is blunt: the feed-forward networks hold 63.9% and attention holds 31.8%, so the part everyone talks about is under a third of the model. Attention is what makes a transformer a transformer, but the parameters mostly sit in the plain two-layer networks between the attention steps. Everything to do with embeddings and normalisation together is under 5%. Run this and check your total is exactly 824,897, because that number is what the mini-project checker verifies.

```python
# my-work/labs/lab20/count.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from tinygpt import Config, TinyGPT

m = TinyGPT(65, Config())
total = m.n_params()
groups = {}
for name, p in m.named_parameters():
    key = name.split(".")[0]
    if key == "blocks":
        key = "blocks." + name.split(".")[2]       # sum the same part over all 4 layers
    groups[key] = groups.get(key, 0) + p.numel()
for k, v in groups.items():
    print(f"{k:14s} {v:9,}  {100 * v / total:5.1f}%")
print(f"{'total':14s} {total:9,}")

# tok                8,320    1.0%
# pos               16,384    2.0%
# blocks.heads     196,608   23.8%
# blocks.proj       66,048    8.0%
# blocks.ff        526,848   63.9%
# blocks.ln1         1,024    0.1%
# blocks.ln2         1,024    0.1%
# lnf                  256    0.0%
# head               8,385    1.0%
# total            824,897
```

- `p.numel()`: Number of elements in the tensor. n_params() is one line: sum(p.numel() for p in self.parameters()).
- `blocks.ff at 526,848`: Each block's feed-forward is 131,712 parameters against 65,664 for all four heads plus the projection. Two thirds of every block is the plain network, not the attention.
- `head at 8,385`: The output projection back to 65 vocabulary entries. Many real models tie this to tok.weight and save the duplicate. This one does not, which is 8,320 parameters of honest waste.

**The maths, spelled out**

```
Per block: heads 4 x 3 x (128 x 32) = 49,152; proj 128 x 128 + 128 = 16,512; ff (128 x 512 + 512) + (512 x 128 + 128) = 131,712; two layer norms 4 x 128 = 512. Total 197,888, times 4 layers = 791,552. Plus tok 65 x 128 = 8,320, pos 128 x 128 = 16,384, lnf 256, output head 128 x 65 + 65 = 8,385. Grand total 824,897.
```

> **Watch out:** Parameter count scales with vocabulary through tok and head only, so swapping the 65-character vocabulary for your 556-entry BPE vocabulary from B1 adds about 125,000 parameters and nothing else changes. The checker builds with vocab_size=65, so use 65 when you compare totals.

## You are done when

You can point at any line of my-work/labs/_shared/tinygpt.py and say what it does and what breaks without it. You have seen the mask zero out the future, the scale keep softmax soft, the residuals edit rather than replace, and 63.9% of the parameters sitting in the feed-forward layers. Nothing has been trained yet, and that is the point: you now prove the model is right before spending any time on it.

---

## Mini-project: Prove the mask

Prove your model is correct without training it. Three properties, checkable in seconds, and the third is the one that matters: a model that can see the future trains beautifully and generates garbage, so catch it now rather than after eight minutes of training in B3.

- Create my-work/labs/lab20/test_model.py. Add my-work/labs/_shared to sys.path with pathlib (parents[1] / "_shared"), import Config and TinyGPT, and build model = TinyGPT(vocab_size=65, cfg=Config()).eval(). Calling .eval() matters: with dropout active the mask test would turn into noise.
- Check the parameter count: assert model.n_params() == 824_897, and print it. If your number differs you changed Config or the vocabulary size, and the checker will say so.
- Check the shapes: x = torch.randint(0, 65, (2, 16)), then logits, loss = model(x, x) inside torch.no_grad(). Assert tuple(logits.shape) == (2, 16, 65) and loss.dim() == 0. One score per vocabulary entry per position, and a loss that is a single number.
- Now the causal mask, the real test. Copy x into y, change only the last column (y[:, -1] = (y[:, -1] + 1) % 65), and run both through the model. Assert torch.equal(before[:, :-1], after[:, :-1]): every earlier output must be bit-identical, not close, identical. Then assert not torch.equal(before[:, -1], after[:, -1]): the last output must change, otherwise the model is ignoring its newest token.
- Print one short line per check and let the assertions decide the exit code. Then save the checker as my-work/labs/lab20/check_b2.py and run it from inside my-work/labs/lab20.

### Check it

`check_b2.py` is in this folder. Run it:

```bash
Save it as my-work/labs/lab20/check_b2.py, next to your test_model.py, then run `python check_b2.py` from inside my-work/labs/lab20. It needs torch, which you already have. If my-work/labs/lab20/my_gpt.py exists it checks that file instead of the shared one, so you can test your own copy.
```


**You are done when** Your test file runs clean and the checker prints eleven PASS lines. The mask check is the interesting one: earlier positions differ by exactly 0.00e+00 and the last position moves by more than a whole logit. That is a proof, not an impression, and it holds on an untrained model with random weights.

**If you want more:** Add two more checks. First the edges: T = 1 and T = 128 (the full block_size) must both work, and T = 129 must raise IndexError, because pos has only 128 rows. Second, break it on purpose: copy tinygpt.py to my-work/labs/lab20/my_gpt.py, delete the masked_fill line, and rerun the checker (it prefers my_gpt.py when that file exists). Verified: with the mask gone, earlier outputs drift by 6.19e-02 and that one check fails while everything else still passes. Delete my_gpt.py afterwards.
