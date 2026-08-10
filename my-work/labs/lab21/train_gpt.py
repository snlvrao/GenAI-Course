"""
Build track B3 - train your own GPT from nothing.

No pretrained weights. The model starts as random numbers and learns to write
by predicting the next character, several million times.

    python train_gpt.py                 trains on the sample corpus
    python train_gpt.py mytext.txt      trains on your own file
    python train_gpt.py mytext.txt 900  trains for 900 seconds instead of 480

On a laptop processor with no graphics card, the default 8 minute budget gets
roughly 3,600 training steps and a validation loss near 1.55, against 4.17 for
random guessing. That is enough to produce recognisable English: real words,
line breaks in the right places, and speaker names if your text has them.

It writes tinygpt.pt, which B5 serves behind an API.
"""

import math
import os
import pathlib
import sys
import time

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
import myconfig  # noqa: E402
from tinygpt import Config, TinyGPT  # noqa: E402

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
BUDGET_S = int(sys.argv[2]) if len(sys.argv) > 2 else 480
OUT = "tinygpt.pt"

# Your model's size and learning rate come from model_config.json, which YOU
# write. There is no default baked in here on purpose: picking these numbers
# is the decision this module is about. Run `python ../_shared/myconfig.py --new`
# to get a file to edit, then `python ../_shared/myconfig.py` to see what your
# choices will cost before you spend the time.
MY = myconfig.load()
BATCH, LR = int(MY["batch_size"]), float(MY["learning_rate"])

torch.manual_seed(1337)


def load_corpus(path: str) -> str:
    """Your own text if you have it, a public domain sample if you do not."""
    if os.path.exists(path):
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    print(f"No {path} here. Fetching a small public domain sample instead.")
    print("Point this script at your own .txt file to train on your own writing.")
    import urllib.request
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
           "data/tinyshakespeare/input.txt")
    urllib.request.urlretrieve(url, path)
    return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")


text = load_corpus(CORPUS)
if len(text) < 20_000:
    sys.exit(f"FAIL {CORPUS} is only {len(text):,} characters. "
             "Use at least 20,000 or the model has nothing to learn from.")

# The vocabulary is every distinct character in YOUR text. Nothing is
# pretrained, not even the alphabet.
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

cfg = Config(block_size=int(MY["block_size"]), n_embd=int(MY["n_embd"]),
             n_head=int(MY["n_head"]), n_layer=int(MY["n_layer"]))
model = TinyGPT(len(chars), cfg)
print(f"corpus     {len(text):,} characters, {len(chars)} distinct")
print(f"model      {model.n_params():,} parameters, from YOUR model_config.json")
print(f"           {cfg.n_layer} layers, {cfg.n_head} heads, width {cfg.n_embd}, "
      f"context {cfg.block_size}, lr {LR:g}, batch {BATCH}")
print(f"baseline   loss {math.log(len(chars)):.3f} if it only ever guessed at random")


def get_batch(which: str):
    d = train_data if which == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (BATCH,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def val_loss() -> float:
    """Loss on text it never trained on. This is the number that matters."""
    model.eval()
    out = torch.stack([model(*get_batch("val"))[1] for _ in range(20)]).mean().item()
    model.train()
    return out


opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"\ntraining for {BUDGET_S}s. Stop it early with Ctrl+C and it still saves.\n")

history, step, t0 = [], 0, time.time()
try:
    while time.time() - t0 < BUDGET_S:
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()        # which way should every number move
        opt.step()             # move them a little
        step += 1
        if step % 200 == 0:
            v = val_loss()
            el = time.time() - t0
            history.append({"step": step, "val_loss": round(v, 4),
                            "seconds": round(el, 1)})
            print(f"  step {step:>5}  train {loss.item():.3f}  val {v:.3f}"
                  f"   ({el:.0f}s, {step / el:.1f} steps/s)")
except KeyboardInterrupt:
    print("\nstopped early, saving what you have")

final = val_loss()
print(f"\n{step} steps in {time.time() - t0:.0f}s. Final validation loss {final:.3f} "
      f"(random guessing would be {math.log(len(chars)):.3f}).")

torch.save({"state": model.state_dict(), "stoi": stoi, "itos": itos,
            "config": cfg, "val_loss": final, "steps": step,
            "history": history, "corpus_chars": len(text),
            "my_config": MY, "n_params": model.n_params()}, OUT)
print(f"saved {OUT}  ({os.path.getsize(OUT):,} bytes)")

start = torch.zeros((1, 1), dtype=torch.long)
sample = "".join(itos[i] for i in model.generate(start, 600)[0].tolist())
print("\n--- 600 characters your model invented ---")
print(sample)
pathlib.Path("sample.txt").write_text(sample, encoding="utf-8")
print("\n(also written to sample.txt)")
