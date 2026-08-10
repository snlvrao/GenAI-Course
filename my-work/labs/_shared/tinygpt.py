"""
TinyGPT - a working transformer, written out rather than imported.

This is the model you build in Build-track module B2 and train in B3. Every
piece of it appears in Module 4 as a diagram. There is no magic in here and
no library doing the interesting part: attention is eleven lines.

Sizes are deliberately small so it trains on a laptop processor in minutes.
Nothing about the shape changes when you scale it up, which is the point.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    """Everything that decides how big your model is."""
    block_size: int = 128     # how many tokens of context it can see at once
    n_embd: int = 128         # width of the vector carried per token
    n_head: int = 4           # how many attention heads per block
    n_layer: int = 4          # how many blocks stacked
    dropout: float = 0.0


class Head(nn.Module):
    """One attention head. This is the whole idea of a transformer."""

    def __init__(self, cfg: Config, head_size: int):
        super().__init__()
        # Three different views of the same token. Query is "what am I looking
        # for", key is "what do I offer", value is "what I will hand over".
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # A lower-triangular matrix of ones. This is the causal mask: it is
        # what stops position 5 from reading position 6 and cheating.
        self.register_buffer("tril", torch.tril(
            torch.ones(cfg.block_size, cfg.block_size)))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        # How much each token cares about each earlier token. The divide by
        # the square root of the head size keeps the numbers from getting so
        # large that softmax turns into a hard pick of one token.
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        return att @ v            # a weighted blend of the earlier values


class Block(nn.Module):
    """Attention, then a small feed-forward network. Repeat n_layer times."""

    def __init__(self, cfg: Config):
        super().__init__()
        hs = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, hs) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ff = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout),
        )
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        # x = x + something, twice. Those additions are the residual
        # connections: each block EDITS the running vector rather than
        # replacing it, which is why very deep stacks still train.
        h = self.ln1(x)
        x = x + self.proj(torch.cat([head(h) for head in self.heads], dim=-1))
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    """The whole model: embed, stack of blocks, project back to vocabulary."""

    def __init__(self, vocab_size: int, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.block_size = self.cfg.block_size
        self.tok = nn.Embedding(vocab_size, self.cfg.n_embd)
        # A transformer has no built-in sense of order, so position is a
        # second thing you look up and add on. Remove this line and the model
        # sees your text as a bag of tokens.
        self.pos = nn.Embedding(self.cfg.block_size, self.cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(self.cfg) for _ in range(self.cfg.n_layer)])
        self.lnf = nn.LayerNorm(self.cfg.n_embd)
        self.head = nn.Linear(self.cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.lnf(self.blocks(x))
        logits = self.head(x)                 # one score per vocabulary entry
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        """Predict the next token, append it, repeat. That is all generation is."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])   # only the last block fits
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # A shape check you can run before training anything.
    cfg = Config()
    m = TinyGPT(vocab_size=65, cfg=cfg)
    x = torch.randint(0, 65, (2, 16))
    logits, loss = m(x, x)
    print(f"parameters      {m.n_params():,}")
    print(f"input  {tuple(x.shape)}  ->  logits {tuple(logits.shape)}")
    print(f"loss on random data {loss.item():.3f} "
          f"(a fair coin over 65 options would be {torch.log(torch.tensor(65.0)):.3f})")
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), 20)
    print(f"generated {out.shape[1] - 1} tokens from an empty prompt")
