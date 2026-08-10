"""
Your model's parameters, chosen by you.

Nothing in the Build track ships with a size baked in. You write
model_config.json, this file reads it, checks the choices make sense together,
and tells you what they cost before you spend eight minutes training.

    python myconfig.py            show the config and what it costs
    python myconfig.py --new      write a starter model_config.json to edit

Every field is a real decision with a consequence, and this prints the
consequence. There is no "correct" answer, but there are wrong combinations,
and it will refuse those rather than let you find out after training.
"""

from __future__ import annotations

import json
import pathlib
import sys

TEMPLATE = {
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "block_size": 128,
    "learning_rate": 3e-3,
    "batch_size": 32,
    "why": {
        "n_layer": "REPLACE THIS with your reason",
        "n_head": "REPLACE THIS with your reason",
        "n_embd": "REPLACE THIS with your reason",
        "block_size": "REPLACE THIS with your reason",
        "learning_rate": "REPLACE THIS with your reason",
        "batch_size": "REPLACE THIS with your reason",
    },
}

FIELDS = ["n_layer", "n_head", "n_embd", "block_size", "learning_rate", "batch_size"]

# What each dial actually does, printed next to your choice.
MEANING = {
    "n_layer": "how many times the model gets to revise its understanding",
    "n_head": "how many things it can pay attention to at once",
    "n_embd": "how much room each token has to carry meaning",
    "block_size": "how far back it can see, in tokens",
    "learning_rate": "how big a step it takes downhill each time",
    "batch_size": "how many chunks of text it looks at per step",
}

LIMITS = {
    "n_layer": (1, 12), "n_head": (1, 16), "n_embd": (16, 512),
    "block_size": (16, 512), "learning_rate": (1e-5, 1e-1), "batch_size": (1, 128),
}


def path_for(start: pathlib.Path | None = None) -> pathlib.Path:
    """model_config.json, looked for beside the script then one folder up."""
    here = (start or pathlib.Path.cwd()).resolve()
    for folder in (here, here.parent):
        p = folder / "model_config.json"
        if p.exists():
            return p
    return here / "model_config.json"


def write_template(p: pathlib.Path) -> None:
    p.write_text(json.dumps(TEMPLATE, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    print("Open it, change the numbers, and replace every 'why' line with your reason.")


def load(p: pathlib.Path | None = None) -> dict:
    """Read the config and refuse the combinations that cannot work."""
    p = p or path_for()
    if not p.exists():
        raise SystemExit(
            f"No model_config.json found (looked in {p.parent}).\n"
            "Run:  python myconfig.py --new\n"
            "then edit it. The Build track will not pick your model size for you.")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{p.name} is not valid JSON: {e.msg} on line {e.lineno}")

    missing = [f for f in FIELDS if f not in cfg]
    if missing:
        raise SystemExit(f"{p.name} is missing: {', '.join(missing)}")

    for f in FIELDS:
        lo, hi = LIMITS[f]
        v = cfg[f]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise SystemExit(f"{f} must be a number, got {v!r}")
        if not (lo <= v <= hi):
            raise SystemExit(f"{f} is {v}, which is outside {lo} to {hi}. "
                             "Those bounds are what a laptop processor can finish.")

    # The one combination that is genuinely illegal rather than just unwise:
    # each head takes an equal slice of the embedding width.
    if cfg["n_embd"] % cfg["n_head"] != 0:
        raise SystemExit(
            f"n_embd ({cfg['n_embd']}) must divide evenly by n_head ({cfg['n_head']}), "
            f"because the heads split the width between them.\n"
            f"Nearest that works: {cfg['n_head'] * round(cfg['n_embd'] / cfg['n_head'])}")
    return cfg


def n_params(cfg: dict, vocab_size: int) -> dict:
    """Exact parameter count for TinyGPT, broken down by where they live.

    Verified against the real model: vocab 65 with the template config gives
    824,897, which is what torch reports.
    """
    e, L = cfg["n_embd"], cfg["n_layer"]
    tok = vocab_size * e
    pos = cfg["block_size"] * e
    attn = 3 * e * e                     # query, key and value, no bias
    proj = e * e + e
    ff = (e * 4 * e + 4 * e) + (4 * e * e + e)
    norms = 4 * e                        # two layer norms, weight and bias
    per_block = attn + proj + ff + norms
    final = 2 * e + (e * vocab_size + vocab_size)
    return {
        "token embedding": tok,
        "position embedding": pos,
        f"{L} blocks": per_block * L,
        "output layer": final,
        "total": tok + pos + per_block * L + final,
    }


def report(cfg: dict, vocab_size: int = 65) -> None:
    print("Your choices\n")
    for f in FIELDS:
        v = cfg[f]
        shown = f"{v:g}"
        why = (cfg.get("why") or {}).get(f, "")
        flag = "  <- still the template reason" if "REPLACE THIS" in str(why) else ""
        print(f"  {f:<14} {shown:>8}   {MEANING[f]}")
        if why:
            print(f"                 \"{why}\"{flag}")

    parts = n_params(cfg, vocab_size)
    total = parts.pop("total")
    print(f"\nWhat that costs, for a {vocab_size} character vocabulary\n")
    for k, v in parts.items():
        print(f"  {k:<22} {v:>12,}  ({100 * v / total:.1f}%)")
    print(f"  {'TOTAL':<22} {total:>12,} parameters")
    print(f"\n  memory for the weights   {total * 4 / 1e6:.1f} MB at 4 bytes each")
    print(f"  and again for the optimiser, so about {total * 12 / 1e6:.1f} MB while training")

    # Rough guide from the measured run: 824,897 parameters managed about
    # 5 steps a second on a laptop processor with no graphics card.
    rate = 5.0 * (824_897 / max(total, 1)) * (128 / max(cfg["block_size"], 1)) * (32 / max(cfg["batch_size"], 1))
    print(f"\n  expect roughly {rate:.1f} training steps per second on a laptop processor")
    print(f"  so an 8 minute run gets you about {int(rate * 480):,} steps")
    if rate * 480 < 800:
        print("  WARNING: that is too few steps to learn much. Shrink the model or the batch.")


if __name__ == "__main__":
    p = path_for()
    if "--new" in sys.argv:
        if p.exists() and "--force" not in sys.argv:
            print(f"{p} already exists. Add --force to overwrite it.")
        else:
            write_template(p)
        raise SystemExit(0)
    report(load(p), int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 65)
