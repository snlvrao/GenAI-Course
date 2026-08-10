"""
Checks that the model you trained is one YOU designed.

Run it from my-work/labs/lab20 or my-work/labs/lab21:   python ../_shared/check_decisions.py

It does not care which numbers you picked. It cares that you picked them, wrote
down why, and that the model you actually trained matches what you wrote.
"""

import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
import myconfig  # noqa: E402

TEMPLATE = myconfig.TEMPLATE
FIELDS = myconfig.FIELDS
fails = []


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def main() -> int:
    p = myconfig.path_for()
    if not p.exists():
        print(f"FAIL no model_config.json (looked in {p.parent} and its parent).")
        print("     Run: python ../_shared/myconfig.py --new")
        return 1
    print(f"reading {p}\n")

    try:
        cfg = myconfig.load(p)
    except SystemExit as e:
        print("FAIL " + str(e).replace("\n", "\n     "))
        return 1
    check(True, "model_config.json is valid and the numbers work together")

    # 1. Did you actually change anything, or is this still my template?
    changed = [f for f in FIELDS if cfg[f] != TEMPLATE[f]]
    check(len(changed) >= 2,
          f"at least 2 of the 6 numbers differ from the template "
          f"(you changed {len(changed)}: {', '.join(changed) or 'none'})")

    # 2. Reasons. A number without a reason is a guess, not a decision.
    why = cfg.get("why") or {}
    left = [f for f in FIELDS if "REPLACE THIS" in str(why.get(f, ""))]
    check(not left, f"every 'why' is your own words (still template: {', '.join(left) or 'none'})")
    thin = [f for f in FIELDS if len(str(why.get(f, "")).strip()) < 20]
    check(not thin, f"every reason is a real sentence (too short: {', '.join(thin) or 'none'})")

    # 3. Does the arithmetic you were shown match the model you would build?
    ckpt = pathlib.Path("tinygpt.pt")
    if not ckpt.exists():
        print("NOTE  no tinygpt.pt here, so the trained model was not compared "
              "against this config. Run this again from my-work/labs/lab21 after training.")
    else:
        import torch
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        saved = blob.get("my_config")
        if saved is None:
            check(False, "tinygpt.pt was saved before config support, retrain it")
        else:
            same = [f for f in FIELDS if saved.get(f) != cfg[f]]
            check(not same,
                  "the checkpoint was trained with THIS config "
                  f"(differs on: {', '.join(same) or 'nothing'})")
        vocab = len(blob.get("stoi") or {})
        want = myconfig.n_params(cfg, vocab)["total"]
        got = blob.get("n_params")
        if got is None:
            check(False, "tinygpt.pt has no parameter count, retrain it")
        else:
            check(want == got,
                  f"the model really has the {want:,} parameters your config predicts "
                  f"(checkpoint says {got:,})")
        loss = blob.get("val_loss")
        import math
        base = math.log(max(vocab, 2))
        check(isinstance(loss, (int, float)) and loss < base * 0.75,
              f"it learned: validation loss {loss:.3f} against {base:.3f} for random guessing")

    print()
    if fails:
        print(f"{len(fails)} CHECKS FAILED")
        return 1
    print("ALL CHECKS PASSED")
    print("NOT CHECKED: whether your reasons are good ones. Nothing can check that. "
          "Train a second config and see whether the reasoning held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
