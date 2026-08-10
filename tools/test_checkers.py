"""
Run every mini-project checker with NO learner work present.

A good checker must then: exit non-zero, and say plainly what is missing.
A bad checker throws a traceback, which teaches the learner nothing.
"""
import io, json, os, subprocess, sys, tempfile, shutil

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The interpreter running this file, rather than a guessed path. A hardcoded
# .venv location breaks the moment the environment moves, and a bare "python"
# picks up whichever one is on PATH, which is usually the wrong one.
PY = sys.executable

C = json.load(io.open(os.path.join(SP, "module_content.json"), encoding="utf-8"))

good = bad = skipped = 0
for c in sorted(C, key=lambda x: x["module"]):
    n = c["module"]
    chk = c["mini"].get("checker") or {}
    code = (chk.get("code") or "").strip()
    if not code:
        print(f"  m{n:02d}  NO CHECKER")
        skipped += 1
        continue

    tmp = tempfile.mkdtemp(prefix=f"chk{n}_")
    try:
        fn = os.path.basename(chk.get("filename") or "check.py")
        p = os.path.join(tmp, fn)
        io.open(p, "w", encoding="utf-8").write(code + "\n")

        # 1. does it even compile?
        r = subprocess.run([PY, "-m", "py_compile", p], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"  m{n:02d}  SYNTAX ERROR: {(r.stderr or '').strip().splitlines()[-1][:90]}")
            bad += 1
            continue

        # 2. run it in an empty folder: the learner has done nothing yet
        r = subprocess.run([PY, p], cwd=tmp, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        traceback = "Traceback (most recent call last)" in out
        exits_nonzero = r.returncode != 0

        if traceback:
            last = [l for l in out.splitlines() if l.strip()][-1][:80]
            print(f"  m{n:02d}  TRACEBACK on missing work: {last}")
            bad += 1
        elif not exits_nonzero:
            print(f"  m{n:02d}  EXIT 0 with no work present (should be non-zero)")
            bad += 1
        else:
            first = next((l for l in out.splitlines() if l.strip()), "")[:74]
            print(f"  m{n:02d}  ok   exit {r.returncode}  \"{first}\"")
            good += 1
    except subprocess.TimeoutExpired:
        print(f"  m{n:02d}  TIMEOUT")
        bad += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

print(f"\nclean failure on missing work: {good}   problems: {bad}   no checker: {skipped}")
sys.exit(1 if bad else 0)
