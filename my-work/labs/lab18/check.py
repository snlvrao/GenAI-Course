#!/usr/bin/env python3
"""Checks the Module 18 mini-project. Run from my-work/labs/lab18/ship/:  python check.py"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOSTS = {"modal", "hugging face spaces", "cloudflare workers ai", "render"}
SECTIONS = ["## What this does", "## Install", "## Build the index",
            "## Run it", "## Example", "## Cost", "## What it cannot do"]
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

# The artefact must exist and be valid JSON before anything else can run.
p = HERE / "RELEASE.json"
if not p.exists():
    sys.exit(f"FAIL RELEASE.json not found at {p}\n      Write it, then run check.py again.")
try:
    r = json.loads(p.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    sys.exit(f"FAIL RELEASE.json is not valid JSON: {e.msg} on line {e.lineno}")

check(str(r.get("host", "")).lower() in HOSTS,
      f"host is a free tier that still exists (got {r.get('host')!r})")
check(str(r.get("public_url", "")).startswith("https://"),
      f"public_url is an https address (got {r.get('public_url')!r})")
check(r.get("index_mode") in ("sample", "user_builds"),
      f"index_mode says what ships (got {r.get('index_mode')!r})")

# The deployed copy is unwatched, so its limits must be tighter than the laptop's.
ceil, caps = r.get("deployed_usd_ceiling"), r.get("deployed_max_searches")
check(isinstance(ceil, (int, float)) and 0 < ceil < 0.25,
      f"deployed ceiling is under the laptop's $0.25 (got {ceil!r})")
check(isinstance(caps, int) and 0 < caps < 4,
      f"deployed search cap is under the laptop's 4 (got {caps!r})")
check(isinstance(r.get("cost_per_question_usd"), (int, float)),
      "cost_per_question_usd is a number you measured")

# The example answer must survive the same citation check the agent's answers do.
ex = r.get("example") or {}
labels = ex.get("labels") or {}
used = set(re.findall(r"\[(S\d+)\]", str(ex.get("answer", ""))))
check(len(str(ex.get("question", "")).strip()) > 10, "example question is a real question")
check(bool(used), "example answer carries at least one [S1]-style label")
check(not sorted(used - set(labels)),
      f"every label used is declared (missing {sorted(used - set(labels))})")
check(bool(labels) and all(isinstance(v, dict) and v.get("doc") for v in labels.values()),
      "every declared label names the document behind it")

cannot = r.get("cannot_do") or []
check(isinstance(cannot, list) and len(cannot) >= 3 and all(len(str(c)) >= 15 for c in cannot),
      f"cannot_do lists 3 or more real limits (got {len(cannot)})")

st = r.get("stranger_test") or {}
check(isinstance(st.get("minutes"), (int, float)) and 0 < st["minutes"] <= 10,
      f"a stranger got a cited answer within 10 minutes (got {st.get('minutes')!r})")
check(isinstance(st.get("readme_fixes"), list),
      "stranger_test.readme_fixes lists what you had to fix")

# README sections, present and in the order a stranger reads them.
readme = HERE / "README.md"
text = readme.read_text(encoding="utf-8") if readme.exists() else ""
check(bool(text), "README.md exists next to RELEASE.json")
found = [text.find(s) for s in SECTIONS]
for s, i in zip(SECTIONS, found):
    check(i >= 0, f"README has a {s} section")
check(all(a >= 0 for a in found) and found == sorted(found),
      "README sections appear in reading order")

# Limits must live in environment variables, and no key may sit in the folder.
src = (HERE / "config.py").read_text(encoding="utf-8") if (HERE / "config.py").exists() else ""
check("USD_CEILING" in src and ("environ" in src or "getenv" in src),
      "config.py reads the ceiling from an environment variable")
check(not (HERE / ".env").exists(), "no .env file sits in the shipped folder")
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,})")
leaks = sorted(f.name for f in HERE.rglob("*") if f.is_file()
               and f.suffix in (".py", ".md", ".json", ".yaml", ".toml", ".txt", ".example")
               and SECRET.search(f.read_text(encoding="utf-8", errors="ignore")))
check(not leaks, f"no API-key-shaped string in the shipped files (found in {leaks})")

print("\nnot checked automatically: whether the URL is live, whether the README actually\n"
      "reads well, and whether the host enforces the ceiling. Open the URL and ask your\n"
      "example question to settle those three yourself.")
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
