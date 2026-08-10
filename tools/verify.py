"""End-to-end verification of the built course. Run after generating pages."""
import ast, io, json, os, re, subprocess, sys, urllib.parse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fails, warns = [], []


def fail(m): fails.append(m); print("  FAIL  " + m)
def warn(m): warns.append(m); print("  warn  " + m)
def ok(m):   print("  ok    " + m)


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


print("\n=== 1. index and structure ===")
src = read(os.path.join(ROOT, "assets", "data.js"))
INDEX = json.loads(src[src.index("=") + 1:].rstrip().rstrip(";"))
mods = INDEX["modules"]
ok(f"data.js parses, {len(mods)} modules, {len(INDEX['phases'])} phases")

for m in mods:
    p = os.path.join(ROOT, "modules", m["file"])
    if not os.path.exists(p):
        fail(f"module {m['n']}: missing page {m['file']}")
if not fails:
    ok(f"all {len(mods)} module pages exist")

for f in ["index.html", "setup.html", "README.md", ".env.example", ".gitignore",
          "assets/course.css", "assets/course.js", "assets/widgets.js", "assets/data.js",
          "my-work/labs/_shared/llm.py", "my-work/labs/requirements.txt"]:
    if not os.path.exists(os.path.join(ROOT, f)):
        fail(f"missing file: {f}")


print("\n=== 2. no AI attribution anywhere ===")
BAD = ["generated with claude", "co-authored-by", "as an ai", "language model, i",
       "anthropic.com/claude-code", "🤖", "this brief", "the user asked",
       "chatgpt", "openai assistant"]
hits = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames
                   if d not in (".venv", "__pycache__", ".git", "tools")]
    for fn in filenames:
        if not fn.endswith((".html", ".js", ".css", ".md", ".py", ".txt", ".example")):
            continue
        full = os.path.join(dirpath, fn)
        low = read(full).lower()
        for b in BAD:
            if b in low:
                rel = os.path.relpath(full, ROOT)
                # "chatgpt" legitimately appears in a video title
                if b == "chatgpt" and rel.replace("\\", "/") in ("assets/data.js",):
                    continue
                fail(f"attribution/leak '{b}' in {rel}")
                hits += 1
if not hits:
    ok("clean - no attribution, no brief leakage")


print("\n=== 3. internal links resolve ===")
bad_links = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in (".venv", "__pycache__", ".git", "tools")]
    for fn in filenames:
        if not fn.endswith(".html"):
            continue
        full = os.path.join(dirpath, fn)
        html = read(full)
        for href in re.findall(r'(?:href|src)="([^"]+)"', html):
            if href.startswith(("http", "data:", "#", "mailto:")):
                continue
            target = os.path.normpath(os.path.join(dirpath, urllib.parse.unquote(href.split("#")[0])))
            if not os.path.exists(target):
                fail(f"broken link in {os.path.relpath(full, ROOT)} -> {href}")
                bad_links += 1
if not bad_links:
    ok("every relative link and asset reference resolves")


print("\n=== 4. no fetch() - required for file:// to work ===")
for f in ["assets/course.js", "assets/widgets.js"]:
    js = read(os.path.join(ROOT, f))
    if re.search(r"\bfetch\s*\(", js) or "XMLHttpRequest" in js:
        fail(f"{f} uses fetch/XHR - this breaks when opened from disk")
    else:
        ok(f"{f} has no network calls")

print("\n=== 5. page structure ===")
need = {"data-chrome=\"masthead\"": "header", "data-chrome=\"sidebar\"": "sidebar",
        "data-chrome=\"modhead\"": "module header", "data-quiz": "quiz",
        "data-notes": "notes box", 'data-mark="read"': "read mark",
        'data-mark="lab"': "lab mark", 'data-mark="mini"': "mini mark",
        "data-videos=\"core\"": "core videos", "data-chrome=\"pager\"": "prev/next"}
for m in mods:
    html = read(os.path.join(ROOT, "modules", m["file"]))
    for k, label in need.items():
        if k not in html:
            fail(f"module {m['n']} missing {label}")
    nq = html.count('class="q"')
    if nq != 3:
        warn(f"module {m['n']} has {nq} quiz questions, expected 3")
    ncon = html.count('class="concept"')
    if ncon < 5:
        warn(f"module {m['n']} has only {ncon} concept cards")
ok("structural checks done")


print("\n=== 6. widgets referenced actually exist ===")
wjs = read(os.path.join(ROOT, "assets", "widgets.js"))
defined = set(re.findall(r"W\.(\w+)\s*=\s*function", wjs))
used = set()
for m in mods:
    html = read(os.path.join(ROOT, "modules", m["file"]))
    used |= set(re.findall(r'data-widget="(\w+)"', html))
missing = used - defined
if missing:
    fail(f"pages reference undefined widgets: {sorted(missing)}")
else:
    ok(f"{len(defined)} widgets defined, {len(used)} used, all resolve")
unused = defined - used
if unused:
    warn(f"widgets defined but never placed on a page: {sorted(unused)}")

# Read gen_pages.py's caption table without importing it (importing would run
# the generator). A widget placed with no caption used to emit an empty
# paragraph, so this catches the drift at source.
gp = ast.parse(read(os.path.join(ROOT, "tools", "gen_pages.py")))
captioned = set()
for node in ast.walk(gp):
    if (isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "WIDGET_TITLE" for t in node.targets)):
        captioned = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
missing_caption = used - captioned
if missing_caption:
    warn(f"widgets placed with no caption in WIDGET_TITLE: {sorted(missing_caption)}")


print("\n=== 7. every video is still embeddable (live oembed check) ===")
seen, checked, dead, mismatched = set(), 0, 0, 0
for m in mods:
    for v in m["videos"]:
        if v["id"] in seen:
            continue
        seen.add(v["id"])
        url = ("https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v="
               + v["id"] + "&format=json")
        try:
            out = subprocess.run(["curl", "-s", "-m", "25", url],
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace").stdout
            d = json.loads(out)
        except Exception:
            fail(f"video {v['id']} ({v['t'][:40]}) is NOT embeddable")
            dead += 1
            continue
        checked += 1
        if d.get("author_name", "").strip() != v["c"].strip():
            warn(f"channel changed for {v['id']}: page says '{v['c']}', "
                 f"YouTube says '{d.get('author_name')}'")
            mismatched += 1
print(f"  checked {checked} unique videos: {dead} not embeddable, {mismatched} metadata drift")
if not dead:
    ok("every video still embeds")


print("\n=== 8. text encoding ===")
# Every text file must be strict UTF-8 and free of U+FFFD. A file read with
# errors="replace" turns a mis-decoded byte into U+FFFD permanently, so a stray
# replacement character means real characters were lost upstream, not just
# displayed oddly. read() above is deliberately lenient, so check the bytes.
enc_bad = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames
                   if d not in (".venv", "__pycache__", ".git")]
    for fn in filenames:
        if not fn.endswith((".html", ".js", ".css", ".md", ".py", ".json",
                            ".tsv", ".txt", ".example")):
            continue
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, ROOT)
        raw = io.open(full, "rb").read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            fail(f"{rel}: not valid UTF-8 ({e})")
            enc_bad += 1
            continue
        # chr(), not a literal: a literal one here would flag this file itself
        if chr(0xFFFD) in text:
            fail(f"{rel}: {text.count(chr(0xFFFD))} replacement characters "
                 f"(U+FFFD) - characters were lost when this file was written")
            enc_bad += 1
        # Raw control characters make editors, diffs and grep treat a text file
        # as binary. In source they almost always mean an escape sequence such
        # as \x00 was written out as the character it names.
        ctrl = sorted({ord(c) for c in text
                       if ord(c) < 32 and c not in "\t\n\r"} | {ord(c) for c in text if ord(c) == 127})
        if ctrl:
            fail(f"{rel}: raw control characters {[hex(c) for c in ctrl]} "
                 f"- write these as escapes, not as the characters themselves")
            enc_bad += 1
if not enc_bad:
    ok("every text file is strict UTF-8 with no lost characters")


print("\n=== 9. no em dashes in authored content ===")
# A house rule, so it is checked rather than remembered. The only allowed
# instances are inside YouTube titles, which are other people's words quoted
# exactly and must not be edited to suit us.
# Both spellings. The entity renders identically and is how ~750 of them hid
# from a character count the first time this was swept. Assembled rather than
# written out, so this file does not report itself.
_A = "&"
EM_FORMS = (chr(0x2014), _A + "mdash;", _A + "#8212;", _A + "#x2014;")
QUOTED = {"assets/data.js", "tools/videos.tsv"}
em_bad = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in (".venv", "__pycache__", ".git")]
    for fn in filenames:
        if not fn.endswith((".html", ".js", ".css", ".md", ".py", ".json", ".txt")):
            continue
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, ROOT).replace("\\", "/")
        if rel in QUOTED:
            continue
        body = read(full)
        n = sum(body.count(f) for f in EM_FORMS)
        if n:
            fail(f"{rel}: {n} em dash(es). Use a comma, a full stop or brackets")
            em_bad += n
if not em_bad:
    ok("none outside the quoted video titles")


print("\n" + "=" * 60)
print(f"FAILURES: {len(fails)}   warnings: {len(warns)}")
if fails:
    print("\nMust fix:")
    for f in fails:
        print("  - " + f)
sys.exit(1 if fails else 0)
