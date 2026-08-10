"""
Reconcile authored module content against the code that was actually RUN.

The module authors wrote plausible code. Some of it does not work against the
real installed SDKs, and some of it invented file paths and environment
variables that this repo does not use. Everything replaced here was verified by
execution, not by reading.
"""
import io, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SP)

SRC = os.path.join(SP, "module_content_raw.json")
DST = os.path.join(SP, "module_content.json")

content = json.load(io.open(SRC, encoding="utf-8"))
by = {c["module"]: c for c in content}
changes = []


# --- 1. verified replacement for the Module 13 MCP client -------------------
# The authored version used the low-level ClientSession(read, write) pattern.
# Running it against mcp 2.0.0 fails with "Invalid request parameters", because
# the low-level session does not attach the _meta block (protocol version and
# client capabilities) that the 2026-07-28 spec now REQUIRES on every request.
# The high-level Client does. This version was run and works.
LAB13 = os.path.join(ROOT, "my-work", "labs", "lab13")

# Read the client straight off disk, so the code shown on the page is byte for
# byte the code that was executed. They cannot drift apart.
MCP_CLIENT = io.open(os.path.join(LAB13, "mcp_client.py"), encoding="utf-8").read()
HELLO_SERVER = io.open(os.path.join(LAB13, "hello_server.py"), encoding="utf-8").read()

_UNUSED = '''# my-work/labs/lab13/probe.py
# Your own MCP client. This is what makes the lab vendor-neutral: no Claude
# Desktop, no Cursor, no VS Code, no API key, no internet.
import asyncio
import json
import sys

from mcp import Client, StdioServerParameters, stdio_client
from mcp.types import TextContent

import server_hello


def text_of(result) -> str:
    """Flatten a tool result into something printable."""
    if getattr(result, "structured_content", None) is not None:
        return json.dumps(result.structured_content)
    return "\\n".join(b.text for b in result.content if isinstance(b, TextContent))


async def explore(client, label):
    print(f"\\n--- {label} ---")
    print("server:  ", client.server_info.name, client.server_info.version)
    print("protocol:", client.protocol_version)     # prints 2026-07-28

    tools = await client.list_tools()
    print("tools:   ", ", ".join(t.name for t in tools.tools))

    result = await client.call_tool("add", {"a": 2, "b": 40})
    print("add(2,40) ->", text_of(result))

    res = await client.read_resource("greeting://you")
    print("resource ->", res.contents[0].text)


async def main():
    # Way 1: in memory. No subprocess at all. Fastest while you are still
    # writing the server, and what you would use in a unit test.
    async with Client(server_hello.mcp) as client:
        await explore(client, "in-memory")

    # Way 2: over stdio, exactly as a real host app would run it.
    #
    # sys.executable, NOT the word "python". The bare word resolves against
    # your PATH, which is usually not your virtual environment, so the
    # subprocess starts, cannot import mcp, and dies. All you would see is
    # "Connection closed", which tells you nothing.
    params = StdioServerParameters(command=sys.executable, args=["server_hello.py"])
    async with Client(stdio_client(params)) as client:
        await explore(client, "stdio subprocess")


if __name__ == "__main__":
    asyncio.run(main())
'''

if 13 in by:
    for st in by[13]["lab"]["steps"]:
        code = st.get("code") or ""
        if "ClientSession" in code:
            st["code"] = MCP_CLIENT
            st["body"] = (
                "Save this as <code>my-work/labs/lab13/mcp_client.py</code> and run "
                "<code>python mcp_client.py</code>. There is no "
                "<code>session.initialize()</code> call anywhere, and that is not an oversight: "
                "the 2026-07-28 protocol removed the handshake. Use the high-level "
                "<code>Client</code> class rather than the lower-level "
                "<code>ClientSession</code>. <code>Client</code> attaches the <code>_meta</code> "
                "block that every request now requires; the low-level session does not, and you "
                "get a confusing <code>Invalid request parameters</code> error instead. You "
                "should see the server answer twice, once in memory and once as a real subprocess."
            )
            changes.append("m13: replaced ClientSession client with the verified Client version")
        elif "@mcp.tool()" in code and "hello" in code.lower() and "sqlite" not in code:
            st["code"] = HELLO_SERVER
            changes.append("m13: hello server code now read from the tested file")

    # streamablehttp_client does not exist in SDK v2. Verified by running it:
    # the import raises ImportError. The name is streamable_http_client, and
    # for a plain URL you do not need the transport at all.
    HTTP_CLIENT = io.open(os.path.join(LAB13, "http_client.py"), encoding="utf-8").read()
    for st in by[13]["lab"]["steps"]:
        code = st.get("code") or ""
        if "streamablehttp_client" in code:
            head = code.split("# ---------------------------------------------------------------------------")[0]
            st["code"] = head.rstrip() + (
                "\n\n\n# ---------------------------------------------------------------------------\n"
                + HTTP_CLIENT
            )
            changes.append("m13: fixed streamablehttp_client -> Client(URL), verified by running it")

    # The doc-search server guessed at Module 10's table names. Use the real ones.
    for st in by[13]["lab"]["steps"]:
        if st.get("code") and "chunk_vec" in st["code"]:
            st["code"] = (st["code"].replace("chunk_vec", "vec_chunks")
                                    .replace("c.source", "c.doc")
                                    .replace('"source": r[0]', '"source": r[0]'))
            changes.append("m13: doc-search SQL now matches Module 10's actual table names")


# --- 2. repo paths -----------------------------------------------------------
# Authors guessed at folder names. The real layout is my-work/labs/labNN/.
PATH_FIXES = [
    (re.compile(r"labs[/\\]module(\d\d)"), r"my-work/labs/lab\1"),
    (re.compile(r"labs[/\\]m(\d\d)(?![a-z0-9])"), r"my-work/labs/lab\1"),
]
for c in content:
    blob = json.dumps(c, ensure_ascii=False)
    fixed = blob
    for rx, rep in PATH_FIXES:
        fixed = rx.sub(rep, fixed)
    if fixed != blob:
        new = json.loads(fixed)
        c.clear()
        c.update(new)
        changes.append(f"m{c['module']:02d}: corrected lab folder paths to my-work/labs/labNN/")


# --- 3. invented environment variables --------------------------------------
# llm.py reads LLM_PROVIDER, LLM_MODEL and <PROVIDER>_API_KEY. Nothing else.
GHOST_VARS = ["OLLAMA_BASE_URL", "OLLAMA_MODEL", "OPENAI_BASE_URL", "OPENAI_MODEL",
              "GROQ_BASE_URL", "GROQ_MODEL", "GEMINI_BASE_URL", "GEMINI_MODEL",
              "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"]
for c in content:
    for st in c["lab"]["steps"]:
        code = st.get("code")
        if not code:
            continue
        if any(g in code for g in GHOST_VARS):
            lines = [ln for ln in code.splitlines()
                     if not any(ln.strip().startswith(g) or ln.strip().startswith("# " + g)
                                for g in GHOST_VARS)]
            st["code"] = "\n".join(lines)
            changes.append(f"m{c['module']:02d}: removed environment variables llm.py does not read")


# --- 4. deprecated sentence-transformers method -----------------------------
# Verified by running it: 5.6.1 warns that get_sentence_embedding_dimension
# has been renamed to get_embedding_dimension.
for c in content:
    for st in c["lab"]["steps"]:
        if st.get("code") and "get_sentence_embedding_dimension" in st["code"]:
            st["code"] = st["code"].replace("get_sentence_embedding_dimension",
                                            "get_embedding_dimension")
            changes.append(f"m{c['module']:02d}: get_sentence_embedding_dimension -> get_embedding_dimension")


# --- 5. even out the quiz answer positions ----------------------------------
# The authors independently put the correct answer in slot B far too often
# (33 of 54). A learner who noticed could score 61% by always picking B, which
# would make the self-check worthless. Rotate the options so the correct answer
# lands in a balanced spread. Deterministic, so the pages are reproducible.
import random  # noqa: E402

rng = random.Random(20260805)
slots, qi = [0, 1, 2, 3], 0
plan = []
while len(plan) < 400:
    block = slots[:]
    rng.shuffle(block)
    plan.extend(block)

# Some explanations name option letters ("option A fails because..."). Moving
# the options would make those explanations point at the wrong thing, so those
# questions keep their original order.
LETTERS = re.compile(r"\b(option|options|answer|choice)s?\s+[ABCD]\b", re.I)

for c in content:
    for q in c["quiz"]:
        target = plan[qi]
        qi += 1
        opts, ans = q["options"], q["answer"]
        if LETTERS.search(q["why"]):
            changes.append(f"m{c['module']:02d}: quiz order kept (explanation names option letters)")
            continue
        if target >= len(opts) or target == ans:
            continue
        correct = opts[ans]
        rest = [o for i, o in enumerate(opts) if i != ans]
        new = rest[:target] + [correct] + rest[target:]
        q["options"] = new
        q["answer"] = target
        changes.append(f"m{c['module']:02d}: balanced quiz answer positions")

spread = {}
for c in content:
    for q in c["quiz"]:
        spread[q["answer"]] = spread.get(q["answer"], 0) + 1
print("quiz answer spread after balancing:", dict(sorted(spread.items())))

json.dump(content, io.open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"reconciled {len(content)} modules -> module_content.json")
for ch in sorted(set(changes)):
    print("  *", ch)
if not changes:
    print("  (no changes needed)")
