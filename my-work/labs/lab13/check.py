"""
check.py - Module 13 mini-project checker.

Run:  python check.py        (from inside my-work/labs/lab13/mini/)

Needs my_server.py, pins.json and report.json in this same folder.
No API key and no network: it talks to your server in memory.
"""

import asyncio, importlib.util, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def load_server(path):
    """Import my_server.py by file path, so no package layout is needed."""
    spec = importlib.util.spec_from_file_location("my_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def text_of(result):
    """Flatten any tool result into one searchable string."""
    blocks = " ".join(getattr(b, "text", "") for b in getattr(result, "content", []))
    return blocks + json.dumps(getattr(result, "structured_content", None) or {})


async def run():
    missing = [f for f in ("my_server.py", "pins.json", "report.json") if not (HERE / f).exists()]
    if missing:
        print("FAIL missing file(s): " + ", ".join(missing))
        print("Create them in " + str(HERE) + ", then run check.py again.")
        return 1

    report = json.loads((HERE / "report.json").read_text(encoding="utf-8"))
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(HERE))
    from mcp import Client  # installed by step 1 of the lab

    async with Client(load_server(HERE / "my_server.py").mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        names = sorted(tools)
        check(1 <= len(tools) <= 3, "tool count is %d, the limit is 3" % len(tools))
        check(names == sorted(report.get("tools", [])), "report.json tools match the server, which exposes %s" % names)
        check(sorted(pins) == names, "pins.json covers exactly those tools, it has %s" % sorted(pins))
        short = [n for n, t in tools.items() if len((t.description or "").strip()) < 40]
        check(not short, "every description is 40+ characters, too short: %s" % short)

        # Anything that writes or sends must default to doing nothing.
        writes = report.get("write_tools", [])
        check(isinstance(writes, list), "report.json has a write_tools list, use [] for a read-only server")
        for n in writes if isinstance(writes, list) else []:
            props = (getattr(tools.get(n), "input_schema", None) or {}).get("properties", {})
            arg = props.get("confirm", {})
            check(arg.get("type") == "boolean" and arg.get("default", False) is False,
                  "%s takes confirm as a boolean that defaults to false" % n)

        # Re-run the bad call for real. A refusal string passes, an exception does not.
        bad = report.get("bad_call", {})
        try:
            out = text_of(await client.call_tool(bad.get("tool", ""), bad.get("arguments", {})))
            check("refus" in out.lower(), "bad call to %r is refused, it returned: %s" % (bad.get("tool"), out[:70]))
            check("Traceback" not in out and str(HERE) not in out, "the refusal leaks no traceback and no absolute path")
        except Exception as e:
            check(False, "bad call raised %s instead of returning a refusal string" % type(e).__name__)

    tri = report.get("trifecta", {})
    check(all(isinstance(tri.get(k), bool) for k in ("private_data", "untrusted_content", "outbound_path")),
          "trifecta answers all three legs with true or false")
    check(len(str(report.get("cannot_reach", "")).split()) >= 5, "cannot_reach names what the server can never touch")
    print("Not checked automatically: whether your trifecta answers and cannot_reach line are true. Read them again yourself.")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(run())
    if FAILS or rc:
        print("%d check(s) failed." % (len(FAILS) or 1))
        sys.exit(1)
    print("ALL CHECKS PASSED")
