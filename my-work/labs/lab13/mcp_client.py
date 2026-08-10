"""
Lab 13, step 3 - your own MCP client.

This is the point of the whole module. You do not need Claude Desktop, or
Cursor, or VS Code, or an API key, or an internet connection. An MCP server is
just a program that answers JSON, and this is the program that asks.

Run:  python mcp_client.py
Or:   python mcp_client.py <server.py> <tool_name> '<json arguments>'

Notice there is no session.initialize() call anywhere. That is not an oversight.
The 2026-07-28 protocol removed the handshake entirely.
"""

import asyncio
import json
import sys

from mcp import Client, StdioServerParameters, stdio_client
from mcp.types import TextContent

import hello_server


def text_of(result) -> str:
    """Flatten a tool result into something printable.

    A result can carry structured data, plain text, or both, so check for the
    structured form first and fall back to the text blocks.
    """
    if getattr(result, "structured_content", None) is not None:
        return json.dumps(result.structured_content)
    return "\n".join(b.text for b in result.content if isinstance(b, TextContent))


async def explore(client: Client, label: str) -> None:
    print(f"\n--- {label} ---")
    print("server:  ", client.server_info.name, client.server_info.version)
    print("protocol:", client.protocol_version)  # prints 2026-07-28

    tools = await client.list_tools()
    print("tools:   ", ", ".join(t.name for t in tools.tools))

    # Call a tool exactly the way a model would: by name, with a dict of args.
    print("add(2,40)   ->", text_of(await client.call_tool("add", {"a": 2, "b": 40})))
    print("shout('hi') ->", text_of(await client.call_tool("shout", {"text": "hi"})))

    # Resources are addressed by URI, not called like functions.
    res = await client.read_resource("greeting://you")
    print("resource    ->", res.contents[0].text)

    prompts = await client.list_prompts()
    print("prompts: ", ", ".join(p.name for p in prompts.prompts))


async def call_one(server_file: str, tool: str, args: dict) -> None:
    """Start any server file and call one named tool on it.

    This is the mode steps 4 and 6 use, because explore() above is hard-wired
    to the hello server's own tool names.
    """
    params = StdioServerParameters(command=sys.executable, args=[server_file])
    async with Client(stdio_client(params)) as client:
        tools = await client.list_tools()
        print("tools:", ", ".join(t.name for t in tools.tools))
        print(f"{tool} ->", text_of(await client.call_tool(tool, args)))


async def main() -> None:
    # Command line mode:  python mcp_client.py <server.py> <tool> '<json args>'
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        await call_one(sys.argv[1], sys.argv[2], args)
        return

    # Way 1: in memory. No subprocess, no ports. The fastest way to test while
    # you are still writing the server, and what you would use in a unit test.
    async with Client(hello_server.mcp) as client:
        await explore(client, "in-memory (no subprocess)")

    # Way 2: over stdio, exactly as a real host app would run it. If this
    # works, any MCP host can run your server.
    #
    # sys.executable, NOT the word "python". The bare word resolves against
    # your PATH, which is usually not the virtual environment you are sitting
    # in, so the subprocess starts, fails to import mcp, and dies. All you
    # would see is "Connection closed", which tells you nothing useful.
    params = StdioServerParameters(command=sys.executable, args=["hello_server.py"])
    async with Client(stdio_client(params)) as client:
        await explore(client, "stdio subprocess (what a real host does)")

    print("\nBoth transports worked. Your server is a real MCP server.")


if __name__ == "__main__":
    asyncio.run(main())