"""
Lab 13, step 2 - the smallest MCP server that is actually complete.

It offers one of each of the three things an MCP server can expose:

  tool     - something the AI can DO      (the model decides when to call it)
  resource - something the AI can READ    (your app decides when to attach it)
  prompt   - a canned instruction         (you pick it, usually from a menu)

Run this file directly and nothing appears to happen. That is correct. An MCP
server talks JSON over its standard input and output and waits to be spoken to.
Use mcp_client.py, or the one-line raw JSON call in README.md, to talk to it.
"""

from mcp.server import MCPServer  # NOT mcp.server.fastmcp. That module is gone.

mcp = MCPServer("hello", instructions="A tiny demonstration server.", version="1.0.0")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the total."""
    # The decorators do more than they look like they do. The function name
    # becomes the tool name, this docstring becomes the description the model
    # reads to decide whether to call it, and the type hints become the
    # argument schema. A lazy docstring makes a tool the model uses badly.
    return a + b


@mcp.tool()
def shout(text: str) -> str:
    """Return the given text in capital letters."""
    return text.upper()


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """A fixed greeting line for one person."""
    return f"Hello, {name}. Nice to meet you."


@mcp.prompt()
def summarise(topic: str) -> str:
    """A reusable instruction for summarising a topic."""
    return f"Write a five line summary of {topic}. Plain English, no jargon."


if __name__ == "__main__":
    # stdio: the client starts this file as a subprocess and talks over the
    # pipes. That is the transport you use for a server on your own machine.
    # The other standard option is transport="streamable-http".
    mcp.run()
