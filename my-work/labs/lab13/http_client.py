# my-work/labs/lab13/http_client.py
"""
Lab 13, step 5 - the same client questions, over HTTP.

Start http_server.py in another terminal first.

If you have seen streamablehttp_client in a tutorial, note the spelling
changed. Version 2 of the SDK calls it streamable_http_client, with an
underscore all the way through. You do not need it here anyway: Client takes
a plain URL and works out the transport itself.
"""

import asyncio

from mcp import Client

from mcp_client import explore  # the exact same function you wrote in step 3

URL = "http://127.0.0.1:8000/mcp"


async def main() -> None:
    # Client accepts three kinds of thing: a URL string like this one, a
    # transport object, or an MCPServer instance for in-memory testing. Give
    # it a URL and it picks streamable HTTP for you.
    async with Client(URL) as client:
        await explore(client, "streamable HTTP")

    print("\nSame server, same answers, different pipe.")


if __name__ == "__main__":
    asyncio.run(main())
