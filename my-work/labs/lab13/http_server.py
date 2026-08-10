# my-work/labs/lab13/http_server.py
"""
Lab 13, step 5 - the same server, on HTTP.

The server object itself does not change at all. Only the run() line does.

Terminal 1:  python http_server.py     (leave this one running)
Terminal 2:  python http_client.py
Stop it with Ctrl+C in terminal 1.
"""

from hello_server import mcp  # the same server object, not a copy of it

if __name__ == "__main__":
    # stdio:           the client starts you and talks down a private pipe.
    # streamable-http: you listen on a port, and anything that can reach that
    #                  port can talk to you. That is the whole difference,
    #                  and it is also the whole new security problem.
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
