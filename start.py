"""Open the course with videos working.

Why this exists. The pages are plain HTML and you can open index.html by
double-clicking it. Everything works that way except one thing: YouTube will
not run its player inside a page that has no address. Opened from disk a page
has none, so every video comes back "Error 153, video player configuration
error". That is YouTube's rule and nothing in the page can talk it round.

Giving the pages an address fixes it. This serves the course folder on your own
machine and opens it. Nothing leaves your computer, no account, no internet
needed except by the videos themselves, and it uses only what comes with
Python.

Run it:   python start.py
Stop it:  close this window, or press Ctrl+C
"""

import contextlib
import http.server
import os
import socket
import socketserver
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_CHOICE = 8000


def free_port(start):
    """First port from `start` that nothing else is holding."""
    for port in range(start, start + 50):
        with contextlib.closing(socket.socket()) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("Could not find a free port between "
                     f"{start} and {start + 49}. Close something and retry.")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):
        pass    # one line per file would bury the instructions below


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    if not os.path.exists(os.path.join(HERE, "index.html")):
        raise SystemExit("index.html is not next to this script, so this is not "
                         "the course folder.")

    port = free_port(FIRST_CHOICE)
    url = f"http://127.0.0.1:{port}/index.html"

    with Server(("127.0.0.1", port), Handler) as httpd:
        print()
        print("  The course is open at")
        print(f"      {url}")
        print()
        print("  Videos play inside the page here. Leave this window open while")
        print("  you read, and close it when you are finished.")
        print()
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Closed. Your progress is saved in the browser.\n")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
