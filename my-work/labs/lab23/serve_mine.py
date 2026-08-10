"""
Serve YOUR model behind the same interface every other provider uses.

Once this is running, the whole course can call the model you trained:

    LLM_PROVIDER=mine        in your .env

and every lab, every agent, and your capstone go through it. Nothing else
changes, because the rest of the course only ever talks to this one shape.

Standard library only. No Flask, no FastAPI, no Docker.

Run:  python serve_mine.py
Then: python -c "import sys; sys.path.append('../_shared'); from llm import chat; print(chat('Hello'))"
"""

import json
import os
import pathlib
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))

PORT = int(os.environ.get("MINE_PORT", "8100"))
CKPT = os.environ.get("MINE_CKPT", str(
    pathlib.Path(__file__).resolve().parents[1] / "lab21" / "tinygpt.pt"))

# ---------------------------------------------------------------------------
# Load your model. If you have not trained one yet, this falls back to a
# deliberately silly stand-in so you can prove the plumbing works first.
# ---------------------------------------------------------------------------
_model = None
_encode = _decode = None


def load_model():
    """Load the checkpoint written by train_gpt.py, if there is one."""
    global _model, _encode, _decode
    if not os.path.exists(CKPT):
        print(f"No checkpoint at {CKPT}. Serving the stand-in model instead.")
        print("Train one with train_gpt.py, then restart this file.")
        return False
    import torch
    from tinygpt import TinyGPT              # the model you wrote in B2

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    stoi, itos = blob["stoi"], blob["itos"]
    _model = TinyGPT(len(stoi), blob["config"])
    _model.load_state_dict(blob["state"])
    _model.eval()
    _encode = lambda s: [stoi[c] for c in s if c in stoi]
    _decode = lambda ids: "".join(itos[i] for i in ids)
    print(f"Loaded {CKPT}: {sum(p.numel() for p in _model.parameters()):,} parameters")
    return True


def generate(prompt: str, max_tokens: int) -> str:
    """Your model's answer to a prompt."""
    if _model is None:
        # The stand-in. Proves the wiring without needing a trained model.
        return ("[stand-in model] I have no weights yet. Train one with "
                "train_gpt.py and restart the server.")
    import torch
    ids = _encode(prompt) or [0]
    out = _model.generate(torch.tensor([ids[-_model.block_size:]]), max_tokens)
    return _decode(out[0].tolist()[len(ids):])


def count_tokens(s: str) -> int:
    """Characters are this model's tokens, so the count is honest for it."""
    return len(s)


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # The client asks what models exist. Answer with the one you trained.
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [
                {"id": "mine", "object": "model", "owned_by": "you"}]})
        else:
            self._send({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send({"error": {"message": "not found"}}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")

        # Flatten the message list into one prompt. A real chat model would
        # apply a template here; yours was trained on plain text, so this is
        # the honest thing to do.
        prompt = "\n".join(m.get("content", "") for m in req.get("messages", []))
        answer = generate(prompt, int(req.get("max_tokens") or 120))

        self._send({
            "id": "chatcmpl-mine",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "mine"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": count_tokens(prompt),
                "completion_tokens": count_tokens(answer),
                "total_tokens": count_tokens(prompt) + count_tokens(answer),
            },
        })

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))


if __name__ == "__main__":
    load_model()
    print(f"\nYour model is serving on http://127.0.0.1:{PORT}/v1")
    print("Put LLM_PROVIDER=mine in your .env, then run any lab.\n")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
