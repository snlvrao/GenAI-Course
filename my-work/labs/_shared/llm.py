"""
llm.py - the one place the course talks to a language model.

Every lab imports this. You change which model you use by editing .env,
never by editing code. That is the whole point: nothing in this course is
tied to one company, and if a provider disappears or gets expensive you
change one line and carry on.

This works because almost every provider now speaks the same HTTP dialect,
the one OpenAI published. So a single client library, pointed at a different
address, talks to all of them - including a model running on your own laptop
with no account and no internet.

    from llm import chat
    print(chat("Say hello in five words."))

Switch provider from the command line:

    set LLM_PROVIDER=groq        (Windows)
    export LLM_PROVIDER=groq     (Mac/Linux)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI

try:  # optional - the labs work without it, it just saves you exporting vars
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Provider:
    """Everything needed to reach one provider."""

    name: str
    base_url: str
    key_env: str          # which environment variable holds the key ("" if none)
    default_model: str
    local: bool = False   # True = runs on your machine, no account, no internet
    note: str = ""


# ---------------------------------------------------------------------------
# The registry. Adding a provider is one line - that is the anti-lock-in point.
#
# Model names go stale faster than anything else here. If a call fails with
# "model not found", the model was renamed or retired; check the provider's
# model list and update. Nothing else needs to change.
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Provider] = {
    # --- free hosted, no card needed to start -----------------------------
    "groq": Provider(
        "groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
        "llama-3.1-8b-instant",
        note="Very generous free tier and extremely fast. Best default for labs.",
    ),
    "gemini": Provider(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY", "gemini-3.6-flash",
        note="Google AI Studio. Free, no card. Also does embeddings.",
    ),
    "huggingface": Provider(
        "huggingface", "https://router.huggingface.co/v1", "HF_TOKEN",
        "openai/gpt-oss-120b:cheapest",
        note="Routes to whichever company is cheapest. Open models only.",
    ),
    "openrouter": Provider(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
        "z-ai/glm-4.5-air:free",
        note="One key, many providers. Has some free models.",
    ),

    # --- paid, if you already have a key ----------------------------------
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5.6-luna"),
    "deepseek": Provider("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-v4-flash"),
    "together": Provider("together", "https://api.together.ai/v1", "TOGETHER_API_KEY", "openai/gpt-oss-120b"),
    "anthropic": Provider(
        "anthropic", "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-sonnet-5",
        note="Works, but Anthropic says this compatibility mode is for testing, "
             "not production. It quietly ignores response_format, seed and caching. "
             "Module 9 uses that as a lesson.",
    ),

    # --- a model you trained yourself (the Build track) --------------------
    "mine": Provider(
        "mine", "http://127.0.0.1:8100/v1", "", "mine", local=True,
        note="Your own model, served by my-work/labs/lab_b/serve_mine.py. Start that first.",
    ),

    # --- fully local: no key, no account, no internet ---------------------
    "ollama": Provider(
        "ollama", "http://localhost:11434/v1/", "", "qwen2.5:3b-instruct", local=True,
        note="Easiest local option. Install Ollama, run: ollama pull qwen2.5:3b-instruct",
    ),
    "llamacpp": Provider(
        "llamacpp", "http://127.0.0.1:8080/v1", "", "local-model", local=True,
        note="llama-server.exe. Start it with --jinja or tool calling silently does nothing.",
    ),
    "lmstudio": Provider(
        "lmstudio", "http://localhost:1234/v1", "", "local-model", local=True,
        note="LM Studio's built-in server. Turn it on in the Developer tab.",
    ),
    "gpt4all": Provider(
        "gpt4all", "http://localhost:4891/v1", "", "local-model", local=True,
        note="GPT4All is an application, not a model, in the same family as Ollama "
             "and LM Studio. Enable its API server in Settings, then set LLM_MODEL "
             "to whichever model you loaded.",
    ),
}

DEFAULT = os.environ.get("LLM_PROVIDER", "groq")


class MissingKey(RuntimeError):
    """Raised with an explanation you can actually act on."""


class EmptyAnswer(RuntimeError):
    """Raised when the model replied, but with no answer in it."""


# The example text in setup.html and .env.example. Leaving one of these in
# place is the commonest setup mistake, and it is worth catching by name: the
# provider answers a placeholder with the same 401 it uses for a revoked key,
# which sends you looking at your account instead of at your .env.
PLACEHOLDERS = {
    "paste_your_key_here", "your_key_here", "your-key-here", "sk-...",
    "abc123", "xxx", "changeme", "<your key>", "<your_key>",
}


def is_placeholder(key: str) -> bool:
    return key.strip().strip("\"'").lower() in PLACEHOLDERS


def client_for(provider: str | None = None) -> tuple[OpenAI, str]:
    """Return (client, model_name) for a provider. Same shape for all of them."""
    name = (provider or DEFAULT).strip().lower()
    if name not in REGISTRY:
        raise MissingKey(
            f"Unknown provider {name!r}.\nChoose one of: {', '.join(sorted(REGISTRY))}"
        )
    p = REGISTRY[name]

    if p.local:
        # Local servers ignore the key but the client library insists on one.
        key = "not-needed"
    else:
        key = os.environ.get(p.key_env, "").strip()
        if not key:
            raise MissingKey(
                f"No API key found for '{name}'.\n"
                f"  1. Get a free key, then\n"
                f"  2. put this line in your .env file:  {p.key_env}=your_key_here\n"
                f"Or use a local model instead with:      set LLM_PROVIDER=ollama"
            )
        if is_placeholder(key):
            raise MissingKey(
                f"{p.key_env} still holds the example text, not a key.\n"
                f"  It reads: {key}\n"
                "Replace that whole value with the key from your provider.\n"
                "The provider would reject this with a 401, which reads like a\n"
                "broken key rather than an unedited line."
            )

    model = os.environ.get("LLM_MODEL", "").strip() or p.default_model
    return OpenAI(base_url=p.base_url, api_key=key), model


def chat(prompt: str, *, system: str | None = None, provider: str | None = None,
         temperature: float = 0.2, max_tokens: int = 800) -> str:
    """Ask a question, get a string back. The simplest possible call."""
    client, model = client_for(provider)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if text:
        return text

    # An empty answer is nearly always a thinking model. Models like qwen3
    # write their reasoning first, and providers return that in a separate
    # field, so "content" stays empty until the reasoning finishes. If the
    # token budget runs out first, you get a reply with no answer in it.
    # Returning "" here would look like nothing happened, so say what did.
    thinking = (getattr(choice.message, "reasoning", None)
                or getattr(choice.message, "reasoning_content", None) or "")
    used = getattr(resp.usage, "completion_tokens", 0) or 0
    if thinking:
        raise EmptyAnswer(
            f"{model} thought for {used} tokens and never got to an answer.\n"
            f"It stopped because: {choice.finish_reason}.\n\n"
            "This is a thinking model. It writes its reasoning first, and that\n"
            "reasoning is what filled the budget. Two ways forward:\n"
            f"  - give it more room: chat(prompt, max_tokens={max(2000, max_tokens * 4)})\n"
            "  - or use a model that answers directly. For the offline path,\n"
            "    qwen2.5:3b-instruct and granite4.1:3b both do.\n"
            "    Change LLM_MODEL in your .env, not this code."
        )
    raise EmptyAnswer(
        f"{model} returned an empty reply, and no reasoning either.\n"
        f"It stopped because: {choice.finish_reason}.\n"
        "If that says 'length', raise max_tokens. Otherwise check the model\n"
        "name in your .env against the list the provider actually serves."
    )


def chat_raw(messages: list[dict], *, tools: list | None = None,
             provider: str | None = None, **kw):
    """Full access, for the labs that need tool calling or the raw response."""
    client, model = client_for(provider)
    args: dict = {"model": model, "messages": messages}
    if tools:
        args["tools"] = tools
        args["tool_choice"] = "auto"
    args.update(kw)
    return client.chat.completions.create(**args)


def whoami(provider: str | None = None) -> str:
    """Print what you are actually about to use. Run this when confused."""
    name = (provider or DEFAULT).strip().lower()
    p = REGISTRY.get(name)
    if not p:
        return f"Unknown provider: {name}"
    model = os.environ.get("LLM_MODEL", "").strip() or p.default_model
    where = "your own machine" if p.local else p.base_url
    raw = os.environ.get(p.key_env, "")
    if p.local:
        key = "none needed"
    elif not raw.strip():
        key = f"MISSING ({p.key_env})"
    elif is_placeholder(raw):
        key = f"STILL THE EXAMPLE TEXT ({p.key_env})"
    else:
        key = "set"
    return f"provider={p.name}  model={model}  endpoint={where}  key={key}"


if __name__ == "__main__":
    # Run `python llm.py` as a health check before starting any lab.
    print(whoami())
    try:
        print("reply:", chat("Reply with exactly: OK", max_tokens=16))
        print("\nWorking. You are ready to start.")
    except MissingKey as e:
        print("\n" + str(e))
    except Exception as e:  # network down, model renamed, server not started
        print(f"\nCall failed: {type(e).__name__}: {e}")
        print("If you are using a local provider, is the server actually running?")
