"""
Lab 1 - your first model call, two ways.

Run it:   python hello_llm.py

It prints which provider it is using, answers a question the simple way, then
answers it again the honest way so you can see what the simple way was hiding.

Nothing here is tied to one company. Change LLM_PROVIDER in your .env file and
run this exact same file again against a completely different model.
"""

import pathlib
import sys

# Make my-work/labs/_shared importable no matter which folder you run this from.
# parents[1] is the "labs" folder, so this points at my-work/labs/_shared.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))

from llm import MissingKey, chat, chat_raw, whoami  # noqa: E402

QUESTION = "In one sentence, what is a Python virtual environment?"


def main() -> None:
    # 1. Say out loud what we are about to use. When something goes wrong
    #    later in the course, this line is almost always the answer.
    print("Active setup:", whoami())

    # 2. The easy way. One string in, one string out. This is all most
    #    programs ever need.
    print("\n--- chat() ---")
    print(chat(QUESTION))

    # 3. The honest way. Underneath, a chat model does not take a string. It
    #    takes a LIST of messages, each with a role, and it returns an object
    #    with far more in it than the text. Everything else in this course
    #    builds on this shape, so it is worth seeing once.
    messages = [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": QUESTION},
    ]
    raw = chat_raw(messages)

    print("\n--- chat_raw() ---")
    print("model that actually answered:", raw.model)
    print("text:", raw.choices[0].message.content)

    # 4. Token counts. This is what you are billed for, and in Module 6 you
    #    will turn these numbers into money. Not every provider reports them.
    usage = getattr(raw, "usage", None)
    if usage:
        print("input tokens: ", usage.prompt_tokens)
        print("output tokens:", usage.completion_tokens)
    else:
        print("this provider did not report token counts")


if __name__ == "__main__":
    try:
        main()
    except MissingKey as e:
        # A missing key is a setup problem, not a crash. Say so kindly.
        print(e)
    except Exception as e:
        print(f"\nThe call failed: {type(e).__name__}: {e}")
        print("Check setup.html for the usual causes. If you are on a local")
        print("provider, the most common one is that the server is not running.")
