"""Send one Jev-shaped request with all three primitives; print the response.

    uv run python scripts/demo_request.py --model HuggingFaceTB/SmolLM2-135M
"""

from __future__ import annotations

import argparse
import json

from jev.engine import SystemOneEngine

REQUEST = {
    "state": (
        "ticket_message: I've been charged twice for my flight to Berlin and "
        "nobody is answering the phone. I want my money back immediately or "
        "I am disputing this with my bank."
    ),
    "questions": {
        "refund_requested": {
            "type": "noul",
            "instructions": "Does ticket_message request a refund?",
        },
        "request_type": {
            "type": "choice",
            "instructions": "What is the main request in ticket_message?",
            "criteria": {
                "refund": "The customer wants money returned.",
                "rebooking": "The customer wants a replacement flight.",
                "information": "The customer wants information only.",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": [
                "Calm; neutral tone, no complaints.",
                "Annoyed; complains but remains cooperative.",
                "Angry; threats, ultimatums, or escalation demands.",
            ],
        },
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    args = ap.parse_args()

    engine = SystemOneEngine(args.model)
    response = engine.respond(REQUEST)

    assert set(response["answers"]) == set(REQUEST["questions"]), "missing answers"
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
