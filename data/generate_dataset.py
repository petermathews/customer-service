"""Generate a labelled customer-service ticket dataset.

Why synthetic? Real support queues are proprietary — the exact constraint
behind this whole exercise. So we generate a reproducible stand-in: realistic
ticket text, some with attachments (receipts and photos), each labelled with
the intent / sentiment / route a coach would have a team predict. Swapping in a
real public dataset (e.g. HuggingFace `banking77`) is a one-cell change shown
in the notebook.

Run:  python data/generate_dataset.py
Writes: data/tickets.csv  (and prints a class summary)
"""

from __future__ import annotations

import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# Templates per intent. Multiple phrasings give TF-IDF and the neural net
# something real to learn — the words overlap across intents on purpose, so the
# task isn't trivially separable (just like real tickets).
TEMPLATES = {
    "billing_question": [
        "I was charged ${amt} this month but my plan should be cheaper, can you explain the bill?",
        "Why is there an extra charge of ${amt} on my latest invoice?",
        "I don't understand my billing statement, there's a line item I didn't expect.",
        "Can you break down what the ${amt} fee on my account is for?",
        "My subscription price went up, what changed on my billing?",
    ],
    "refund_request": [
        "I'd like a refund for order {oid}, the item never worked.",
        "Please refund the ${amt} I paid, I'm returning the product.",
        "I want my money back for order {oid}, it's not what I expected.",
        "Requesting a refund — order {oid} arrived too late to be useful.",
        "Can I get a refund? I was double charged ${amt}.",
    ],
    "technical_issue": [
        "The app keeps crashing every time I open the dashboard.",
        "I can't get the export feature to work, it just spins forever.",
        "Getting a 500 error when I try to save my settings.",
        "The sync between my devices stopped working yesterday.",
        "Login page loads but the buttons don't respond at all.",
    ],
    "damaged_product": [
        "The package arrived and the screen is cracked, see the photo.",
        "My order {oid} came damaged, there's a dent on the side.",
        "Item showed up broken, attaching a picture of the damage.",
        "The product is scratched badly out of the box, photo attached.",
        "Received a torn and damaged unit for order {oid}.",
    ],
    "order_status": [
        "Where is my order {oid}? It was supposed to arrive last week.",
        "Can you tell me the shipping status of order {oid}?",
        "My tracking hasn't updated in days, what's going on with {oid}?",
        "When will order {oid} be delivered?",
        "Has my package shipped yet? Order number is {oid}.",
    ],
    "account_access": [
        "I'm locked out of my account and the reset email never arrives.",
        "Someone may have accessed my account, I can't log in anymore.",
        "I forgot my password and the recovery link isn't working.",
        "My two-factor codes stopped working, I can't get in.",
        "I think my account was hacked, please help me secure it.",
    ],
    "complaint": [
        "This is the third time I've contacted you and nobody has helped. Unacceptable.",
        "I'm extremely frustrated with the service I've received, this is terrible.",
        "Worst support experience ever, I've been waiting for weeks.",
        "I'm very disappointed and considering cancelling entirely.",
        "Absolutely fed up — your team keeps ignoring my emails.",
    ],
    "general_inquiry": [
        "Do you offer a student discount on annual plans?",
        "What are your support hours over the holidays?",
        "Is there a mobile app available for Android?",
        "Can I change my plan in the middle of a billing cycle?",
        "Do you ship internationally to Canada?",
    ],
}

# Deliberately ambiguous tickets that borrow vocabulary across intents — a
# refund ticket that talks about charges, a tech issue that mentions the
# account. Real tickets do this constantly, and it's what keeps intent accuracy
# realistic (high 80s–90s, not a suspicious 100%). The label is the *primary*
# intent a human would route on.
AMBIGUOUS = {
    "billing_question": [
        "I see a refund was supposed to come back but I was charged ${amt} again.",
        "There's a charge and a credit on the same statement, can you explain the bill?",
    ],
    "refund_request": [
        "The charge on my bill is wrong, I need that ${amt} returned for order {oid}.",
        "I was billed twice, please refund order {oid}.",
    ],
    "technical_issue": [
        "I can't access my account because the app keeps erroring out on login.",
        "The dashboard crashes whenever I open my account settings.",
    ],
    "account_access": [
        "There's an error every time I try to log in and the page crashes.",
        "App keeps failing when I enter my password, I'm locked out.",
    ],
    "complaint": [
        "I've asked for a refund three times and still nothing, this is unacceptable.",
        "Your app is broken AND nobody answers — I'm beyond frustrated.",
    ],
    "order_status": [
        "I was charged but order {oid} still hasn't shipped, where is it?",
        "Paid for express shipping on {oid} and tracking shows nothing.",
    ],
}
for _intent, _extra in AMBIGUOUS.items():
    TEMPLATES[_intent].extend(_extra)

# Shared filler added to many tickets — greetings, sign-offs, and hedges that
# appear across every intent. This is what real tickets look like, and it's
# what stops a classifier from trivially memorising templates: the same words
# show up under different labels, so the model has to learn the signal, not the
# boilerplate. It pulls accuracy down into a realistic band.
PREAMBLES = [
    "", "", "Hi team, ", "Hello, ", "Hey there — ", "Good morning. ",
    "Thanks for reading this. ", "Quick question — ", "Sorry to bother you, but ",
]
SUFFIXES = [
    "", "", " Thanks in advance.", " Please let me know.", " Appreciate any help.",
    " Looking forward to your reply.", " This is pretty urgent for me.",
    " I've been a customer for years.",
]


# Which intents tend to carry which sentiment, so labels aren't random noise.
SENTIMENT_BIAS = {
    "complaint": ["negative", "negative", "negative", "neutral"],
    "damaged_product": ["negative", "negative", "neutral"],
    "refund_request": ["negative", "neutral", "neutral"],
    "technical_issue": ["negative", "neutral", "neutral"],
    "account_access": ["negative", "neutral"],
    "billing_question": ["neutral", "neutral", "negative"],
    "order_status": ["neutral", "neutral", "negative"],
    "general_inquiry": ["neutral", "neutral", "positive"],
}

# Intents that sometimes ship with an attachment, and of what kind.
ATTACHMENT_RULE = {
    "refund_request": "document",     # a receipt to extract from
    "damaged_product": "image",       # a photo to classify
}


def make_receipt(order_id: str, amount: float, date: str) -> str:
    """A tiny text 'receipt' — the document a GenAI/agent approach extracts
    fields from, and that rules can only regex at."""
    return (
        f"ORDER CONFIRMATION\n"
        f"Order #: {order_id}\n"
        f"Date: {date}\n"
        f"Item: Wireless Headphones\n"
        f"Total: ${amount:.2f}\n"
        f"Thank you for your purchase!"
    )


def main() -> None:
    rng = random.Random(42)  # deterministic — same dataset every run
    rows = []
    n_per_intent = 30

    for intent, templates in TEMPLATES.items():
        for i in range(n_per_intent):
            template = templates[i % len(templates)]
            oid = f"A{rng.randint(10000, 99999)}"
            amt = round(rng.uniform(19.0, 240.0), 2)
            date = f"2026-0{rng.randint(1, 6)}-{rng.randint(10, 28)}"
            body = template.format(oid=oid, amt=f"{amt:.2f}")
            text = f"{rng.choice(PREAMBLES)}{body}{rng.choice(SUFFIXES)}"

            sentiment = rng.choice(SENTIMENT_BIAS.get(intent, ["neutral"]))

            attachment_type = "none"
            attachment_content = ""
            kind = ATTACHMENT_RULE.get(intent)
            # ~70% of attachment-eligible intents actually include one.
            if kind and rng.random() < 0.7:
                attachment_type = kind
                if kind == "document":
                    attachment_content = make_receipt(oid, amt, date)
                else:  # image — store a label the image classifier "sees"
                    attachment_content = rng.choice(
                        ["photo_damaged", "photo_damaged", "photo_intact"]
                    )

            rows.append(
                {
                    "id": f"T{len(rows):04d}",
                    "text": text,
                    "intent": intent,
                    "sentiment": sentiment,
                    "attachment_type": attachment_type,
                    "attachment_content": attachment_content,
                }
            )

    rng.shuffle(rows)

    out_path = os.path.join(HERE, "tickets.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "text",
                "intent",
                "sentiment",
                "attachment_type",
                "attachment_content",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Print a quick summary so it's obvious what we built.
    print(f"Wrote {len(rows)} tickets to {out_path}")
    counts = {}
    for r in rows:
        counts[r["intent"]] = counts.get(r["intent"], 0) + 1
    print("Tickets per intent:")
    for intent, c in sorted(counts.items()):
        print(f"  {intent:18s} {c}")
    n_doc = sum(1 for r in rows if r["attachment_type"] == "document")
    n_img = sum(1 for r in rows if r["attachment_type"] == "image")
    print(f"Attachments: {n_doc} documents, {n_img} images")


if __name__ == "__main__":
    main()
