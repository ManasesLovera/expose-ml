"""
email_vectorizer.py
-------------------
Converts raw email text into the 57-feature vector used by the UCI Spambase dataset.

Feature layout (matches dataset column order):
  [0:48]  word_freq_*              – % of words in email matching each keyword
  [48:54] char_freq_; ( [ ! $ #   – % of characters that are each special char
  [54]    capital_run_length_average
  [55]    capital_run_length_longest
  [56]    capital_run_length_total
"""

import re
import math
from typing import Union


# ── Feature definitions (order matches UCI Spambase column order) ──────────────

WORD_FEATURES = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference",
]

CHAR_FEATURES = [";", "(", "[", "!", "$", "#"]

COLUMN_NAMES = (
    [f"word_freq_{w}" for w in WORD_FEATURES]
    + [f"char_freq_{c}" for c in CHAR_FEATURES]
    + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
)


# ── Core helpers ───────────────────────────────────────────────────────────────

def _word_frequencies(text: str) -> list[float]:
    """Percentage of words in the email that match each keyword."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    total = len(tokens)
    if total == 0:
        return [0.0] * len(WORD_FEATURES)

    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    return [round(counts.get(w, 0) / total * 100, 3) for w in WORD_FEATURES]


def _char_frequencies(text: str) -> list[float]:
    """Percentage of characters in the email that are each special character."""
    total = len(text)
    if total == 0:
        return [0.0] * len(CHAR_FEATURES)

    return [round(text.count(ch) / total * 100, 3) for ch in CHAR_FEATURES]


def _capital_run_stats(text: str) -> tuple[float, int, int]:
    """
    Returns (average_run_length, longest_run, total_capitals).
    A 'run' is a consecutive sequence of uppercase letters.
    """
    runs = []
    current = 0
    for ch in text:
        if ch.isupper():
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0
    if current > 0:
        runs.append(current)

    if not runs:
        return 0.0, 0, 0

    total = sum(runs)
    longest = max(runs)
    average = round(total / len(runs), 3)
    return average, longest, total


# ── Public API ─────────────────────────────────────────────────────────────────

def email_to_vector(text: str) -> list[float]:
    """
    Convert raw email text to a 57-element feature vector.

    The vector matches the exact column order of the UCI Spambase dataset,
    so it can be fed directly into any classifier trained on that data.

    Parameters
    ----------
    text : str
        The full email text (headers + body, or body alone).

    Returns
    -------
    list[float]
        57 numeric features in Spambase column order.

    Example
    -------
    >>> vec = email_to_vector("Get free money now!!! CLICK HERE!!!")
    >>> len(vec)
    57
    """
    word_freqs = _word_frequencies(text)
    char_freqs = _char_frequencies(text)
    cap_avg, cap_long, cap_total = _capital_run_stats(text)

    return word_freqs + char_freqs + [cap_avg, cap_long, cap_total]


def email_to_dict(text: str) -> dict[str, float]:
    """
    Same as email_to_vector() but returns a dict keyed by feature name.

    Useful for inspection or building a pandas DataFrame row.

    Example
    -------
    >>> d = email_to_dict("Free offer! Win money now!!!")
    >>> d["word_freq_free"]
    33.333
    >>> d["char_freq_!"]
    8.333
    """
    return dict(zip(COLUMN_NAMES, email_to_vector(text)))


# ── Optional: pretty-print non-zero features ──────────────────────────────────

def summarize(text: str) -> None:
    """Print a human-readable summary of non-zero features for an email."""
    d = email_to_dict(text)
    print(f"{'Feature':<35} {'Value':>8}")
    print("-" * 45)
    for name, val in d.items():
        if val != 0.0:
            print(f"{name:<35} {val:>8.3f}")


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spam_email = """
    CONGRATULATIONS! You have been selected to receive a FREE credit report!
    Click NOW to CLAIM your $1000 MONEY reward. LIMITED TIME OFFER!!!
    Remove yourself from future mailings. Our business is helping YOU make MONEY FAST.
    Order today! Free internet access! Send your email addresses for more FREE offers!!!
    """

    ham_email = """
    Hi George,

    Following up on our meeting from yesterday. I wanted to share the project report
    we discussed. The lab data looks promising for the 1999 technology study.

    Please review the table in the attached document before the conference next week.

    Best regards,
    HP Labs Team
    """

    print("=== SPAM EMAIL ===")
    summarize(spam_email)

    print("\n=== HAM EMAIL ===")
    summarize(ham_email)

    print("\n=== RAW VECTOR (first 10 features) ===")
    vec = email_to_vector(spam_email)
    print(vec[:10], "...")
    print(f"Total features: {len(vec)}")
