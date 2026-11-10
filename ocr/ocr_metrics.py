"""OCR evaluation metrics: CER, WER and exact-match accuracy.

All metrics are built on the Levenshtein (edit) distance between the
predicted text and the ground-truth text:
    distance = S + D + I   (substitutions, deletions, insertions)
    CER = char_distance / number_of_reference_characters
    WER = word_distance / number_of_reference_words
"""
import unicodedata


def levenshtein(ref, hyp):
    """Return (distance, substitutions, deletions, insertions) between two sequences.

    Works on strings (characters) or lists (words). Classic dynamic programming,
    O(len(ref) * len(hyp)) time.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, S, D, I) to turn ref[:i] into hyp[:j]
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            c, s, d, ins = dp[i - 1][j - 1]
            sub = (c + 1, s + 1, d, ins)
            c, s, d, ins = dp[i - 1][j]
            dele = (c + 1, s, d + 1, ins)
            c, s, d, ins = dp[i][j - 1]
            add = (c + 1, s, d, ins + 1)
            dp[i][j] = min(sub, dele, add, key=lambda t: t[0])
    return dp[n][m]


def normalize(text, lower=True, strip_accents=False, collapse_spaces=True):
    """Text normalisation applied before scoring. Always report which one you used!"""
    if lower:
        text = text.lower()
    if strip_accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
    if collapse_spaces:
        text = " ".join(text.split())
    return text


def cer(refs, hyps):
    """Corpus-level Character Error Rate (total edits / total reference chars)."""
    edits = sum(levenshtein(r, h)[0] for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


def wer(refs, hyps):
    """Corpus-level Word Error Rate (total word edits / total reference words)."""
    edits = sum(levenshtein(r.split(), h.split())[0] for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return edits / max(total, 1)


def exact_match(refs, hyps):
    """Fraction of lines/fields recognised perfectly (sequence accuracy)."""
    return sum(r == h for r, h in zip(refs, hyps)) / max(len(refs), 1)


def demo():
    refs = [
        "Invoice number 2026-117",
        "Total amount: 1250.00 MAD",
        "Institut National des Postes et Telecommunications",
        "Rabat, Morocco",
    ]
    # Simulated OCR output with typical errors (O/0, l/1, missing chars, merged words)
    hyps = [
        "Invoice number 2026-117",
        "Tota1 amount: 125O.00 MAD",
        "Institut Nationaldes Postes et Telecomunications",
        "rabat, Morocco",
    ]
    print("=== OCR metrics ===")
    for r, h in zip(refs, hyps):
        d, s, de, i = levenshtein(r, h)
        print(f"REF: {r!r}\nHYP: {h!r}\n  char edits={d} (S={s}, D={de}, I={i})"
              f"  CER={d / len(r):.3f}  WER={wer([r], [h]):.3f}")
    print(f"\nCorpus CER         : {cer(refs, hyps):.3%}")
    print(f"Corpus WER         : {wer(refs, hyps):.3%}")
    print(f"Exact match        : {exact_match(refs, hyps):.3%}")
    n_refs = [normalize(r) for r in refs]
    n_hyps = [normalize(h) for h in hyps]
    print(f"CER (lowercased)   : {cer(n_refs, n_hyps):.3%}")
    print(f"WER (lowercased)   : {wer(n_refs, n_hyps):.3%}")
    return {
        "cer": cer(refs, hyps), "wer": wer(refs, hyps),
        "exact": exact_match(refs, hyps),
        "cer_norm": cer(n_refs, n_hyps), "wer_norm": wer(n_refs, n_hyps),
        "refs": refs, "hyps": hyps,
    }


if __name__ == "__main__":
    demo()
