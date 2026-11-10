"""OCR evaluation metrics: CER, WER and exact-match accuracy.

All metrics are built on the Levenshtein (edit) distance between the
predicted text and the ground-truth text:
    distance = S + D + I   (substitutions, deletions, insertions)
    CER = char_distance / number_of_reference_characters
    WER = word_distance / number_of_reference_words

CONCEPT (FR)
------------
Un OCR lit une image et renvoie du texte. Pour l'évaluer, on compare ce texte
(l'hypothèse, "hyp") au texte correct écrit à la main (la référence, "ref").

On compte le nombre MINIMAL d'opérations pour transformer ref en hyp :
  - Substitution (S) : un caractère remplacé par un autre   "l" -> "1"
  - Délétion     (D) : un caractère oublié                  "mm" -> "m"
  - Insertion    (I) : un caractère en trop                 "ab" -> "abc"
Ce nombre s'appelle la distance de Levenshtein (ou distance d'édition).

On divise ensuite par la longueur de la référence pour obtenir un TAUX :
  CER = erreurs de caractères / nb de caractères de la référence
  WER = erreurs de mots       / nb de mots de la référence
0 % = parfait. Le taux peut dépasser 100 % si l'OCR invente beaucoup de texte.
"""
import unicodedata


def levenshtein(ref, hyp):
    """Return (distance, substitutions, deletions, insertions) between two sequences.

    Works on strings (characters) or lists (words). Classic dynamic programming,
    O(len(ref) * len(hyp)) time.
    """
    # Programmation dynamique : on remplit un tableau (n+1) x (m+1).
    # La case dp[i][j] contient le coût minimal pour transformer les i premiers
    # éléments de ref en les j premiers éléments de hyp.
    # On garde aussi le détail (S, D, I) pour savoir QUEL type d'erreur domine.
    #
    # Le même code marche pour les caractères (ref est une chaîne) et pour les
    # mots (ref est une liste de mots) : c'est ce qui distingue le CER du WER.
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, S, D, I) to turn ref[:i] into hyp[:j]
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)
    # Première colonne : hyp est vide, il faut tout supprimer (i délétions).
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    # Première ligne : ref est vide, il faut tout insérer (j insertions).
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Même élément : aucun coût, on reprend la diagonale.
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            # Sinon on essaie les 3 opérations et on garde la moins chère.
            c, s, d, ins = dp[i - 1][j - 1]
            sub = (c + 1, s + 1, d, ins)          # remplacer ref[i-1] par hyp[j-1]
            c, s, d, ins = dp[i - 1][j]
            dele = (c + 1, s, d + 1, ins)         # supprimer ref[i-1]
            c, s, d, ins = dp[i][j - 1]
            add = (c + 1, s, d, ins + 1)          # insérer hyp[j-1]
            dp[i][j] = min(sub, dele, add, key=lambda t: t[0])
    # La case en bas à droite = distance entre les deux séquences complètes.
    return dp[n][m]


def normalize(text, lower=True, strip_accents=False, collapse_spaces=True):
    """Text normalisation applied before scoring. Always report which one you used!"""
    # La normalisation décide de ce qu'on considère comme une "erreur".
    # Ex. : "Rabat" vs "rabat" est une erreur SANS normalisation, mais pas avec
    # lower=True. Deux articles ne sont comparables que s'ils normalisent pareil.
    if lower:
        text = text.lower()
    if strip_accents:
        # NFD sépare "é" en "e" + accent ; on retire ensuite les accents (catégorie Mn).
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
    if collapse_spaces:
        text = " ".join(text.split())
    return text


def cer(refs, hyps):
    """Corpus-level Character Error Rate (total edits / total reference chars)."""
    # Niveau CORPUS : on additionne toutes les erreurs puis on divise par le total
    # des caractères. Ce n'est pas la moyenne des CER par ligne : une ligne
    # longue pèse plus qu'une ligne courte, ce qui est le comportement voulu.
    edits = sum(levenshtein(r, h)[0] for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


def wer(refs, hyps):
    """Corpus-level Word Error Rate (total word edits / total reference words)."""
    # Même calcul, mais .split() découpe en mots : l'unité devient le mot.
    # Un seul caractère faux rend le mot entier faux, d'où un WER >> CER.
    edits = sum(levenshtein(r.split(), h.split())[0] for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return edits / max(total, 1)


def exact_match(refs, hyps):
    """Fraction of lines/fields recognised perfectly (sequence accuracy)."""
    # Utile pour les champs où une seule erreur rend le résultat inutilisable :
    # un numéro de facture, un IBAN, une date. C'est tout ou rien.
    return sum(r == h for r, h in zip(refs, hyps)) / max(len(refs), 1)


def demo():
    # Références = ce qui est réellement écrit sur le document.
    refs = [
        "Invoice number 2026-117",
        "Total amount: 1250.00 MAD",
        "Institut National des Postes et Telecommunications",
        "Rabat, Morocco",
    ]
    # Simulated OCR output with typical errors (O/0, l/1, missing chars, merged words)
    # Sorties OCR simulées avec des erreurs classiques :
    #   ligne 2 : "l" lu comme "1" et "0" lu comme "O"  -> 2 substitutions
    #   ligne 3 : espace manquant + "m" manquant         -> 2 délétions
    #   ligne 4 : majuscule perdue                       -> 1 substitution
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
    # Même données, mais en ignorant la casse : l'erreur "Rabat"/"rabat" disparaît.
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
