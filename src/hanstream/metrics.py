"""Deterministic edit alignment and micro-averaged corpus scoring."""

from collections import Counter
from dataclasses import dataclass

from .text import TextPolicy, tokens


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def rate(self) -> float:
        """Empty-reference insertions count as their raw count (denominator one)."""
        return self.errors / max(1, self.reference_length)


def align(reference, hypothesis) -> list[tuple[str, str | None, str | None]]:
    """Levenshtein alignment; ties prefer diagonal, then deletion, then insertion."""
    ref, hyp = list(reference), list(hypothesis)
    table = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        table[i][0] = i
    for j in range(len(hyp) + 1):
        table[0][j] = j
    for i, r in enumerate(ref, 1):
        for j, h in enumerate(hyp, 1):
            table[i][j] = min(
                table[i - 1][j - 1] + (r != h), table[i - 1][j] + 1, table[i][j - 1] + 1
            )
    i, j, output = len(ref), len(hyp), []
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            output.append(('hit' if ref[i - 1] == hyp[j - 1] else 'sub', ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            output.append(('del', ref[i - 1], None))
            i -= 1
        else:
            output.append(('ins', None, hyp[j - 1]))
            j -= 1
    return list(reversed(output))


def edit_counts(reference, hypothesis) -> EditCounts:
    """Count each edit category from a deterministic alignment."""
    counts = Counter(op for op, _, _ in align(reference, hypothesis))
    return EditCounts(counts['sub'], counts['del'], counts['ins'], counts['hit'])


def score(
    reference: str, hypothesis: str, unit: str = 'word', policy: TextPolicy = TextPolicy()
) -> EditCounts:
    """Score normalized words or characters."""
    return edit_counts(tokens(reference, unit, policy), tokens(hypothesis, unit, policy))


def corpus_score(references, hypotheses, unit: str = 'word') -> EditCounts:
    """Micro-average errors; reject dropped or unmatched utterances."""
    refs, hyps = list(references), list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError('reference and hypothesis counts differ')
    results = [score(r, h, unit) for r, h in zip(refs, hyps)]
    return EditCounts(
        *(
            sum(getattr(r, key) for r in results)
            for key in ('substitutions', 'deletions', 'insertions', 'hits')
        )
    )


def confusion_pairs(
    reference: str, hypothesis: str, unit: str = 'char'
) -> dict[tuple[str, str], int]:
    """Return substitution pairs; exclude insertions and deletions."""
    return dict(
        Counter(
            (r, h)
            for op, r, h in align(tokens(reference, unit), tokens(hypothesis, unit))
            if op == 'sub'
        )
    )


def sentence_error_rate(references, hypotheses, unit: str = 'word') -> float:
    """Fraction of utterances containing any normalized edit."""
    refs, hyps = list(references), list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError('reference and hypothesis counts differ')
    return sum(score(r, h, unit).errors > 0 for r, h in zip(refs, hyps)) / max(1, len(refs))
