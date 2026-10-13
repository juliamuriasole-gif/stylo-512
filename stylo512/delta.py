"""Role: Attributor. Burrows' Delta nearest-author classification."""
import math

from .features import FUNCTION_WORDS, frequencies, tokenize


def _chunks(tokens, size):
    return [tokens[i:i + size] for i in range(0, len(tokens) - size + 1, size)]


class DeltaModel:
    """Burrows' Delta (2002): z-score function-word frequencies across training
    chunks, then attribute a text to the author whose mean z-profile has the
    smallest mean absolute difference."""

    def __init__(self, vocab=FUNCTION_WORDS, chunk_size=500):
        self.vocab = tuple(vocab)
        self.chunk_size = chunk_size
        self.means = None
        self.stds = None
        self.profiles = {}

    def fit(self, author_tokens):
        """author_tokens: {author: [tokens]}. Needs >= 2 authors."""
        if len(author_tokens) < 2:
            raise ValueError("need at least two authors")
        rows = {}
        for author, toks in author_tokens.items():
            chunks = _chunks(toks, self.chunk_size) or [toks]
            if not toks:
                raise ValueError(f"author {author!r} has no text")
            rows[author] = [frequencies(c, self.vocab) for c in chunks]
        allrows = [r for rs in rows.values() for r in rs]
        k = len(self.vocab)
        n = len(allrows)
        self.means = [sum(r[j] for r in allrows) / n for j in range(k)]
        self.stds = []
        for j in range(k):
            var = sum((r[j] - self.means[j]) ** 2 for r in allrows) / max(n - 1, 1)
            self.stds.append(math.sqrt(var))
        self.profiles = {}
        for author, rs in rows.items():
            avg = [sum(r[j] for r in rs) / len(rs) for j in range(k)]
            self.profiles[author] = self._z(avg)
        return self

    def _z(self, freqs):
        # Words with zero variance carry no information: z = 0.
        return [(f - m) / s if s > 0 else 0.0 for f, m, s in zip(freqs, self.means, self.stds)]

    def deltas(self, text_or_tokens):
        """Delta distance to each author (lower = more similar)."""
        if self.means is None:
            raise RuntimeError("model is not fitted")
        toks = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else list(text_or_tokens)
        z = self._z(frequencies(toks, self.vocab))
        k = len(z)
        return {a: sum(abs(x - y) for x, y in zip(z, p)) / k for a, p in self.profiles.items()}

    def predict(self, text_or_tokens):
        d = self.deltas(text_or_tokens)
        return min(sorted(d), key=lambda a: d[a])
