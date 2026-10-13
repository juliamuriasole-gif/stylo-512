"""Role: Experimenter. Measures attribution accuracy as a function of sample length.

Protocol (no leakage): each author's token stream is split once: the first
`train_frac` is training text, the rest is held out. Test samples of N words are
drawn only from held-out text, so the model never sees test words in training.
"""
import random

from .delta import DeltaModel


def split(author_tokens, train_frac=0.7):
    train, test = {}, {}
    for a, toks in author_tokens.items():
        cut = int(len(toks) * train_frac)
        train[a], test[a] = toks[:cut], toks[cut:]
    return train, test


def accuracy_curve(author_tokens, lengths=(32, 64, 128, 256, 512, 1024),
                   trials=50, seed=0, train_frac=0.7, chunk_size=500):
    """Return list of dicts: {words, accuracy, trials, chance}. Lengths longer than
    an author's held-out text are skipped for that author; if no author can supply
    a sample the length is omitted."""
    rng = random.Random(seed)
    train, test = split(author_tokens, train_frac)
    model = DeltaModel(chunk_size=chunk_size).fit(train)
    authors = sorted(author_tokens)
    out = []
    for n in lengths:
        correct = total = 0
        for a in authors:
            held = test[a]
            if len(held) < n:
                continue
            for _ in range(trials):
                start = rng.randrange(0, len(held) - n + 1)
                if model.predict(held[start:start + n]) == a:
                    correct += 1
                total += 1
        if total:
            out.append({"words": n, "accuracy": correct / total, "trials": total,
                        "chance": 1 / len(authors)})
    return out


def words_needed(curve, target=0.9):
    """Smallest sample length whose accuracy reaches target, else None."""
    for row in curve:
        if row["accuracy"] >= target:
            return row["words"]
    return None
