"""Role: Data generator. Synthetic 'authors' with distinct function-word habits,
so tests and the offline demo need no downloads."""
import random

from .features import FUNCTION_WORDS

CONTENT = ("river stone garden letter window morning city voice market table night road "
           "house paper light winter friend story music water field train coat door").split()


def make_author(seed, n_words=20000, style_strength=1.0):
    rng = random.Random(seed)
    # Each author over/under-uses function words by a personal multiplier.
    weights = [max(0.05, 1 + style_strength * rng.gauss(0, 0.6)) for _ in FUNCTION_WORDS]
    words = []
    for _ in range(n_words):
        if rng.random() < 0.5:
            words.append(rng.choices(FUNCTION_WORDS, weights)[0])
        else:
            words.append(rng.choice(CONTENT))
    return words


def make_corpus(n_authors=5, n_words=20000, style_strength=1.0, seed=42):
    return {f"author_{i}": make_author(seed * 1000 + i, n_words, style_strength)
            for i in range(n_authors)}
