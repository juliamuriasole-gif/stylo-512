"""Role: Feature extractor. Turns text into function-word frequency vectors."""
import re

# Common English function words: topic-independent, so they capture style, not subject.
FUNCTION_WORDS = tuple("""
a about above after again against all also although am an and any are as at be because been
before being below between both but by can could did do does doing down during each either
enough even ever every few for from further had has have having he her here hers herself him
himself his how however i if in into is it its itself just least less like may me might more
most much must my myself neither no nor not now of off often on once only or other ought our
ours ourselves out over own perhaps quite rather same shall she should since so some still such
than that the their theirs them themselves then there these they this those though through thus
to too under until up upon us very was we were what when where whether which while who whom
whose why will with within without would yet you your yours yourself
""".split())

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text):
    """Lower-case word tokens."""
    return _WORD.findall(text.lower())


def frequencies(tokens, vocab=FUNCTION_WORDS):
    """Relative frequency (per token) of each vocab word. Empty input -> zeros."""
    n = len(tokens)
    if n == 0:
        return [0.0] * len(vocab)
    counts = dict.fromkeys(vocab, 0)
    for t in tokens:
        if t in counts:
            counts[t] += 1
    return [counts[w] / n for w in vocab]
