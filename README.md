# stylo512

> "you can be identified from your writing, with probably as little as 512 words"
> — [@yacineMTB on X](https://x.com/yacineMTB/status/2109846435260940788)

A small research tool that **tests that claim**: it measures how often an author can be
identified from a sample of N words, using Burrows' Delta (function-word frequencies).
Pure Python standard library, no dependencies, no API keys. Research toy only.

## Quick start

```bash
python -m stylo512 demo                         # offline, synthetic authors
python -m unittest discover -v                  # 17 tests
```

### Real authors (public-domain Project Gutenberg texts)

```bash
mkdir corpus
for p in "austen 1342" "dickens 98" "twain 76" "doyle 1661" "wilde 174"; do
  set -- $p; curl -sL "https://www.gutenberg.org/cache/epub/$2/pg$2.txt" -o corpus/$1.txt
done
python -m stylo512 curve corpus                 # accuracy vs sample length
python -m stylo512 attribute corpus mystery.txt # rank authors for one text
```

Corpus format: a folder with one `<author>.txt` per author. Gutenberg licence
headers/footers are stripped automatically.

## Result on 5 classic novelists (one novel each, seed 0, 40 samples/author)

| words | accuracy | chance |
|------:|---------:|-------:|
| 32    | 20.5%    | 20%    |
| 128   | 34.0%    | 20%    |
| 256   | 54.5%    | 20%    |
| **512** | **76.5%** | 20% |
| 1024  | 89.5%    | 20%    |

So with this simple method, 512 words gets you to ~3/4 right among 5 candidates:
well above chance, but not "identified". Stronger features (character n-grams,
more training text per author) would push this higher; the point of the tool is to
measure it rather than assume it. Numbers above come from a single run on one book
per author and will vary with corpus and seed.

## How it works (roles)

| Module | Role |
|---|---|
| `features.py` | Feature extractor: 158 English function words -> relative frequencies |
| `delta.py` | Attributor: Burrows' Delta (z-scored frequencies, mean absolute distance) |
| `experiment.py` | Experimenter: train on first 70% of each author, sample test windows only from the held-out 30% (no leakage) |
| `synthetic.py` | Data generator: fake authors with tunable style strength for tests/demo |
| `cli.py` | Interface: `demo`, `curve`, `attribute` |

## Limitations

- English only; one closed set of candidate authors (it always picks someone).
- `attribute` trains on the whole corpus, so don't test it on text already in the corpus.
- Not a forensic tool. Don't use it to de-anonymise real people.
