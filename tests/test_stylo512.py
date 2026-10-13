import io
import contextlib
import pathlib
import tempfile
import unittest

from stylo512.cli import main, strip_gutenberg, load_corpus
from stylo512.delta import DeltaModel
from stylo512.experiment import accuracy_curve, split, words_needed
from stylo512.features import FUNCTION_WORDS, frequencies, tokenize
from stylo512.synthetic import make_corpus


class Features(unittest.TestCase):
    def test_tokenize_lowercases_and_keeps_apostrophes(self):
        self.assertEqual(tokenize("The cat's HAT, it's 2 big!"), ["the", "cat's", "hat", "it's", "big"])

    def test_frequencies_sum_and_empty(self):
        f = frequencies(["the", "the", "cat", "of"])
        self.assertAlmostEqual(f[FUNCTION_WORDS.index("the")], 0.5)
        self.assertAlmostEqual(sum(f), 0.75)
        self.assertEqual(frequencies([]), [0.0] * len(FUNCTION_WORDS))

    def test_vocab_has_no_duplicates(self):
        self.assertEqual(len(FUNCTION_WORDS), len(set(FUNCTION_WORDS)))


class Delta(unittest.TestCase):
    def setUp(self):
        self.corpus = make_corpus(4, n_words=15000, style_strength=1.0)

    def test_needs_two_authors(self):
        with self.assertRaises(ValueError):
            DeltaModel().fit({"a": ["the"] * 10})

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            DeltaModel().predict("the cat")

    def test_attributes_long_heldout_samples(self):
        train, test = split(self.corpus)
        m = DeltaModel().fit(train)
        for a, toks in test.items():
            self.assertEqual(m.predict(toks[:2000]), a)

    def test_accepts_raw_string(self):
        m = DeltaModel().fit(self.corpus)
        self.assertIn(m.predict(" ".join(self.corpus["author_1"][:3000])), self.corpus)

    def test_deltas_cover_all_authors(self):
        m = DeltaModel().fit(self.corpus)
        self.assertEqual(set(m.deltas(self.corpus["author_0"][:500])), set(self.corpus))


class Experiment(unittest.TestCase):
    def test_split_no_overlap(self):
        corpus = {"a": list(range(100)), "b": list(range(100, 200))}
        tr, te = split(corpus, 0.7)
        self.assertEqual(len(tr["a"]), 70)
        self.assertFalse(set(tr["a"]) & set(te["a"]))

    def test_accuracy_increases_with_length_and_beats_chance(self):
        curve = accuracy_curve(make_corpus(5, style_strength=0.5), lengths=(32, 1024), trials=30)
        self.assertLess(curve[0]["accuracy"], curve[1]["accuracy"])
        self.assertGreater(curve[1]["accuracy"], curve[1]["chance"] * 3)

    def test_deterministic_with_seed(self):
        c = make_corpus(3, n_words=8000)
        self.assertEqual(accuracy_curve(c, trials=10, seed=1), accuracy_curve(c, trials=10, seed=1))

    def test_skips_lengths_longer_than_heldout(self):
        c = make_corpus(3, n_words=1000)  # held-out = 300 words
        words = [r["words"] for r in accuracy_curve(c, lengths=(64, 512), trials=5)]
        self.assertEqual(words, [64])

    def test_words_needed(self):
        curve = [{"words": 64, "accuracy": 0.5}, {"words": 256, "accuracy": 0.92}]
        self.assertEqual(words_needed(curve, 0.9), 256)
        self.assertIsNone(words_needed(curve, 0.95))


class Cli(unittest.TestCase):
    def test_strip_gutenberg(self):
        t = "licence\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody text\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nfooter"
        self.assertEqual(strip_gutenberg(t).strip(), "body text")
        self.assertEqual(strip_gutenberg("plain"), "plain")

    def _corpus_dir(self):
        d = tempfile.TemporaryDirectory()
        for a, toks in make_corpus(3, n_words=6000).items():
            pathlib.Path(d.name, a + ".txt").write_text(" ".join(toks))
        return d

    def test_load_corpus_requires_two_files(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "one.txt").write_text("the")
            with self.assertRaises(SystemExit):
                load_corpus(d)

    def test_attribute_and_curve_commands(self):
        with self._corpus_dir() as d:
            text = " ".join(pathlib.Path(d, "author_2.txt").read_text().split()[-1500:])
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
                f.write(text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["attribute", d, f.name]), 0)
                self.assertEqual(main(["curve", d, "--trials", "5", "--json"]), 0)
            self.assertIn("most likely: author_2", out.getvalue())
            self.assertIn('"accuracy"', out.getvalue())

    def test_demo_runs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["demo", "--authors", "3"]), 0)
        self.assertIn("words needed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
