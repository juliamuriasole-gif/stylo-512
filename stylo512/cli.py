"""Role: Interface. Command-line entry point."""
import argparse
import json
import pathlib
import re
import sys

from .delta import DeltaModel
from .experiment import accuracy_curve, words_needed
from .features import tokenize
from .synthetic import make_corpus


def strip_gutenberg(text):
    """Drop Project Gutenberg licence header/footer if present (shared boilerplate leaks style)."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    if start:
        text = text[start.end():end.start() if end and end.start() > start.end() else None]
    return text


def load_corpus(path):
    """A directory with one .txt file per author (file stem = author name)."""
    p = pathlib.Path(path)
    files = sorted(p.glob("*.txt"))
    if len(files) < 2:
        raise SystemExit(f"error: {path} needs at least two <author>.txt files")
    return {f.stem: tokenize(strip_gutenberg(f.read_text(encoding="utf-8", errors="ignore"))) for f in files}


def _print_curve(curve, target):
    print(f"{'words':>6}  {'accuracy':>8}  {'trials':>6}  chance")
    for r in curve:
        bar = "#" * int(r["accuracy"] * 30)
        print(f"{r['words']:>6}  {r['accuracy']:>8.1%}  {r['trials']:>6}  {r['chance']:.1%}  {bar}")
    need = words_needed(curve, target)
    print(f"\nwords needed for {target:.0%} accuracy: {need if need else 'not reached'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stylo512", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve", help="accuracy vs sample length on a corpus dir")
    c.add_argument("corpus")
    c.add_argument("--trials", type=int, default=50)
    c.add_argument("--target", type=float, default=0.9)
    c.add_argument("--json", action="store_true")
    a = sub.add_parser("attribute", help="who most likely wrote a text file?")
    a.add_argument("corpus")
    a.add_argument("text")
    d = sub.add_parser("demo", help="offline demo on synthetic authors")
    d.add_argument("--authors", type=int, default=5)
    d.add_argument("--strength", type=float, default=0.5)
    args = ap.parse_args(argv)

    if args.cmd == "curve":
        curve = accuracy_curve(load_corpus(args.corpus), trials=args.trials)
        print(json.dumps(curve, indent=2)) if args.json else _print_curve(curve, args.target)
    elif args.cmd == "attribute":
        model = DeltaModel().fit(load_corpus(args.corpus))
        text = pathlib.Path(args.text).read_text(encoding="utf-8", errors="ignore")
        d = model.deltas(text)
        for author in sorted(d, key=d.get):
            print(f"{d[author]:.3f}  {author}")
        print(f"\nmost likely: {min(d, key=d.get)}  (lower delta = closer style)")
    else:
        corpus = make_corpus(args.authors, style_strength=args.strength)
        print(f"synthetic corpus: {args.authors} authors, style strength {args.strength}\n")
        _print_curve(accuracy_curve(corpus), 0.9)
    return 0


if __name__ == "__main__":
    sys.exit(main())
