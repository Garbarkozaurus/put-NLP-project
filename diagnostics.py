"""
This module exposes functions which can be used to gain quick and easy insight
into the performance of a spell correcting model.
"""
from difflib import SequenceMatcher
from typing import Callable


def test_single(clean: str, dirty: str, pred: Callable[[str], str]) \
        -> tuple[str, float]:
    """Runs a predictor on a dirty string. Returns the fixed version and its
    similarity to the original, clean text.

    - `clean`: the original text without any noise
    - `dirty`: the original text with noise applied
    - `pred`:  a predictor function whose goal is to transform `dirty` back to
             `clean`
    Returns: A tuple consisting of `predictor`'s output and its similarity to
    `clean` (expressed as a float: 0.0 = no similarity, 1.0 = identical)."""
    denoised = pred(dirty)
    return denoised, SequenceMatcher(None, denoised, clean).ratio()


def test_suite(cleans: list[str], noise: Callable[[str], str],
               pred: Callable[[str], str]) -> None:
    """Intakes text corpora, generates dirty versions, runs everything through
    a predictor and summarizes the results.
    - `cleans`: a list of clean text corpora
    - `noisify`: a noisify function that transforms a clean text into a dirty
      text; need not be deterministic
    - `pred`: a predictor function whose goal is to transform dirty text back
            to its clean equivalent
    """
    # Generate dirty corpora
    dirties = [noise(x) for x in cleans]
    denoised: list[str] = []
    similarities: list[float] = []

    # Prepare dictionary for statistical data
    stats: dict[str, list[tuple[str, str, str, float]]] = {
        "<0.0;0.2)": [],
        "<0.2;0.4)": [],
        "<0.4;0.6)": [],
        "<0.6;0.8)": [],
        "<0.8;1.0)": [],
        "1.0": [],
    }

    # Run through predictor and populate stats
    for clean, dirty in zip(cleans, dirties):
        d, s = test_single(clean, dirty, pred)
        denoised.append(d)
        similarities.append(s)
        key = "1.0"
        if s < 0.2:
            key = "<0.0;0.2)"
        elif s < 0.4:
            key = "<0.2;0.4)"
        elif s < 0.6:
            key = "<0.4;0.6)"
        elif s < 0.8:
            key = "<0.6;0.8)"
        elif s < 1.0:
            key = "<0.8;1.0)"
        stats[key].append((clean, dirty, d, s))

    # Print stats
    print("test_suite statistics:")
    print("----------------------")
    print(f"run on {len(cleans)} corpora")
    for k, v in stats.items():
        print(f"{k:>9}\t{len(v)} ({len(v) / len(cleans) * 100.0:.2f}%)")
