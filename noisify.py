"""
This is a small module for adding noise to text samples. It can be used to
"noisify" any string written in english, to then try to revert this process
with some NLP model.
"""

from numpy.random import Generator, default_rng
from typing import Optional

_ALPHABET = set([c for c in "abcdefghijklmnopqrstuvwxyz"])
_RND_GENERATOR = default_rng()


def noisify(text: str, prob: float = 0.3, *, seed: Optional[int] = None) \
        -> str:
    """Simplest noisifier. Randomizes each character in `text` with
    probability `prob`.

    By default, the module uses its private `numpy.random.Generator` to
    guarantee independent randomness. Reproducibility can be forced by setting
    a fixed `seed`."""

    assert text.islower(), "only lowercase text is supported"
    rng: Generator = _RND_GENERATOR if seed is None else default_rng(seed)

    result: list[str] = []
    for c in text:
        if rng.random() < prob and c != ' ':
            c_new = rng.choice(list(_ALPHABET - set((c, ' '))), 1)[0]
            result.append(c_new)
            continue
        result.append(c)
    return "".join(result)
