#!/usr/bin/env python3
import diagnostics
import unittest
from context_BERT import clean_and_noisy_lists
import os
import tempfile


class DiagnosticsTests(unittest.TestCase):
    def test_test_single(self) -> None:
        # Awful predictor
        def mock_awful_pred(s: str) -> str:
            return "x" * len(s)
        clean = "unlimited sleep deprivation"
        noisy = clean.replace("a", "b")
        denoised, sim = diagnostics.test_single(clean, noisy, mock_awful_pred)
        self.assertNotEqual(denoised, clean)
        self.assertLess(sim, 0.1)

        # High similarity, but prediction not perfect
        def mock_good_pred(s: str) -> str:
            return s.replace("b", "a")[:-1] + "j"
        clean = "dankest memes and vaporwave"
        noisy = clean.replace("a", "b")
        denoised, sim = diagnostics.test_single(clean, noisy, mock_good_pred)
        self.assertNotEqual(denoised, clean)
        self.assertLess(sim, 1.0)
        self.assertGreater(sim, 0.9)

        # Perfect prediction
        def mock_perf_pred(s: str) -> str:
            return s.replace("b", "a")
        clean = "procrastinate at 5am forever that shit is my jam"
        noisy = "procrbstinbte bt 5bm forever thbt shit is my jbm"
        denoised, sim = diagnostics.test_single(clean, noisy, mock_perf_pred)
        self.assertEqual(denoised, clean)
        self.assertEqual(sim, 1.0)


def create_temp_file() -> tempfile._TemporaryFileWrapper:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    sample_text = """This is a sample of clean text used in unit testing.
    It serves no purpose beyond being a short, generic example of natural
    language to test the process of loading and noisifying project data.
    The quick brown fox jumped over the lazy dog. Even more text for more
    accurate randomness. Often we want to create temporary files to save data
    that we can't hold in memory or to pass to external programs that must
    read from a file. The obvious way to do this is to generate a unique file
    name in a common system temporary directory.
    I work as an employee for the friendly neighbourhood department stores,
    and I get home every day by 8 PM at the latest. I don't smoke, but I
    occasionally drink. I'm in bed by 11 PM, and make sure I get eight hours
    of sleep, no matter what. After having a glass of warm milk and doing
    about twenty minutes of stretches before going to bed, I usually have no
    problems sleeping until morning. Just like a baby, I wake up without any
    fatigue or stress in the morning. I was told there were no issues at my
    last check-up. I'm trying to explain that I'm a person who wishes to live
    a very quiet life. I take care not to trouble myself with any enemies,
    like winning and losing, that would cause me to lose sleep at night. That
    is how I deal with society, and I know that is what brings me happiness.
    Although, if I were to fight I wouldn't lose to anyone."""
    sample_text *= 10
    tmp.write(bytes(sample_text.lower(), encoding="utf-8"))
    tmp.close()
    return tmp


class LoadingTests(unittest.TestCase):
    def test_num_words_match(self) -> None:
        rand_seed = 1
        tmp_file = create_temp_file()
        clean_words, noisy_words = clean_and_noisy_lists(tmp_file.name,
                                                         noise_seed=rand_seed)
        os.remove(tmp_file.name)
        clean_len: int = len(clean_words)
        noisy_len: int = len(noisy_words)
        self.assertEqual(clean_len, noisy_len,
                         f"Clean and noisy lists have different lengths: \
                           {clean_len} vs. {noisy_len}")

    def test_word_lengths_match(self) -> None:
        rand_seed = 1
        tmp_file = create_temp_file()
        clean_words, noisy_words = clean_and_noisy_lists(tmp_file.name,
                                                         noise_seed=rand_seed)
        os.remove(tmp_file.name)
        different_len_count: int = 0
        different_len_words: list[tuple[str, str]] = []
        for clean_word, noisy_word in zip(clean_words, noisy_words):
            if len(clean_word) != len(noisy_word):
                different_len_count += 1
                different_len_words.append((clean_word, noisy_word))
        self.assertEqual(different_len_count, 0, f"{different_len_count} \
                        words have different lengths: \
                        {different_len_words}")

    def test_randomness_as_specified(self) -> None:
        rand_seed = 1
        NOISE_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
        tolerance = 0.005
        for noise_level in NOISE_LEVELS:
            tmp_file = create_temp_file()
            clean_words, noisy_words = \
                clean_and_noisy_lists(tmp_file.name,
                                      random_char_prob=noise_level,
                                      noise_seed=rand_seed)
            os.remove(tmp_file.name)

            different_char_count: int = 0
            all_char_count = sum([len(word) for word in clean_words])
            for clean_word, noisy_word in zip(clean_words, noisy_words):
                diff = sum(1 for x, y in zip(clean_word, noisy_word) if
                           x.lower() != y.lower())
                different_char_count += diff
            calc_prob = different_char_count / all_char_count
            self.assertLessEqual(abs(noise_level-calc_prob), tolerance,
                                 "Incorrect randomness. Desired: "
                                 + f"{noise_level} calculated: {calc_prob} "
                                 + f"(tolerance is ±{tolerance})")


if __name__ == "__main__":
    unittest.main()
