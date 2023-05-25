#!/usr/bin/env python3
import diagnostics
import unittest


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


if __name__ == "__main__":
    unittest.main()
