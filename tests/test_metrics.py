import numpy as np
import unittest

from lowflow_coherence.metrics import brier_score, relative_skill_percent


class MetricTests(unittest.TestCase):
    def test_brier_score_known_values(self):
        observed = np.array([0, 0, 1, 1])
        probability = np.array([0.0, 0.5, 0.5, 1.0])
        self.assertAlmostEqual(brier_score(observed, probability), 0.125)

    def test_relative_skill_positive_when_candidate_is_better(self):
        self.assertAlmostEqual(relative_skill_percent(0.08, 0.10), 20.0)

    def test_probability_range_is_validated(self):
        with self.assertRaisesRegex(ValueError, "probabilities must lie"):
            brier_score([0, 1], [0.2, 1.2])


if __name__ == "__main__":
    unittest.main()
