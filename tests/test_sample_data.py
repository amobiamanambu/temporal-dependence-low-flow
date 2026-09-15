from pathlib import Path
import unittest

from lowflow_coherence.analysis import horizon_verification, load_sample_forecasts


ROOT = Path(__file__).resolve().parents[1]


class SampleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_sample_forecasts(
            ROOT / "data" / "sample" / "forecast_predictions_sample.csv.gz"
        )

    def test_compact_sample_schema_and_scope(self):
        self.assertEqual(len(self.frame), 1000)
        self.assertEqual(self.frame["GAGE_ID"].nunique(), 5)
        self.assertEqual(sorted(self.frame["lead_days"].unique().tolist()), [30, 45, 60, 90])

    def test_horizon_summary_is_finite(self):
        summary = horizon_verification(self.frame)
        self.assertEqual(len(summary), 4)
        self.assertTrue(summary["brier_score"].between(0, 1).all())
        self.assertTrue(summary["forecast_initializations"].gt(0).all())


if __name__ == "__main__":
    unittest.main()
