"""Unit tests for data acquisition, preprocessing, and EDA pipelines (unittest-compatible)."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.ml_pipeline.tasks.download_data import acquire_data
from src.ml_pipeline.tasks.preprocess import clean_data, load_raw_data, COLUMN_NAMES
from src.ml_pipeline.tasks.eda import run_eda


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

        # Create temporary raw data file with missing values ('?') and multi-class target
        sample_rows = [
            "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n",
            "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n",
            "53.0,0.0,3.0,128.0,216.0,0.0,2.0,115.0,0.0,0.0,1.0,0.0,?,0\n",
            "52.0,1.0,3.0,138.0,223.0,0.0,0.0,169.0,0.0,0.0,1.0,?,3.0,1\n"
        ]
        self.sample_raw_file = self.tmp_path / "sample_raw.data"
        self.sample_raw_file.write_text("".join(sample_rows))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_data_acquisition(self):
        out_file = self.tmp_path / "test_download.data"
        fallback = Path("heart+disease/processed.cleveland.data")
        result = acquire_data(str(out_file), fallback_path=str(fallback))
        self.assertTrue(result.exists())
        self.assertGreater(result.stat().st_size, 0)

    def test_load_raw_data(self):
        df = load_raw_data(str(self.sample_raw_file))
        self.assertEqual(list(df.columns), COLUMN_NAMES)
        self.assertEqual(len(df), 4)
        # Ensure missing values '?' are parsed as NaN
        self.assertEqual(df["ca"].isnull().sum(), 1)
        self.assertEqual(df["thal"].isnull().sum(), 1)

    def test_clean_data(self):
        raw_df = load_raw_data(str(self.sample_raw_file))
        clean_df = clean_data(raw_df)

        # No missing values remaining
        self.assertEqual(clean_df.isnull().sum().sum(), 0)

        # Binary target transformation (0 remains 0, 2 becomes 1)
        self.assertTrue(set(clean_df["target"].unique()).issubset({0, 1}))
        self.assertEqual(clean_df.iloc[1]["target"], 1)
        self.assertEqual(clean_df.iloc[0]["target"], 0)

        # Check discrete column types
        self.assertIn(clean_df["thal"].dtype.name, ["int64", "int32"])

    def test_eda_generation(self):
        clean_csv = "data/processed/heart_disease_cleaned.csv"
        if not os.path.exists(clean_csv):
            self.skipTest("Processed dataset not found.")

        run_eda(clean_csv, str(self.tmp_path))
        self.assertTrue((self.tmp_path / "eda_summary.json").exists())
        self.assertTrue((self.tmp_path / "figures" / "01_class_balance.png").exists())
        self.assertTrue((self.tmp_path / "figures" / "02_numerical_distributions.png").exists())
        self.assertTrue((self.tmp_path / "figures" / "03_categorical_relationships.png").exists())
        self.assertTrue((self.tmp_path / "figures" / "04_correlation_matrix.png").exists())


if __name__ == "__main__":
    unittest.main()
