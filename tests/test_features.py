from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_dsp_routing.data import (
    AvazuFeaturePipeline,
    chronological_sample,
    chronological_split,
    load_avazu,
    sort_chronologically,
)


def sample_frame(rows: int = 12) -> pd.DataFrame:
    timestamps = pd.date_range("2014-10-21", periods=rows, freq="h")
    frame = pd.DataFrame(
        {
            "id": np.arange(rows, dtype="float64") + 1e12,
            "click": np.arange(rows) % 2,
            "hour": timestamps.strftime("%y%m%d%H").astype("int64"),
            "C1": 1001 + np.arange(rows) % 3,
            "banner_pos": np.arange(rows) % 2,
            "site_category": [f"site_{i % 5}" for i in range(rows)],
            "app_category": [f"app_{i % 4}" for i in range(rows)],
            "device_type": np.arange(rows) % 3,
            "device_conn_type": np.arange(rows) % 2,
            "C15": 300 + (np.arange(rows) % 2) * 20,
            "C16": 50 + (np.arange(rows) % 2) * 40,
            "C18": np.arange(rows) % 4,
            "datetime": timestamps,
            "date": timestamps.date,
            "__index_level_0__": np.arange(rows),
        }
    )
    return frame


class FeatureTests(unittest.TestCase):
    def test_pipeline_is_bounded_and_click_cannot_change_features(self) -> None:
        frame = sample_frame()
        training, evaluation = chronological_split(frame, train_fraction=0.5)
        pipeline = AvazuFeaturePipeline().fit(training)
        original = pipeline.transform(evaluation)
        changed = evaluation.copy()
        changed["click"] = 1 - changed["click"]
        after_label_change = pipeline.transform(changed)

        self.assertLessEqual(original.shape[1], 64)
        self.assertEqual((original != after_label_change).nnz, 0)
        self.assertFalse(any("click" in name for name in pipeline.get_feature_names_out()))

    def test_sort_and_split_are_chronological(self) -> None:
        shuffled = sample_frame().sample(frac=1.0, random_state=7)
        ordered = sort_chronologically(shuffled)
        training, evaluation = chronological_split(shuffled, train_fraction=0.5)
        self.assertTrue(ordered["datetime"].is_monotonic_increasing)
        self.assertLessEqual(training["datetime"].max(), evaluation["datetime"].min())

    def test_systematic_sample_keeps_order_and_full_span(self) -> None:
        frame = sample_frame(12)
        sampled = chronological_sample(frame, 5)
        self.assertEqual(sampled["datetime"].iloc[0], frame["datetime"].iloc[0])
        self.assertEqual(sampled["datetime"].iloc[-1], frame["datetime"].iloc[-1])
        self.assertTrue(sampled["datetime"].is_monotonic_increasing)

    def test_loader_recombines_source_partitions_before_prefix(self) -> None:
        frame = sample_frame(8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.iloc[[6, 0, 4, 2]].to_parquet(
                root / "avazu_train.parquet", index=False
            )
            frame.iloc[[7, 1, 5, 3]].to_parquet(
                root / "avazu_test.parquet", index=False
            )
            loaded = load_avazu(root, n_events=4)
        self.assertListEqual(
            loaded["__index_level_0__"].tolist(),
            [0, 1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
