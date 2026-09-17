import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from training_pipeline import train_model_from_arrays


class TrainingProtocolTests(unittest.TestCase):
    def test_scaler_is_fit_only_on_training_partition(self):
        # Deliberately make holdout values influence the full-data mean.
        X = np.arange(80, dtype=float).reshape(20, 4)
        y = np.array([0, 1] * 10)
        paths = [f"sample-{i}" for i in range(len(y))]

        result = train_model_from_arrays(X, y, paths, test_size=0.2, random_state=42)

        X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
            X,
            y,
            paths,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        np.testing.assert_allclose(result.scaler.mean_, X_train.mean(axis=0))
        self.assertFalse(np.allclose(result.scaler.mean_, X.mean(axis=0)))
        self.assertEqual(result.metadata["train_samples"], len(X_train))
        self.assertEqual(result.metadata["test_samples"], len(X_test))
        self.assertEqual(result.metadata["evaluation"], "deterministic 80/20 stratified holdout")
        self.assertEqual(result.metadata["random_state"], 42)

    def test_export_metadata_identifies_artifact_lineage(self):
        X = np.arange(120, dtype=float).reshape(30, 4)
        y = np.array([0, 1] * 15)
        paths = [f"sample-{i}" for i in range(len(y))]

        result = train_model_from_arrays(X, y, paths, test_size=0.2, random_state=42)
        self.assertEqual(result.metadata["protocol_version"], 2)
        self.assertEqual(result.metadata["scaler_fit_scope"], "training_partition_only")
        self.assertIn("test_paths_sha256", result.metadata)


if __name__ == "__main__":
    unittest.main()
