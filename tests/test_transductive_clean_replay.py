import tempfile
import unittest
from pathlib import Path

import run_citeseer_svd100_to_cora_svd100_transductive_2bb_3methods_corrected_budget_180 as controller


class CleanReplayAuditTest(unittest.TestCase):
    @staticmethod
    def _config_context(replay, device_name="GPU-A", target_svd="target-svd"):
        return {
            "code_sha256": "code",
            "target_receipt": {"reduced_x_sha256": target_svd},
            "target_cache_file_sha256": "target-cache",
            "target_clean_replay": replay,
            "runtime_environment": {
                "torch_version": "2.8.0",
                "torch_geometric_version": "2.7.0",
                "device_name": device_name,
            },
        }

    def test_all_frozen_receipts_map_to_integer_counts(self):
        expected = {
            "peak_bb": (142, 1280),
            "stable_bb": (137, 1348),
        }

        for backbone in controller.BACKBONES:
            with self.subTest(backbone=backbone["id"]):
                self.assertEqual(
                    controller._accuracy_to_correct_count(
                        backbone["clean_val_accuracy"], "val"
                    ),
                    expected[backbone["id"]][0],
                )
                self.assertEqual(
                    controller._accuracy_to_correct_count(
                        backbone["clean_test_accuracy"], "test"
                    ),
                    expected[backbone["id"]][1],
                )

    def test_exact_correct_counts_pass(self):
        result = controller.compare_clean_replay_metrics(
            "0.516981", "0.559801", 137 / 265, 1348 / 2408
        )

        self.assertEqual(result["val_correct_node_drift"], 0)
        self.assertEqual(result["test_correct_node_drift"], 0)
        self.assertEqual(result["status"], "historical_correct_counts_exact")

    def test_two_report_only_test_nodes_pass(self):
        result = controller.compare_clean_replay_metrics(
            "0.516981", "0.559801", 137 / 265, 1350 / 2408
        )

        self.assertEqual(result["val_correct_node_drift"], 0)
        self.assertEqual(result["test_correct_node_drift"], 2)
        self.assertEqual(result["test_correct_node_delta"], 2)
        self.assertEqual(
            result["status"],
            "historical_reference_drift_recorded",
        )

    def test_negative_two_report_only_test_nodes_pass(self):
        result = controller.compare_clean_replay_metrics(
            "0.516981", "0.559801", 137 / 265, 1346 / 2408
        )

        self.assertEqual(result["test_correct_node_drift"], 2)
        self.assertEqual(result["test_correct_node_delta"], -2)

    def test_larger_cross_runtime_test_drift_is_recorded(self):
        result = controller.compare_clean_replay_metrics(
            "0.516981", "0.559801", 137 / 265, 1360 / 2408
        )

        self.assertEqual(result["test_correct_node_delta"], 12)
        self.assertEqual(result["status"], "historical_reference_drift_recorded")

    def test_validation_drift_is_recorded_without_reselecting_checkpoint(self):
        result = controller.compare_clean_replay_metrics(
            "0.516981", "0.559801", 138 / 265, 1348 / 2408
        )

        self.assertEqual(result["val_correct_node_delta"], 1)
        self.assertEqual(result["status"], "historical_reference_drift_recorded")

    def test_non_count_accuracy_still_fails_receipt_validation(self):
        with self.assertRaisesRegex(RuntimeError, "does not map to an integer"):
            controller.compare_clean_replay_metrics(
                "0.516981", "0.559801", 0.5, 1348 / 2408
            )

    def test_config_hash_ignores_runtime_observation_but_anchors_runtime(self):
        exact = self._config_context([{"test_correct_node_delta": 0}])
        drifted = self._config_context([{"test_correct_node_delta": 12}])
        other_runtime = self._config_context(
            [{"test_correct_node_delta": 12}], device_name="GPU-B"
        )
        other_target_svd = self._config_context(
            [{"test_correct_node_delta": 12}], target_svd="other-target-svd"
        )

        self.assertEqual(
            controller.config_sha256(exact), controller.config_sha256(drifted)
        )
        self.assertNotEqual(
            controller.config_sha256(exact), controller.config_sha256(other_runtime)
        )
        self.assertNotEqual(
            controller.config_sha256(exact), controller.config_sha256(other_target_svd)
        )

    def test_replay_receipt_atomically_keeps_latest_runtime_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "target_svd_clean_replay.tsv"
            controller.atomic_write_text(receipt, "test_delta\t0\n")
            controller.atomic_write_text(receipt, "test_delta\t2\n")

            self.assertEqual(receipt.read_text(encoding="utf-8"), "test_delta\t2\n")


if __name__ == "__main__":
    unittest.main()
