import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rate_utils import rate_tag


class SummarizePoisonedPipelineTest(unittest.TestCase):
    def write_rate_fixture(self, root, token, teacher_acc, prompt_acc):
        tag = rate_tag(token)
        teacher_root = root / tag / 'GCN'
        prompt_root = root / tag / 'AttrPrompt_dynamic'
        teacher_root.mkdir(parents=True)
        prompt_root.mkdir(parents=True)

        fingerprint = f'graph-{token}'
        split_fingerprint = f'split-{token}'
        teacher = {
            'protocol': {
                'adjacency_fingerprint': fingerprint,
                'split_fingerprint': split_fingerprint,
            },
            'metrics': {
                'test_accuracy_mean_pct': teacher_acc,
            },
        }
        prompt = {
            'protocol': {
                'adjacency_fingerprint': fingerprint,
                'split_fingerprint': split_fingerprint,
            },
            'rates': {
                token: {
                    'teacher_accuracy_mean_pct': teacher_acc,
                    'prompt_accuracy_mean_pct': prompt_acc,
                    'prompt_accuracy_std_pct': 1.0,
                    'prompt_gain_mean_pct': prompt_acc - teacher_acc,
                    'prompt_f1_mean_pct': prompt_acc - 2.0,
                },
            },
            'structure_sanity': {
                'self_only_accuracy_mean_pct': 20.0,
                'first_layer_embedding_l2_delta_mean': 1.5,
                'output_abs_delta_mean': 0.4,
            },
        }
        (teacher_root / 'summary.json').write_text(
            json.dumps(teacher), encoding='utf-8')
        (prompt_root / 'summary.json').write_text(
            json.dumps(prompt), encoding='utf-8')

    def test_each_rate_and_combined_csv_are_distinct_and_protected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.write_rate_fixture(root, '0.05', 60.0, 62.0)
            self.write_rate_fixture(root, '0.10', 50.0, 54.0)

            command = [
                sys.executable,
                str(Path(__file__).with_name(
                    'summarize_poisoned_pipeline.py')),
                '--output_root', str(root),
                '--ptb_rates', '0.05', '0.1',
                '--prompt_type', 'dynamic',
            ]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            first = subprocess.run(
                command, capture_output=True, text=True, env=environment)
            self.assertEqual(first.returncode, 0, first.stderr)

            rate_005 = root / 'M0p05' / 'result_M0p05.csv'
            rate_010 = root / 'M0p10' / 'result_M0p10.csv'
            combined = root / 'poisoned_pipeline_summary.csv'
            self.assertTrue(rate_005.exists())
            self.assertTrue(rate_010.exists())
            self.assertTrue(combined.exists())
            self.assertNotEqual(rate_005, rate_010)

            with combined.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['ptb_rate'] for row in rows],
                             ['0.05', '0.10'])

            second = subprocess.run(
                command, capture_output=True, text=True, env=environment)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('Refusing to overwrite', second.stderr)


if __name__ == '__main__':
    unittest.main()
