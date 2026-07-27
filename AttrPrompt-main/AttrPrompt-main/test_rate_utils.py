import unittest

from rate_utils import (
    canonical_rate_tokens,
    rate_tag,
    rate_token,
)


class RateUtilsTest(unittest.TestCase):
    def test_dataset_tokens_use_two_decimals(self):
        self.assertEqual(rate_token('0'), '0.00')
        self.assertEqual(rate_token('0.1'), '0.10')
        self.assertEqual(rate_token('0.10'), '0.10')
        self.assertEqual(rate_token('0.2'), '0.20')

    def test_equivalent_spellings_share_one_directory_tag(self):
        self.assertEqual(rate_tag('0.1'), 'M0p10')
        self.assertEqual(rate_tag('0.10'), 'M0p10')

    def test_duplicate_aliases_are_rejected(self):
        with self.assertRaisesRegex(
                ValueError, 'Duplicate perturbation rate'):
            canonical_rate_tokens(['0.1', '0.10'])

    def test_default_sweep_has_unique_tokens(self):
        self.assertEqual(
            canonical_rate_tokens(
                ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25']),
            ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25'],
        )


if __name__ == '__main__':
    unittest.main()
