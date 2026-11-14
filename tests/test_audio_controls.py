import unittest

import numpy as np

from src.main import apply_master_gain, change_playback_speed


class AudioControlTests(unittest.TestCase):
    def test_speed_control_changes_duration_and_keeps_unity_unchanged(self):
        samples = np.sin(np.linspace(0, 20 * np.pi, 1000))

        slower = change_playback_speed(samples, 0.5)
        faster = change_playback_speed(samples, 2.0)

        self.assertEqual(len(slower), 2000)
        self.assertEqual(len(faster), 500)
        np.testing.assert_array_equal(change_playback_speed(samples, 1.0), samples)

    def test_speed_control_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            change_playback_speed(np.ones(100), 0)

    def test_master_gain_adjusts_amplitude_without_clipping(self):
        samples = np.array([-0.25, 0.5, 1.0]) * 0.75

        quiet = apply_master_gain(samples, -6.0)
        louder = apply_master_gain(samples, 2.0)

        self.assertAlmostEqual(float(np.max(np.abs(quiet))), 0.75 * 10 ** (-6 / 20), places=6)
        self.assertAlmostEqual(float(np.max(np.abs(louder))), 0.75 * 10 ** (2 / 20), places=6)
        self.assertLessEqual(float(np.max(np.abs(louder))), 1.0)


if __name__ == '__main__':
    unittest.main()
