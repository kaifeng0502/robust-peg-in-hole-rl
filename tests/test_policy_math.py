import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_insertion"))

from policy_math import SpiralSchedule, bounded_target


class SpiralScheduleTest(unittest.TestCase):
    def test_starts_at_center(self):
        self.assertEqual(SpiralSchedule().offset(0.0), (0.0, 0.0))

    def test_radius_grows_and_saturates(self):
        schedule = SpiralSchedule(radial_speed_m_s=0.001, turns_per_second=1.0, max_radius_m=0.004)
        for t, expected in [(1.0, 0.001), (3.0, 0.003), (10.0, 0.004)]:
            x, y = schedule.offset(t)
            self.assertAlmostEqual(math.hypot(x, y), expected, places=12)

    def test_negative_time_is_clamped(self):
        self.assertEqual(SpiralSchedule().offset(-1.0), (0.0, 0.0))

    def test_invalid_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            SpiralSchedule(radial_speed_m_s=0.0).offset(1.0)


class BoundedTargetTest(unittest.TestCase):
    def test_clamps_both_directions(self):
        self.assertEqual(bounded_target(1.0, 3.0, 0.25), 1.25)
        self.assertEqual(bounded_target(1.0, -3.0, 0.25), 0.75)


if __name__ == "__main__":
    unittest.main()
