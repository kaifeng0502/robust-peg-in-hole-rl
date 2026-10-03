import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_insertion"))

from evaluation_metrics import EpisodeMetrics, EvaluationCriteria, summarize_trials, wilson95


class EpisodeMetricsTest(unittest.TestCase):
    def make_episode(self, **overrides):
        return EpisodeMetrics(EvaluationCriteria(dt_s=1.0 / 15.0, **overrides), target_depth_m=0.0245)

    def add_success(self, episode, samples):
        for _ in range(samples):
            episode.update(0.0, 0.0245, 2.0, 1.5)

    def test_transient_success_then_retraction_is_failure(self):
        episode = self.make_episode()
        self.add_success(episode, 2)
        episode.update(0.0, 0.0, -1.0, 2.0)
        result = episode.result()
        self.assertTrue(result["ever_success"])
        self.assertFalse(result["final_success"])
        self.assertFalse(result["held_success"])
        self.assertFalse(result["had_hold_anytime"])
        self.assertIsNone(result["success_time_s"])
        self.assertEqual(result["failure_category"], "lost_insertion")
        self.assertEqual(result["max_depth_m"], 0.0245)
        self.assertEqual(result["peak_commanded_force_n"], 2.0)
        self.assertEqual(result["episode_reward"], 3.0)

    def test_separated_success_intervals_do_not_accumulate_into_hold(self):
        episode = self.make_episode()
        self.add_success(episode, 10)
        episode.update(0.01, 0.0245, 0.0, 1.0)
        self.add_success(episode, 10)
        result = episode.result()
        self.assertTrue(result["final_success"])
        self.assertFalse(result["held_success"])
        self.assertFalse(result["had_hold_anytime"])
        self.assertEqual(result["failure_category"], "insufficient_hold")
        self.assertAlmostEqual(result["trailing_hold_duration_s"], 10.0 / 15.0)

    def test_failure_inside_interval_breaks_hold_even_if_final_geometry_succeeds(self):
        episode = self.make_episode()
        self.add_success(episode, 14)
        episode.update(0.0, 0.0245, 1.0, 2.0, interval_success=False, interval_ever_success=True)
        result = episode.result()
        self.assertTrue(result["ever_success"])
        self.assertTrue(result["final_success"])
        self.assertFalse(result["held_success"])
        self.assertEqual(result["trailing_hold_duration_s"], 0.0)
        self.add_success(episode, 15)
        self.assertEqual(episode.result()["success_time_s"], 2.0)

    def test_success_only_inside_interval_counts_as_ever_not_final(self):
        episode = self.make_episode()
        episode.update(0.0, 0.0, 1.0, 2.0, interval_success=False, interval_ever_success=True)
        result = episode.result()
        self.assertTrue(result["ever_success"])
        self.assertFalse(result["final_success"])
        self.assertFalse(result["held_success"])
        self.assertEqual(result["failure_category"], "lost_insertion")
        self.assertAlmostEqual(result["first_success_time_s"], 1.0 / 15.0)

    def test_interval_flags_do_not_override_failed_final_geometry(self):
        episode = self.make_episode()
        episode.update(0.0, 0.0, 1.0, 2.0, interval_success=True)
        self.assertFalse(episode.result()["final_success"])
        self.assertFalse(episode.result()["held_success"])

    def test_exact_one_second_at_fifteen_hz_qualifies(self):
        episode = self.make_episode()
        self.add_success(episode, 14)
        self.assertFalse(episode.result()["held_success"])
        self.add_success(episode, 1)
        result = episode.result()
        self.assertEqual(episode.criteria.required_hold_samples, 15)
        self.assertTrue(result["held_success"])
        self.assertEqual(result["trailing_hold_duration_s"], 1.0)
        self.assertEqual(result["success_time_s"], 1.0)
        self.assertAlmostEqual(result["first_success_time_s"], 1.0 / 15.0)

    def test_hold_lost_at_horizon_is_not_held_success(self):
        episode = self.make_episode()
        self.add_success(episode, 15)
        episode.update(0.0, -0.002, 0.0, 1.0)
        result = episode.result()
        self.assertTrue(result["had_hold_anytime"])
        self.assertEqual(result["first_hold_time_s"], 1.0)
        self.assertFalse(result["held_success"])
        self.assertIsNone(result["success_time_s"])

    def test_regained_hold_completion_time_uses_final_run(self):
        episode = self.make_episode()
        self.add_success(episode, 15)
        episode.update(0.0, 0.0, 0.0, 1.0)
        self.add_success(episode, 20)
        result = episode.result()
        self.assertEqual(result["first_hold_time_s"], 1.0)
        self.assertAlmostEqual(result["success_time_s"], 31.0 / 15.0)
        self.assertAlmostEqual(result["trailing_hold_duration_s"], 20.0 / 15.0)

    def test_depth_boundary_is_inclusive_and_excess_depth_fails(self):
        for depth in (0.0235, 0.0255):
            with self.subTest(depth=depth):
                episode = self.make_episode()
                episode.update(0.0025, depth, 0.0, 0.0)
                self.assertTrue(episode.result()["final_success"])
        episode = self.make_episode()
        episode.update(0.0, 0.026, 0.0, 0.0)
        self.assertFalse(episode.result()["final_success"])
        self.assertEqual(episode.result()["failure_category"], "excessive_depth")
        episode = self.make_episode()
        episode.update(0.0025001, 0.0245, 0.0, 0.0)
        self.assertEqual(episode.result()["failure_category"], "misalignment")

    def test_nonfinite_sample_is_rejected_without_partial_update(self):
        for index in range(4):
            for invalid in (math.nan, math.inf, -math.inf):
                episode = self.make_episode()
                self.add_success(episode, 1)
                before = episode.result()
                sample = [0.0, 0.0245, 1.0, 1.0]
                sample[index] = invalid
                with self.subTest(index=index, value=invalid):
                    with self.assertRaises(ValueError):
                        episode.update(*sample)
                    self.assertEqual(episode.result(), before)

    def test_invalid_criteria_and_empty_episode_are_rejected(self):
        for key in ("dt_s", "hold_duration_s", "xy_tolerance_m", "depth_tolerance_m"):
            for invalid in (0.0, -1.0, math.nan, math.inf):
                kwargs = {"dt_s": 1.0 / 15.0, key: invalid}
                with self.subTest(key=key, value=invalid):
                    with self.assertRaises(ValueError):
                        EvaluationCriteria(**kwargs)
        with self.assertRaises(ValueError):
            self.make_episode().result()
        with self.assertRaises(ValueError):
            EpisodeMetrics(EvaluationCriteria(dt_s=0.1), target_depth_m=-1.0)

    def test_non_integral_hold_duration_rounds_up(self):
        episode = self.make_episode(hold_duration_s=1.01)
        self.add_success(episode, 15)
        self.assertFalse(episode.result()["held_success"])
        self.add_success(episode, 1)
        self.assertTrue(episode.result()["held_success"])


class SummaryMetricsTest(unittest.TestCase):
    def make_row(self, success):
        episode = EpisodeMetrics(EvaluationCriteria(dt_s=0.5), target_depth_m=0.0245)
        for _ in range(2):
            episode.update(0.0, 0.0245 if success else 0.0, 1.0, 2.0)
        return episode.result()

    def test_wilson_zero_and_all_success(self):
        lower, upper = wilson95(0, 100)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 0.03699349820698568)
        all_lower, all_upper = wilson95(100, 100)
        self.assertEqual(all_upper, 1.0)
        self.assertAlmostEqual(all_lower, 1.0 - upper)
        for successes, trials in ((0, 0), (-1, 2), (3, 2), (1.5, 2)):
            with self.assertRaises(ValueError):
                wilson95(successes, trials)

    def test_summary_counts_terminal_hold_and_times_only_successes(self):
        episode = EpisodeMetrics(EvaluationCriteria(dt_s=0.5), target_depth_m=0.0245)
        for _ in range(2):
            episode.update(0.0, 0.0245, 1.0, 2.0)
        episode.update(0.0, 0.0, 0.0, 3.0)
        failed_after_hold = episode.result()
        result = summarize_trials([self.make_row(True), self.make_row(False), failed_after_hold])
        self.assertEqual(result["success_rate"], 1.0 / 3.0)
        self.assertEqual(result["ever_success_count"], 2)
        self.assertEqual(result["had_hold_anytime_count"], 2)
        self.assertEqual(result["final_success_count"], 1)
        self.assertEqual(result["held_success_count"], 1)
        self.assertEqual(result["mean_success_time_s"], 1.0)
        self.assertEqual(result["median_success_time_s"], 1.0)
        self.assertEqual(result["max_peak_commanded_force_n"], 3.0)
        self.assertEqual(result["failure_counts"], {"insufficient_depth": 1, "lost_insertion": 1})

    def test_all_failures_have_no_completion_time(self):
        result = summarize_trials([self.make_row(False)])
        self.assertIsNone(result["mean_success_time_s"])
        self.assertIsNone(result["median_success_time_s"])
        self.assertEqual(result["success_rate"], 0.0)

    def test_empty_and_nonfinite_summary_rejected(self):
        with self.assertRaises(ValueError):
            summarize_trials([])
        for key in ("episode_reward", "max_depth_m", "peak_commanded_force_n", "success_time_s"):
            row = self.make_row(True)
            row[key] = math.nan
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    summarize_trials([row])


if __name__ == "__main__":
    unittest.main()
