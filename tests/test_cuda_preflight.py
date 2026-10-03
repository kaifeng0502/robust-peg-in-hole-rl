"""CUDA access checks fail before loading Isaac Sim and work without PyTorch."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_insertion.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("insertion_preflight_under_test", SCRIPT_PATH)
EVALUATOR = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(EVALUATOR)


class CudaPreflightTest(unittest.TestCase):
    def torch_stub(self, available=True, count=2, current=0):
        return SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=Mock(return_value=available),
                device_count=Mock(return_value=count),
                current_device=Mock(return_value=current),
            ),
            empty=Mock(),
        )

    def test_cpu_does_not_require_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            EVALUATOR.validate_cuda_device("cpu")

    def test_cuda_requires_configured_pytorch_environment(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "configured Isaac Lab Python"):
                EVALUATOR.validate_cuda_device("cuda:0")

    def test_unavailable_gpu_has_access_diagnostic_and_never_allocates(self):
        for available, count in ((False, 0), (False, 2), (True, 0)):
            with self.subTest(available=available, count=count):
                torch = self.torch_stub(available, count)
                with patch.dict(sys.modules, {"torch": torch}):
                    with self.assertRaisesRegex(RuntimeError, "No accessible CUDA GPU.*restart the Pod/container"):
                        EVALUATOR.validate_cuda_device("cuda:0")
                torch.empty.assert_not_called()

    def test_invalid_index_is_distinct_from_lost_device_access(self):
        torch = self.torch_stub(count=1)
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(ValueError, "Invalid CUDA device index 1.*0 through 0"):
                EVALUATOR.validate_cuda_device("cuda:1")
        torch.empty.assert_not_called()

    def test_malformed_cuda_index_is_rejected_before_import(self):
        for device in ("cuda:-1", "cuda:x", "cuda:", "cuda:1:2"):
            with self.subTest(device=device), patch.dict(sys.modules, {"torch": None}):
                with self.assertRaisesRegex(ValueError, "Invalid CUDA device"):
                    EVALUATOR.validate_cuda_device(device)

    def test_requested_device_is_initialized(self):
        torch = self.torch_stub(count=2)
        with patch.dict(sys.modules, {"torch": torch}):
            EVALUATOR.validate_cuda_device("cuda:1")
        torch.empty.assert_called_once_with(1, device="cuda:1")
        torch.cuda.current_device.assert_not_called()

    def test_bare_cuda_uses_current_device(self):
        torch = self.torch_stub(count=2, current=1)
        with patch.dict(sys.modules, {"torch": torch}):
            EVALUATOR.validate_cuda_device("cuda")
        torch.empty.assert_called_once_with(1, device="cuda:1")

    def test_visible_but_inaccessible_device_is_reported(self):
        torch = self.torch_stub()
        failure = RuntimeError("CUDA initialization failed")
        torch.empty.side_effect = failure
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "reported available but could not allocate") as caught:
                EVALUATOR.validate_cuda_device("cuda:0")
        self.assertIs(caught.exception.__cause__, failure)

    def test_gpu_query_failure_preserves_cause(self):
        torch = self.torch_stub()
        failure = OSError("device access denied")
        torch.cuda.is_available.side_effect = failure
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "could not query GPU access") as caught:
                EVALUATOR.validate_cuda_device("cuda:0")
        self.assertIs(caught.exception.__cause__, failure)
        torch.empty.assert_not_called()

    def test_main_checks_gpu_before_importing_simulator_or_creating_output(self):
        args = SimpleNamespace(cases=Path("unused.json"), device="cuda:0")
        torch = self.torch_stub(available=False, count=0)
        with (
            patch.object(EVALUATOR, "parse_args", return_value=args),
            patch.object(EVALUATOR, "read_case_set", return_value={}),
            patch.dict(sys.modules, {"torch": torch, "isaaclab": None, "isaaclab.app": None}),
        ):
            with self.assertRaisesRegex(RuntimeError, "No accessible CUDA GPU"):
                EVALUATOR.main([])


if __name__ == "__main__":
    unittest.main()
