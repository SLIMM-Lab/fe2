import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

spec = importlib.util.spec_from_file_location("export_prnn", Path(__file__).parents[1] / "tools/export_prnn.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "weights.pth"
        self.archive = Path(self.temp.name) / "weights.pt"
        self.state = {"fc1.weight": torch.arange(18, dtype=torch.float64).reshape(6, 3) / 20,
                      "fc2.weight": torch.linspace(-2, 1, 18, dtype=torch.float64).reshape(3, 6)}

    def test_round_trip_matches_layers(self):
        torch.save(self.state, self.checkpoint)
        exporter.export(self.checkpoint, self.archive)
        model = torch.jit.load(self.archive)
        eps = torch.tensor([.01, -.03, .08], dtype=torch.float64)
        sig = torch.arange(6, dtype=torch.float64)
        torch.testing.assert_close(model.encode(eps), self.state["fc1.weight"] @ eps)
        torch.testing.assert_close(model.decode(sig), torch.nn.functional.softplus(self.state["fc2.weight"]) @ sig)

    def test_best_state_and_normalization(self):
        torch.save({"model_state_dict": {}, "best_state_dict": self.state}, self.checkpoint)
        model = exporter.export(self.checkpoint, self.archive, "best_state_dict", [-2,-3,-4], [1,2,3])
        eps = torch.tensor([.02,-.01,.03], dtype=torch.float64)
        lo = torch.tensor([-2,-3,-4], dtype=torch.float64)
        hi = torch.tensor([1,2,3], dtype=torch.float64)
        normalized = 2 * (eps-lo) / (hi-lo) - 1
        torch.testing.assert_close(model.encode(eps), self.state["fc1.weight"] @ normalized)

    def test_reject_bias(self):
        self.state["fc1.bias"] = torch.zeros(6)
        torch.save(self.state, self.checkpoint)
        with self.assertRaises(ValueError):
            exporter.export(self.checkpoint, self.archive)

    def test_reject_dimensions_and_nonfinite(self):
        for encoder in [torch.zeros(5,3), torch.full((6,3),float("nan"))]:
            self.state["fc1.weight"] = encoder
            torch.save(self.state, self.checkpoint)
            with self.assertRaises(ValueError):
                exporter.export(self.checkpoint, self.archive)

    def test_reject_degenerate_normalization(self):
        torch.save(self.state, self.checkpoint)
        with self.assertRaises(ValueError):
            exporter.export(self.checkpoint, self.archive, strain_min=[0,0,0], strain_max=[1,0,1])


if __name__ == "__main__":
    unittest.main()
