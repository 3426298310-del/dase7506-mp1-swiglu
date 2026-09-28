"""The disabled gate must reproduce the classroom baseline exactly."""
import unittest
import torch
from model import GPT
from student import build_model


class AblationTests(unittest.TestCase):
    def test_disabled_gate_is_exact_baseline(self):
        config = dict(vocab=2048, width=32, heads=4, depth=2, context=256, gated=False)
        torch.manual_seed(17)
        baseline = GPT(config).eval()
        torch.manual_seed(17)
        ablation = build_model(config).eval()
        x = torch.tensor([[1, 17, 9, 25]])
        with torch.no_grad():
            torch.testing.assert_close(baseline(x), ablation(x), atol=0, rtol=0)
