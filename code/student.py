"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import GPT


class GatedMLP(nn.Module):
    """Parameter-matched SwiGLU: SiLU(gate(x)) * value(x), then project."""
    def __init__(self, width):
        super().__init__()
        hidden = round((8 * width / 3) / 8) * 8
        self.gate_value = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class StudentGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        if config.get('gated', True):
            for block in self.blocks:
                block.mlp = GatedMLP(config['width'])
                block.mlp.apply(self.initialize)


def build_model(config):
    return StudentGPT(config)
