#!/usr/bin/env python3
"""Export pyprnn weights, without tracing its state-resetting sequence forward.

Accepts a state_dict or utils.Trainer.save() checkpoint. Only the two
bias-free layers in SLIMM-Lab/pyprnn are supported. No training is performed.
"""

import argparse
from pathlib import Path
import torch


class OnlineWeights(torch.nn.Module):
    def __init__(self, encoder, decoder, scale, offset):
        super().__init__()
        self.register_buffer("format_version", torch.tensor(1, dtype=torch.int64))
        self.register_buffer("encoder_weight", encoder)
        # Preserve raw SoftLayer weights: C++ applies softplus exactly once.
        self.register_buffer("decoder_weight", decoder)
        self.register_buffer("strain_scale", scale)
        self.register_buffer("strain_offset", offset)

    @torch.jit.export
    def encode(self, strain: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            strain * self.strain_scale + self.strain_offset, self.encoder_weight
        )

    @torch.jit.export
    def decode(self, stress: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            stress, torch.nn.functional.softplus(self.decoder_weight)
        )

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        return self.encode(strain)


def export(checkpoint, output, state_key="model_state_dict", strain_min=None, strain_max=None):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = data[state_key] if state_key in data else data
    if set(state) != {"fc1.weight", "fc2.weight"}:
        raise ValueError("Expected only bias-free fc1.weight and fc2.weight; select the correct state key")
    encoder = state["fc1.weight"].detach().to(dtype=torch.float64).contiguous()
    decoder = state["fc2.weight"].detach().to(dtype=torch.float64).contiguous()
    if encoder.ndim != 2 or decoder.ndim != 2:
        raise ValueError("Weights must be matrices")
    latent, features = encoder.shape
    if features not in (3, 6) or latent == 0 or latent % features or decoder.shape != (features, latent):
        raise ValueError("Expected 3 or 6 features/outputs and whole latent material points")
    if not torch.isfinite(encoder).all() or not torch.isfinite(decoder).all():
        raise ValueError("Weights must be finite")
    scale, offset = torch.ones(features, dtype=torch.float64), torch.zeros(features, dtype=torch.float64)
    if (strain_min is None) != (strain_max is None):
        raise ValueError("Supply both strain-min and strain-max")
    if strain_min is not None:
        lo, hi = torch.tensor(strain_min, dtype=torch.float64), torch.tensor(strain_max, dtype=torch.float64)
        if lo.shape != (features,) or hi.shape != (features,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not (hi > lo).all():
            raise ValueError("Normalization bounds must be finite and increasing, one per feature")
        scale = 2.0 / (hi - lo)
        offset = -1.0 - lo * scale
    module = torch.jit.script(OnlineWeights(encoder, decoder, scale, offset).eval())
    module.save(str(output))
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state-key", default="model_state_dict",
                        help="Use best_state_dict to select Trainer's best validation weights")
    parser.add_argument("--strain-min", type=float, nargs="+")
    parser.add_argument("--strain-max", type=float, nargs="+")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.state_key, args.strain_min, args.strain_max)


if __name__ == "__main__":
    main()
