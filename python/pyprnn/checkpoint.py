"""Default loader for the upstream bias-free architecture.

A different architecture can supply its own load_model(weights, material_factory)
in another Python module without changing any C++ code or weight assumptions.
"""
import torch
from .prnn import PRNN


def load_model(weights, material_factory):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    features = state["fc1.weight"].shape[1]
    latents = state["fc1.weight"].shape[0]
    outputs = state["fc2.weight"].shape[0]
    if features != outputs or latents % features or latents == 0:
        raise ValueError("Invalid checkpoint feature/material-point dimensions")
    model = PRNN(features, outputs, latents // features,
                 material_factory=material_factory).double()
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model.script_neural_layers()
