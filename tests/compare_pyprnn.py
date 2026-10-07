#!/usr/bin/env python3
"""Compare accepted online increments with upstream pyprnn's sequence forward.

Usage: python compare_pyprnn.py PYPRNN_ROOT CHECKPOINT EXPORTED_WEIGHTS
Requires a built ./prnn_material_test in this directory; no training.
"""
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import torch

root, checkpoint, weights = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from prnn import PRNN

torch.set_default_dtype(torch.float64)
state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
n = state["fc1.weight"].shape[0] // 3
model = PRNN(n_features=3, n_outputs=3, n_matpts=n)
model.load_state_dict(state)
# Load, unload, reload, reverse, with coupled shear and transverse strain.
levels = np.r_[np.linspace(0, .025, 20), np.linspace(.025, -.012, 25), np.linspace(-.012, .03, 25)]
path = np.stack([levels, -.17*levels, .31*levels], axis=1)
with torch.no_grad():
    expected = model(torch.tensor(path).unsqueeze(0))[0].numpy()
result = subprocess.run(
    [str(Path(__file__).parent.resolve() / "prnn_material_test"), "--replay", str(weights.resolve())],
    input="\n".join(" ".join(map(str,row)) for row in path), text=True,
    capture_output=True, check=True,
    env={**os.environ, "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"},
)
actual = np.loadtxt(result.stdout.splitlines())
assert actual.shape == expected.shape
np.testing.assert_allclose(actual, expected, rtol=5.e-5, atol=2.e-3)
print(f"Upstream PRNN parity passed for {len(path)} accepted increments; max stress error {np.max(np.abs(actual-expected)):.6g}")
