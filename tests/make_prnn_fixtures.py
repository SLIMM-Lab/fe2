#!/usr/bin/env python3
"""Deterministic, untrained weights for integration tests only."""
import importlib.util
from pathlib import Path
import sys
import torch

spec = importlib.util.spec_from_file_location("export_prnn", Path(__file__).parents[1] / "tools/export_prnn.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
for n in (3, 6):
    encoder = torch.cat([torch.eye(n, dtype=torch.float64), 0.7 * torch.eye(n, dtype=torch.float64)])
    encoder[0, 1] = 0.12
    encoder[n + 1, 0] = -0.09
    raw = torch.full((n, 2 * n), -3.0, dtype=torch.float64)
    for i in range(n):
        raw[i, i] = 0.4
        raw[i, n + i] = -0.2
    state = {"fc1.weight": encoder, "fc2.weight": raw}
    torch.save({"model_state_dict": state, "best_state_dict": state}, root / f"fixture{n}.pth")
    exporter.export(root / f"fixture{n}.pth", root / f"fixture{n}.pt")
    if n == 3:
        exporter.export(root / "fixture3.pth", root / "normalized.pt", strain_min=[-1, -2, -3], strain_max=[1, 2, 3])
print(f"Untrained test fixtures written to {root}")
