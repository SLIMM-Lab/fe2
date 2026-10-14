"""Small, architecture-independent API used by the embedded C++ driver."""
import numpy as np
import torch


def evaluate_with_tangent(model, strain):
    with torch.enable_grad():
        x = torch.as_tensor(np.asarray(strain), dtype=torch.float64).clone().requires_grad_(True)
        stress = model.evaluate(x)
        if stress.shape != x.shape:
            raise ValueError("Macro stress must have the same component count as strain")
        rows = [torch.autograd.grad(stress[i], x, retain_graph=i+1 < stress.numel())[0]
                for i in range(stress.numel())]
        tangent = torch.stack(rows)
    stress = stress.detach().numpy().copy()
    tangent = tangent.detach().numpy().copy()
    if not np.isfinite(stress).all() or not np.isfinite(tangent).all():
        raise ValueError("Nonfinite Python PRNN response")
    return stress, tangent


def observe(model, strain):
    with torch.no_grad():
        stress = model.observe(torch.as_tensor(np.asarray(strain), dtype=torch.float64)).detach().numpy().copy()
    if not np.isfinite(stress).all():
        raise ValueError("Nonfinite Python PRNN stress")
    return stress
