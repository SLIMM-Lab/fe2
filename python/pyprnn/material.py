"""Autograd boundary around the generic C++ Material interface.

The native tangent differentiates the current strain at fixed accepted history.
For training, replay supplies the missing dependence on earlier strains. This
is first-order differentiation; no constitutive parameter gradients are assumed.
"""
import torch
from torch.autograd.function import once_differentiable


def numpy(tensor):
    return tensor.detach().contiguous().numpy()


class MaterialUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, strain, history, native, history_step):
        stress, tangent = native.update(numpy(strain))
        ctx.save_for_backward(torch.from_numpy(tangent), strain.detach(), history.detach())
        ctx.native = native
        ctx.history_step = history_step
        return torch.from_numpy(stress)

    @staticmethod
    @once_differentiable
    def backward(ctx, adjoint):
        tangent, strain, history = ctx.saved_tensors
        current_grad = torch.einsum("pij,pi->pj", tangent, adjoint)
        history_grad = None
        if ctx.needs_input_grad[1] and history.shape[0]:
            history_grad = torch.from_numpy(ctx.native.history_vjp(
                numpy(history), numpy(strain), numpy(adjoint), ctx.history_step))
        return current_grad, history_grad, None, None


class MaterialContext:
    def __init__(self, factory, count, components, training=False, history_step=1.e-7):
        self.native = factory(count)
        if self.native.components != components:
            raise ValueError("Network local components must match the configured C++ material")
        self.training = training
        self.history_step = history_step
        self.history = []
        self.trial = None

    def update(self, strain):
        if strain.device.type != "cpu" or strain.dtype != torch.float64:
            raise ValueError("The Jem/Jive bridge requires CPU float64 tensors")
        history = torch.stack(self.history) if self.history else strain.new_empty((0, *strain.shape))
        stress = MaterialUpdate.apply(strain, history, self.native, self.history_step)
        self.trial = strain
        return stress

    def commit(self):
        if self.trial is not None:
            self.native.commit()
            if self.training:
                # Retain the graph: earlier encoded strains affect later history.
                self.history.append(self.trial)
            self.trial = None

    def cancel(self):
        self.native.cancel()
        self.trial = None
