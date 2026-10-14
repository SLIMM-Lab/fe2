#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Physically Recurrent Neural Network (PRNN)

The PRNN is an encoder-decoder architecture that learns tasks akin to
localization and homogenization operations in micromechanics. Meaningful
representations are encouraged by including between encoder and decoder
a material layer with a set of real (physics-based) constitutive models.

    - Encoder (localization): encodes global strains to a set of local
      strains for an adjustable number of 'fictitious' material points.
      The current setup implements a single affine and fully-connected
      layer. Including non-linearity through hidden layers might be
      beneficial depending on the application.

    - Material layer: converts local strains to local stresses according
      to fixed (non-trainable) and purely physics-based constitutive laws.
      If targets come from microscale simulations, including in the material
      layer the exact same constitutive models as in the microscale model
      tends to yield the best results. Material properties are in principle
      fixed to their microscale values, although the code can be trivially
      extended to make them trainable. State variable evolution (e.g plastic
      strains, damage) is handled exclusively by the material models and
      are therefore not learned from data

    - Decoder (homogenization): decodes the complete set of local stresses
      to a global homogenized stress. To more closely mimic the usual
      micromodel volume averaging, decoder weights are constrained to be
      positive by activating them with softplus.

The code here also includes a small custom layer class implementing the
softplus weight activation.
"""

import math
# Adapted from SLIMM-Lab/pyprnn; see LICENSE and UPSTREAM.md.

import torch

from .material import MaterialContext


class PRNN(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_matpts, **kwargs):
        super(PRNN,self).__init__()

        self.device = kwargs.get('device',torch.device('cpu'))

        self.n_features   = n_features
        self.mat_pts      = n_matpts
        self.n_latents    = self.mat_pts*self.n_features
        self.n_outputs    = n_outputs

        self.material_factory = kwargs.get('material_factory')
        if self.material_factory is None:
            raise ValueError("Supply a native MaterialFactory for the material layer")
        self.history_step = kwargs.get('history_step', 1.e-7)
        self._online = None
        self._trial_input = None
        self._online_count = None

        self.fc1 = torch.nn.Linear(in_features=self.n_features,
                                   out_features=self.n_latents,
                                   device = self.device,
                                   bias = False)
        self.fc2 = SoftLayer(in_features=self.n_latents,
                             out_features=self.n_outputs,
                             device = self.device,
                             bias = False)

    def _step(self, strain, context):
        # Architecture experimentation belongs here and in the neural modules.
        local_strain = self.fc1(strain).reshape(-1, self.n_features)
        local_stress = context.update(local_strain)
        return self.fc2(local_stress.reshape(-1, self.n_latents))

    def forward(self, x):
        """Training: a fresh independent material history per batched path."""
        if x.ndim != 3 or x.shape[2] != self.n_features or x.shape[1] == 0:
            raise ValueError("Expected [batch, nonempty sequence, features]")
        context = MaterialContext(self.material_factory, x.shape[0] * self.mat_pts,
                                  self.n_features, training=True, history_step=self.history_step)
        output = []
        for t in range(x.shape[1]):
            output.append(self._step(x[:, t, :], context))
            context.commit()
        return torch.stack(output, dim=1)

    def evaluate(self, strain):
        """Online trial: initialize once at t=0, never commit implicitly."""
        vector = strain.ndim == 1
        batch = strain.unsqueeze(0) if vector else strain
        if batch.ndim != 2 or batch.shape[1] != self.n_features:
            raise ValueError("Expected [features] or [batch, features]")
        count = batch.shape[0] * self.mat_pts
        if self._online is None:
            self._online = MaterialContext(self.material_factory, count, self.n_features)
            self._online_count = count
        if count != self._online_count:
            raise ValueError("Online batch size cannot change; reset for a new problem")
        result = self._step(batch, self._online)
        self._trial_input = strain.detach().clone()
        return result[0] if vector else result

    def commit(self):
        if self._online is not None:
            self._online.commit()
        self._trial_input = None

    def cancel(self):
        if self._online is not None:
            self._online.cancel()
        self._trial_input = None

    def observe(self, strain):
        """Output query: restore any pending Newton trial after evaluating."""
        candidate = self._trial_input
        try:
            result = self.evaluate(strain)
            if candidate is None:
                self.cancel()
            else:
                self.evaluate(candidate)
            return result
        except Exception:
            self.cancel()
            raise

    def reset(self):
        """Explicit start of a new online problem; not used in Newton updates."""
        self._online = None
        self._online_count = None
        self._trial_input = None

    def script_neural_layers(self):
        """Optional JIT for the pure neural submodules, outside the callback."""
        self.fc1 = torch.jit.script(self.fc1)
        self.fc2 = torch.jit.script(self.fc2)
        return self


class SoftLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SoftLayer,self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sp = torch.nn.Softplus()
        self.weight = torch.nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(
                    torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
      return torch.nn.functional.linear(
              input, self.sp(self.weight), self.bias)
