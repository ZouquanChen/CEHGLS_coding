# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
from torch.optim import Optimizer


class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if beta != 0:
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group["lr"])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs, use_hgls=False):
        """
        SCAFFOLD optimization step

        Args:
            server_cs: Server control variables list
            client_cs: Client control variables list
            use_hgls: Whether to enable HGLS mode
                - False: Apply control variables to entire model (original behavior)
                - True: Apply control variables only to base, head uses normal SGD
        """
        for group in self.param_groups:
            params = list(group["params"])
            if use_hgls and len(params) >= 2:
                # HGLS mode: separate base and head
                # Assume last two parameters are head's weight and bias
                base_params = params[:-2]
                head_params = params[-2:]

                # base parameters apply control variables
                for p, sc, cc in zip(base_params, server_cs[:-2], client_cs[:-2]):
                    if p.grad is not None:
                        p.data.add_(other=(p.grad.data + sc - cc), alpha=-group["lr"])

                # head parameters normal SGD (no control variables)
                for p in head_params:
                    if p.grad is not None:
                        p.data.add_(other=p.grad.data, alpha=-group["lr"])
            else:
                # Original behavior: apply control variables to all parameters
                for p, sc, cc in zip(group["params"], server_cs, client_cs):
                    if p.grad is not None:
                        p.data.add_(other=(p.grad.data + sc - cc), alpha=-group["lr"])


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group["params"], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group["lr"] * (
                    p.grad.data
                    + group["lamda"] * (p.data - localweight.data)
                    + group["mu"] * p.data
                )

        return group["params"]


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group["lr"], d_p)


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group["params"], global_params):
                g = g.to(device)
                d_p = p.grad.data + group["mu"] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group["lr"])
