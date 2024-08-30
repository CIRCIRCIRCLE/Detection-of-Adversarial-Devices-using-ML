from __future__ import annotations

import torch
from utils import flatten_models, unflatten_tensor

from .base import Attacker

class Lie(Attacker):
    def __init__(self, conf, byz_clients):
        super().__init__(conf, byz_clients)
        self.lie_z = conf.attacker.lie.get("lie_z", 1.5)  # 使用.get方法获取lie_z，默认值为1.5

    def attack(self, sampled_clients: list, server):
        ref_models = self.get_ref_models(sampled_clients)
        flat_models, struct = flatten_models(ref_models)

        mu = flat_models.mean(dim=0)
        sigma = flat_models.var(dim=0, unbiased=False)

        # Debugging logs to check the values
        print(f"mu: {mu}")
        print(f"sigma: {sigma}")
        print(f"lie_z: {self.lie_z}")

        if sigma is None:
            raise ValueError("Sigma is None, check the input data for calculation")

        if self.lie_z is None:
            raise ValueError("lie_z is None, check the attacker configuration")

        flat_byz_model = mu - self.lie_z * sigma

        byz_state_dict = unflatten_tensor(flat_byz_model, struct)

        self.set_byz_uploaded_content(sampled_clients, byz_state_dict, server)
