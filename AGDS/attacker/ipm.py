from __future__ import annotations
import torch
from utils import flatten_models, unflatten_tensor
from .base import Attacker

class IPM(Attacker):
    def attack(self, sampled_clients: list, server):
        ref_models = self.get_ref_models(sampled_clients)
        flat_models, struct = flatten_models(ref_models)

        flat_byz_model = torch.zeros_like(flat_models.mean(dim=0))

        byz_state_dict = unflatten_tensor(flat_byz_model, struct)
        self.set_byz_uploaded_content(sampled_clients, byz_state_dict, server)
