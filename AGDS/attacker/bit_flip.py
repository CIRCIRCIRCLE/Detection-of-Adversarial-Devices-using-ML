from __future__ import annotations
import torch
from utils import flatten_models, unflatten_tensor
from .base import Attacker

class BitFlip(Attacker):
    def attack(self, sampled_clients: list, server):
        ref_models = self.get_ref_models(sampled_clients)
        flat_models, struct = flatten_models(ref_models)

        flip_prob = self.conf.attacker.bit_flip.flip_prob
        flat_byz_model = flat_models.to(torch.int)  # Convert to integer representation
        bit_mask = torch.rand_like(flat_byz_model).to(flat_byz_model.device) < flip_prob
        flat_byz_model = flat_byz_model ^ bit_mask.to(flat_byz_model.int)  # Apply bit flip
        flat_byz_model = flat_byz_model.to(flat_models.dtype)  # Convert back to original dtype

        byz_state_dict = unflatten_tensor(flat_byz_model, struct)
        self.set_byz_uploaded_content(sampled_clients, byz_state_dict, server)
