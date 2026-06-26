# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class SpatiaConditioner(nn.Module):
    
    def __init__(
        self,
        num_transitions: int = 10,
        dim_g: int = 512,
        dim_m: int = 10,
        embed_dim: int = 512,
        ctrl_embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_transitions = num_transitions
        
        self.trans_embed = nn.Embedding(num_transitions, embed_dim)
        
        self.proj_g = nn.Sequential(
            nn.Linear(dim_g, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        self.proj_m = nn.Sequential(
            nn.Linear(dim_m, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        self.mlp_pert = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        self.mlp_cond = nn.Sequential(
            nn.Linear(ctrl_embed_dim + embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        z_ctrl: torch.Tensor,
        transition_ids: torch.Tensor,
        delta_g: torch.Tensor,
        delta_m: torch.Tensor,
    ) -> torch.Tensor:
        z_tau = self.trans_embed(transition_ids)
        
        z_g = self.proj_g(delta_g)
        z_m = self.proj_m(delta_m)
        
        z_pert_input = torch.cat([z_tau, z_g, z_m], dim=-1)
        z_pert = self.mlp_pert(z_pert_input)
        
        z_cond_input = torch.cat([z_ctrl, z_pert], dim=-1)
        z_cond = self.mlp_cond(z_cond_input)
        
        return z_cond
    
    def forward_perturbation_only(
        self,
        transition_ids: torch.Tensor,
        delta_g: torch.Tensor,
        delta_m: torch.Tensor,
    ) -> torch.Tensor:
        z_tau = self.trans_embed(transition_ids)
        z_g = self.proj_g(delta_g)
        z_m = self.proj_m(delta_m)
        
        z_pert_input = torch.cat([z_tau, z_g, z_m], dim=-1)
        z_pert = self.mlp_pert(z_pert_input)
        
        return z_pert


class SimpleSpatiaConditioner(nn.Module):
    
    def __init__(
        self,
        num_transitions: int = 10,
        dim_g: int = 512,
        dim_m: int = 10,
        ctrl_embed_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.trans_embed = nn.Embedding(num_transitions, output_dim // 4)
        
        self.proj_ctrl = nn.Linear(ctrl_embed_dim, output_dim // 2)
        self.proj_g = nn.Linear(dim_g, output_dim // 4)
        self.proj_m = nn.Linear(dim_m, output_dim // 8)
        
        input_dim = output_dim // 2 + output_dim // 4 + output_dim // 4 + output_dim // 8
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(
        self,
        z_ctrl: torch.Tensor,
        transition_ids: torch.Tensor,
        delta_g: torch.Tensor,
        delta_m: torch.Tensor,
    ) -> torch.Tensor:
        z_c = self.proj_ctrl(z_ctrl)
        z_t = self.trans_embed(transition_ids)
        z_g = self.proj_g(delta_g)
        z_m = self.proj_m(delta_m)
        
        z_all = torch.cat([z_c, z_t, z_g, z_m], dim=-1)
        z_cond = self.fusion(z_all)
        
        return z_cond


def create_spatia_conditioner(
    config: Dict,
    device: torch.device = None,
) -> SpatiaConditioner:
    conditioner = SpatiaConditioner(
        num_transitions=config.get("num_transitions", 10),
        dim_g=config.get("dim_g", 512),
        dim_m=config.get("dim_m", 10),
        embed_dim=config.get("embed_dim", 512),
        ctrl_embed_dim=config.get("ctrl_embed_dim", 512),
        dropout=config.get("dropout", 0.1),
    )
    
    if device is not None:
        conditioner = conditioner.to(device)
    
    return conditioner
