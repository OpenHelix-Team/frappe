from collections import OrderedDict

import torch
import torch.nn as nn

from pathlib import Path
import sys, os

current_file = Path(__file__)
sys.path.append(current_file.parent.parent)

from rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid,
                        get_multimodal_cond_pos_embed)

def build_mlp(hidden_size, projector_dim, z_dim): 
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

class RDT(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """

    def __init__(self,
                 output_dim=128,
                 horizon=None, 
                 hidden_size=1152,
                 depth=28,
                 num_heads=16,
                 max_lang_cond_len=1024,
                 img_cond_len=4096,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 z_dims=[768],
                 projector_dim=2048,
                 learnable_tokens=None):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config
        self.z_dims = z_dims
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        # We will use trainable sin-cos embeddings
        # add learnalbe preifx
        self.learnable_tokens_num = learnable_tokens
        self.learnable_tokens = nn.Parameter(torch.zeros(1, self.learnable_tokens_num, hidden_size))
        nn.init.normal_(self.learnable_tokens, mean=0, std=0.02)
        # [timestep; state; action]
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon + 3 + self.learnable_tokens_num, hidden_size))
        # Language conditions
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        # Image conditions
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))

        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        x_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                    mm_cond_lens=OrderedDict([
                                                        ('timestep', 1),
                                                        ('ctrl_freq', 1),
                                                        ('state', 1),
                                                        ('action', self.horizon),
                                                        ('learnable_tokens', self.learnable_tokens_num),
                                                    ]))
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size,
                                                                    torch.arange(self.max_lang_cond_len))
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                                                                embed_modality=False)
        self.lang_cond_pos_embed.data.copy_(torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))

        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                               mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                                                               embed_modality=False)
        self.img_cond_pos_embed.data.copy_(torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)

        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, 
                learnable_tokens=None, 
                x_pos_embed=None,
                projectors=None,
                lang_mask=None, img_mask=None, encoder_depth=None, return_hidden_states=False, eval=False):
        
        local_learnable_tokens = self.learnable_tokens if learnable_tokens is None else learnable_tokens
        local_x_pos_embed = self.x_pos_embed if x_pos_embed is None else x_pos_embed
        local_projectors = self.projectors if projectors is None else projectors
        num_learnable = local_learnable_tokens.shape[1]

        t = self.t_embedder(t).unsqueeze(1)
        freq = self.freq_embedder(freq).unsqueeze(1)
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, freq, x], dim=1)


        expanded_learnable_tokens = local_learnable_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat([x, expanded_learnable_tokens], dim=1)

        x = x + local_x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed

        zs_future = None 
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            c, mask = conds[i % 2], masks[i % 2]
            x = block(x, c, mask)
            if encoder_depth is not None and (i + 1) == encoder_depth and num_learnable > 0:
                x_future = x[:, -num_learnable:, :]
                zs_future = [projector(x_future.reshape(-1, x_future.shape[-1])).reshape(x_future.shape[0], num_learnable, -1) for projector in local_projectors]
        
        hidden_states = x[:, -(self.horizon + num_learnable):-num_learnable]
        if return_hidden_states:
            return (hidden_states, zs_future) if zs_future is not None else hidden_states
        
        x = self.final_layer(x)

        if num_learnable > 0:
            x = x[:, -(self.horizon + num_learnable):-num_learnable]
        else: 
            x = x[:, -self.horizon:]
        if eval:
            return x, hidden_states
        return x, zs_future