# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones
import sys
import math

class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
        only_sa: bool = False,
        two_sa: bool = False,
        # gate: bool = False,
        ca_gate: bool = False,
        recursive_residual: int = 0,
        sa_mem: bool = False,
        ca_mem: bool = False,
        sa_align: bool = False,
        ca_align: bool = False,
        has_mlp: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.only_sa = only_sa
        self.two_sa = two_sa
        if ca_gate or (recursive_residual != 0):
            self.gate = nn.Linear(2 * d_model, d_model)
        self.ca_gate = ca_gate
        self.recursive_residual = recursive_residual

        if self.pos_enc_at_cross_attn_keys and (self.ca_gate or (recursive_residual != 0)):
                # x_preds = self.yolo.forward_16_head_world(backbone_features)
                raise ValueError(f'key pos enc add twice!! self.pos_enc_at_cross_attn_keys: {self.pos_enc_at_cross_attn_keys}, self.ca_gate: {self.ca_gate}, self.recursive_residual: {self.recursive_residual}')
        self.sa_mem = sa_mem
        self.ca_mem = ca_mem
        self.sa_align = sa_align
        self.ca_align = ca_align
        self.has_mlp = has_mlp

    def _forward_sa(self, tgt, query_pos):
        # print("sa")
        # Self-Attention
        # print("tgt[0][0][0]:", tgt[0][0][0])
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt
    
    def _forward_ca_align(self, tgt, query_pos, kv, kv_pos):
        # print("sa")
        # Self-Attention
        # print("tgt[0][0][0]:", tgt[0][0][0])
        tgt2 = self.norm1(tgt)
        q = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        k = kv + kv_pos if self.pos_enc_at_attn else kv
        v = kv
        tgt = self.self_attn(q, k, v=v)
        # tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        # print("ca")
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        # print("self.cross_attn_image.rope_k_repeat:", self.cross_attn_image.rope_k_repeat)
        # print("self.cross_attn_image.kv_in_dim:", self.cross_attn_image.kv_in_dim)
        # sys.exit()
        # Cross-Attention
        tgt2 = self.norm2(tgt)
        if not self.two_sa:
            # print("memory:", memory.shape, memory[0][0][0])
            tgt2 = self.cross_attn_image(
                q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
                k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
                v=memory,
                **kwds,
            )
        else:
            # print("self.two_sa:", self.two_sa)
            tgt2 = self.cross_attn_image(
                q=tgt2,
                k=tgt2,
                v=tgt2,
                **kwds,
            )
        # sys.exit()
        tgt = tgt + self.dropout2(tgt2)
        return tgt
    
    def memory_gating_ca(
        self, curr_mem, memory
    ) -> torch.Tensor:    
        B, total_len, C = memory.shape
        # print("memory:", memory.shape)
        H_W = curr_mem.shape[2] * curr_mem.shape[3]
        # print("curr_mem:", curr_mem.shape)
        M = total_len // H_W
        curr_mem = curr_mem.view(B, H_W, C)
        # print("curr_mem2:", curr_mem.shape)
        # H,W = math.sqrt(H_W)

        memory = memory.view(B, M, H_W, C)                 # [B, M, H*W, C]
        curr = curr_mem.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, H*W, C]
        # print("curr_mem3:", curr_mem.shape)

        x = torch.cat([curr, memory], dim=-1)   # [B, M, H*W, 2C]
        # print("x:", x.shape)
        g = torch.sigmoid(self.gate(x))        # [B, M, H*W, C]
        # print("g:", g.shape)
        M_t = g * memory
        # print("M_t:", M_t.shape)
        memory_out = M_t.view(B, M * H_W, C)
        return memory_out
    
    def memory_gating_recursive_residual(
        self, curr_mem, memory
    ) -> torch.Tensor:    
        x = torch.cat([curr_mem, memory], dim=-1)   # [B, H*W, 2C]
        # print("x:", x.shape)
        g = torch.sigmoid(self.gate(x))        # [B, H*W, C]
        # print("g:", g.shape, g)
        M_t = g * curr_mem + (1 - g) * memory       # residual fusion
        # print("M_t:", M_t.shape)
        return M_t

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        curr_mem = None,
        ca_do_sa: bool = False,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        # print("=================================")
        # print("MemoryAttentionLayer:")
        # print("tgt:", tgt.shape)
        # print("memory:", memory.shape)
        # print("pos:", pos.shape)
        # print("query_pos:", query_pos.shape)
        # sys.exit()
        # print("num_k_exclude_rope:", num_k_exclude_rope)
        if self.sa_align:
            # tgt = self._forward_sa(tgt, query_pos)
            # print("sa_align")
            # print("tgt:", tgt.shape, tgt[0][0][0])
            # print("query_pos:", query_pos.shape)
            # print("memory:", memory.shape, memory[0][0][0])
            # print("pos:", pos.shape)
            tgt_len = tgt.size(1)
            memory_len = memory.size(1)
            tgt_memory = self._forward_sa(torch.cat([tgt, memory], dim=1), torch.cat([query_pos, pos], dim=1))
            tgt, memory = torch.split(tgt_memory, [tgt_len, memory_len], dim=1)
            # print("after _forward_sa tgt:", tgt.shape, tgt[0][0][0])
            # print("after _forward_sa memory:", memory.shape, memory[0][0][0])
        elif self.ca_align:
            memory = self._forward_sa(torch.cat([tgt, memory], dim=1), torch.cat([query_pos, pos], dim=1))
        else:
            # print("sa")
            # print("tgt:", tgt.shape, tgt[0][0][0])
            # print("memory:", memory.shape, memory[0][0][0])
            tgt = self._forward_sa(tgt, query_pos)
        if self.recursive_residual == 2:
            tgt_before_ca = tgt.clone()
        atten_mem = None
        if self.sa_mem:
            atten_mem = tgt.clone()
            # print("sa_mem:", atten_mem[0][0][0])
        if not self.only_sa:
            if self.ca_gate:
                gate_memory = self.memory_gating_ca(curr_mem, memory+pos)
                tgt = self._forward_ca(tgt, gate_memory, query_pos, pos, num_k_exclude_rope)
            elif self.recursive_residual == 1:
                # tgt_clone = tgt.clone()
                # memory_clone = memory.clone()
                gate_memory = self.memory_gating_recursive_residual(tgt, memory+pos)
                # print("tgt_clone:", torch.equal(tgt, tgt_clone))
                # print("memory_clone:", torch.equal(memory, memory_clone))
                tgt = self._forward_ca(tgt, gate_memory, query_pos, pos, num_k_exclude_rope)
                # sys.exit()
            elif ca_do_sa:
                tgt = self._forward_ca(tgt, tgt, query_pos, query_pos, num_k_exclude_rope)
                # print("ca_do_sa")
            else:
                # print("normal ca:")
                # print("tgt:", tgt.shape, tgt[0][0][0])
                # print("memory:", memory.shape, memory[0][0][0])
                tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            # print("dont need ca")
        if self.ca_mem:
            atten_mem = tgt.clone()
            # print("ca_mem:", atten_mem[0][0][0])
        
        if self.recursive_residual == 2:
            tgt = self.memory_gating_recursive_residual(tgt_before_ca, tgt)
        # sys.exit()

        # MLP
        if self.has_mlp:
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        return tgt, atten_mem


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
        scale_for_pos_enc_at_input = 0.1,
        two_sa: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        self.scale_for_pos_enc_at_input = scale_for_pos_enc_at_input
        self.two_sa = two_sa

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
        curr_mem = None,
        ca_do_sa: bool = False,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )
            # print("memory attention curr_pos:", curr_pos.shape)

        if (not self.two_sa) and (not ca_do_sa):
            assert (
                curr.shape[1] == memory.shape[1]
            ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + self.scale_for_pos_enc_at_input * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            if (not self.two_sa) and (not ca_do_sa):
                memory = memory.transpose(0, 1)
                memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output, atten_mem = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                curr_mem=curr_mem,
                ca_do_sa=ca_do_sa,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output, atten_mem
