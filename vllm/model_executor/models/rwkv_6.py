# coding=utf-8
"""PyTorch RWKV6 model.(native PyTorch version)"""
"""
author: @Zhiyuan Li
email:
date: 2024-07-22
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


import torch
import torch.nn as nn
from transformers import RwkvConfig
from vllm.config import LoRAConfig, CacheConfig, SchedulerConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.models.interfaces import HasInnerState
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.worker.model_runner import (_BATCH_SIZES_TO_CAPTURE,
                                      _get_graph_batch_size)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
KVCache = Tuple[torch.Tensor, torch.Tensor]

@dataclass
class RwkvCacheParams:
    is_prompt: bool = False
    ssm_state: torch.Tensor = torch.Tensor()


class Rwkv_Block(MyModule):
    def __init__(self, block_w: dict, hidden_size: int, n_head: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = hidden_size // n_head

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln1.weight = nn.Parameter(block_w['ln1.weight'])
        self.ln1.bias = nn.Parameter(block_w['ln1.bias'])
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln2.weight = nn.Parameter(block_w['ln2.weight'])
        self.ln2.bias = nn.Parameter(block_w['ln2.bias'])


        self.silu = nn.SiLU(inplace=False)

        self.att_time_maa_x = nn.Parameter(block_w['att.time_maa_x'])
        self.att_time_maa_w = nn.Parameter(block_w['att.time_maa_w'])
        self.att_time_maa_k = nn.Parameter(block_w['att.time_maa_k'])
        self.att_time_maa_v = nn.Parameter(block_w['att.time_maa_v'])
        self.att_time_maa_r = nn.Parameter(block_w['att.time_maa_r'])
        self.att_time_maa_g = nn.Parameter(block_w['att.time_maa_g'])
        self.att_time_maa_w1 = nn.Parameter(block_w['att.time_maa_w1'])
        self.att_time_maa_w2 = nn.Parameter(block_w['att.time_maa_w2'])
        self.att_time_decay = nn.Parameter(block_w['att.time_decay'])
        self.att_time_decay_w1 = nn.Parameter(block_w['att.time_decay_w1'])
        self.att_time_decay_w2 = nn.Parameter(block_w['att.time_decay_w2'])
        self.att_time_faaaa = nn.Parameter(block_w['att.time_faaaa'])
        self.att_receptance = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.att_receptance.weight = nn.Parameter(block_w['att.receptance.weight'])
        self.att_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.att_key.weight = nn.Parameter(block_w['att.key.weight'])
        self.att_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.att_value.weight = nn.Parameter(block_w['att.value.weight'])
        self.att_output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.att_output.weight = nn.Parameter(block_w['att.output.weight'])
        self.att_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.att_gate.weight = nn.Parameter(block_w['att.gate.weight'])


        self.att_group_norm = nn.GroupNorm(num_groups=n_head, num_channels=hidden_size, eps=1e-5, affine=True)
        self.att_group_norm.weight = nn.Parameter(block_w['att.ln_x.weight'])
        self.att_group_norm.bias = nn.Parameter(block_w['att.ln_x.bias'])

        self.ffn_time_maa_k = nn.Parameter(block_w['ffn.time_maa_k'])
        self.ffn_time_maa_r = nn.Parameter(block_w['ffn.time_maa_r'])
        self.ffn_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ffn_key.weight = nn.Parameter(block_w['ffn.key.weight'])
        self.ffn_receptance = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ffn_receptance.weight = nn.Parameter(block_w['ffn.receptance.weight'])
        self.ffn_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ffn_value.weight = nn.Parameter(block_w['ffn.value.weight'])

    @MyFunction
    def channel_mixing(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        i0 = (2 + self.head_size) * i + 0
        sx = state[:, i0] - x
        state[:, i0] = x
        xk = x + sx * self.ffn_time_maa_k
        xr = x + sx * self.ffn_time_maa_r
        r = torch.sigmoid(self.ffn_receptance(xr))
        k = torch.relu(self.ffn_key(xk)).pow(2)
        output = r * self.ffn_value(k)
        return output, state

    @MyFunction
    def channel_mixing_parallel(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        i0 = (2 + self.head_size) * i + 0

        sx_lerp = torch.empty_like(x)
        sx_lerp[:, 0] = state[:, i0] - x[:, 0]
        sx_lerp[:, 1:] = x[:, :-1] - x[:, 1:]

        state[:, i0] = x[:, -1]

        xk = x + sx_lerp * self.ffn_time_maa_k
        xr = x + sx_lerp * self.ffn_time_maa_r

        r = torch.sigmoid(self.ffn_receptance(xr)) # [Batch, L, hiddle_size]
        k = torch.relu(self.ffn_key(xk)).pow(2)

        output = r * self.ffn_value(k)
        return output, state

    def time_mixing(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        batch_size, H, S = x.size(0), self.n_head, self.head_size
        x, state, g = self.time_mixing_jit(x, state, i, batch_size, H, S)

        x = self.time_mixing_jit2(x, g)

        return x, state

    @MyFunction
    def time_mixing_jit(self, x: torch.Tensor, state: torch.Tensor, i: int,
                                    batch_size: int, H: int, S: int):
        i1 = (2 + S) * i + 1  # i is the block number

        sx = state[:, i1] - x
        state[:, i1] = x # Information is compressed to position 1 on each layer

        xxx = x + sx * self.att_time_maa_x
        xxx = torch.tanh(xxx @ self.att_time_maa_w1).view(batch_size, 5, 1, -1)
        xxx = torch.matmul(xxx, self.att_time_maa_w2).view(batch_size, 5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=1)

        xw = x + sx * (self.att_time_maa_w + mw)
        xk = x + sx * (self.att_time_maa_k + mk)
        xv = x + sx * (self.att_time_maa_v + mv)
        xr = x + sx * (self.att_time_maa_r + mr)
        xg = x + sx * (self.att_time_maa_g + mg)

        # calculate w, r, k, v, g
        w = (self.att_time_decay + (torch.tanh(xw @ self.att_time_decay_w1) @ self.att_time_decay_w2))
        w = -torch.exp(w.view(batch_size, H, S, 1))

        r = self.att_receptance(xr).view(batch_size, H, 1, S)
        k = self.att_key(xk).view(batch_size, H, S, 1)
        v = self.att_value(xv).view(batch_size, H, 1, S)
        g = self.silu(self.att_gate(xg))

        # Update state using attention mechanism
        s = state[:, (2+S)*i+2:(2+S)*(i+1), :].view(batch_size, H, S, S)
        a = k @ v
        x = r @ (self.att_time_faaaa * a + s)
        s = a + torch.exp(w) * s
        # Update the attention parameters of the i-th layer STATE
        state[:, (2+S)*i+2:(2+S)*(i+1), :] = s.view(batch_size, S, -1)
        return x, state, g

    @MyFunction
    def time_mixing_jit2(self, x:torch.Tensor, g):
        return self.att_output(self.att_group_norm(x.flatten(start_dim=1)) * g)

    def time_mixing_parallel(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        batch_size, L, H, S = x.size(0), x.size(1), self.n_head, self.head_size
        x, state, g = self.time_mixing_parallel_jit1(x, state, i, batch_size, L, H, S)

        x = self.time_mixing_parallel_jit2(x, g, batch_size, L)

        return x, state

    @MyFunction
    def time_mixing_parallel_jit1(self, x: torch.Tensor, state: torch.Tensor, i: int,
                                    batch_size: int, L: int, H: int, S: int):
        i1 = (2 + S) * i + 1
        sx_lerp = torch.empty_like(x)
        sx_lerp[:, 0] = state[:, i1] - x[:, 0]

        sx_lerp[:, 1:] = x[:, :-1] - x[:, 1:]

        state[:, i1] = x[:, -1]

        xxx = x + sx_lerp * self.att_time_maa_x # torch.Size([B, L, hiddle_size])
        xxx = torch.tanh(xxx @ self.att_time_maa_w1).view(batch_size, L, 5, 1, -1) # att_time_maa_w1: [hiddle_size, 160]
        xxx = torch.matmul(xxx, self.att_time_maa_w2).view(batch_size, L, 5, -1) # [Batch, L, 5, hiddle_size]

        mw, mk, mv, mr, mg = xxx.unbind(dim=2) # [10, 100, hiddle_size]

        xw = x + sx_lerp * (self.att_time_maa_w + mw) # torch.Size([B, L, hiddle_size])
        xk = x + sx_lerp * (self.att_time_maa_k + mk)
        xv = x + sx_lerp * (self.att_time_maa_v + mv)
        xr = x + sx_lerp * (self.att_time_maa_r + mr)
        xg = x + sx_lerp * (self.att_time_maa_g + mg)

        w = (self.att_time_decay + (torch.tanh(xw @ self.att_time_decay_w1) @ self.att_time_decay_w2))
        w = -torch.exp(w.view(batch_size, L, H, S, 1))

        r = self.att_receptance(xr).view(batch_size, L, H, 1, S)
        k = self.att_key(xk).view(batch_size, L, H, S, 1)
        v = self.att_value(xv).view(batch_size, L, H, 1, S)
        g = self.silu(self.att_gate(xg)) # [10, 100, hiddle_size]
        # TODO, apply kernel here, cuda or fla(triton)


        w = torch.exp(w)
        s = state[:, (2+S)*i+2:(2+S)*(i+1)].view(batch_size, H, S, S)
        a = k @ v # a: [batch_size, L, H, S, S]

        state_s = torch.zeros(batch_size, L, H, S, S, dtype=x.dtype, device=x.device)
        state_s[:, 0] = s

        for l in range(L-1):
            s = a[:, l] + w[:, l] * s
            state_s[:, l+1] = s
        s = a[:, -1] + w[:, -1] * s

        state[:, (2+S)*i+2:(2+S)*(i+1)] = s.view(batch_size, S, -1)

        x = r @ (self.att_time_faaaa * a + state_s)
        return x, state, g

    @MyFunction
    def time_mixing_parallel_jit2(self, x: torch.Tensor, g: torch.Tensor, batch_size: int, L:int):
        return self.att_output(self.att_group_norm(x.flatten(start_dim=2).view(batch_size * L, -1)).view(batch_size, L, -1) * g)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        x_time, state = self.time_mixing(self.ln1(x), state, i)
        x = x + x_time
        x_channel, state = self.channel_mixing(self.ln2(x), state, i)
        x = x + x_channel

        return x, state

    @torch.no_grad()
    def forward_parallel(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        # Time mixing
        x_time, state = self.time_mixing_parallel(self.ln1(x), state, i)
        x = x + x_time

        # Channel mixing
        x_channel, state = self.channel_mixing_parallel(self.ln2(x), state, i)
        x = x + x_channel

        return x, state

class RwkvModel(MyModule):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.load_params()
        self.eval()



    def load_params(self, load_from_file: bool = True, w: dict = None):
        # TODO: vllm
        if load_from_file:
            if not self.args['MODEL_NAME'].endswith('.pth'):
                self.args['MODEL_NAME'] += '.pth'
            w = torch.load(self.args['MODEL_NAME'], map_location="cpu")
        else:
            assert w is not None

        self.num_layer = 0
        for k in w.keys():
            if '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)
            if "blocks" in k: self.num_layer = max(self.num_layer, int(k.split(".")[1]))
        self.num_layer += 1

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.n_embd = w['blocks.0.ln1.weight'].shape[0]
        self.head_size = self.n_embd // self.n_head
        self.state_size = [self.num_layer * (2 + self.head_size), self.n_embd]

        self.emb = nn.Embedding.from_pretrained(w['emb.weight'], freeze=True)


        self.ln0 = nn.LayerNorm(self.n_embd)
        self.ln0.weight = nn.Parameter(w['blocks.0.ln0.weight'])
        self.ln0.bias = nn.Parameter(w['blocks.0.ln0.bias'])


        self.blocks = nn.ModuleList()

        for i in range(self.num_layer):
            block_w = {k[len(f'blocks.{i}.'):]: v for k, v in w.items() if f'blocks.{i}.' in k}
            self.blocks.append(Rwkv_Block(block_w, self.n_embd, self.n_head, self.args))


        self.ln_out = nn.LayerNorm(self.n_embd)
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])


        self.head = nn.Linear(self.n_embd, self.args['vocab_size'], bias=False)
        self.head.weight = nn.Parameter(w['head.weight'])


    @torch.no_grad()
    def forward(self,
                input_ids: torch.Tensor,
                state: torch.Tensor,
                attn_metadata: AttentionMetadata,
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.forward_jit1(input_ids)

        if attn_metadata.prefill_metadata is not None:
            # Prefill phase
            for i, block in enumerate(self.blocks):
                x, state = block.forward_parallel(x, state, i)
        else:
            # Decoding phase
            for i, block in enumerate(self.blocks):
                x, state = block(x, state, i)

        x = self.forward_jit2(x)

        return x, state


    @MyFunction
    def forward_jit1(self, token: torch.Tensor) -> torch.Tensor:
        return self.ln0(self.emb(token))

    @MyFunction
    def forward_jit2(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_out(x))

    @torch.no_grad()
    def forward_parallel_slices(self,
                                input_ids: torch.Tensor,
                                state: torch.Tensor,
                                attn_metadata: AttentionMetadata,
                                slice_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prefill forward with chunks of the RWKV6 model.
        Args:
            x (torch.Tensor): Input tensor, shape [Batch, N_embd].
            state (torch.Tensor): Hidden state tensor, shape [Batch, State Size, N_embd].
            i (int): Time index.
        Returns:
            torch.Tensor: Forward pass result tensor, shape same as input x.
        """
        # FIXME!
        data_len = input_ids.shape[1]
        for i in range((data_len-2)//slice_len+1):
            start = i*slice_len
            end = min((i+1)*slice_len, data_len)
            input_ids_ith_chunk = input_ids[:, start:end]
            token_out, state = self.forward(input_ids_ith_chunk, state, attn_metadata)

        return token_out, state

    def init_state(self, batch_size: int) -> torch.Tensor:
        state = torch.zeros(batch_size, self.state_size[0], self.state_size[1])
        return state

