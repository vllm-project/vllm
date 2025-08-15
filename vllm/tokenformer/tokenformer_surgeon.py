# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod, ABC
import torch
from torch import nn
import math
import os

from vllm.logger import init_logger

logger = init_logger(__name__)


class TokenformerMLPAdapter(nn.Module):
    def __init__(self, layer, hidden_size, device):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.num_heads = int(os.getenv("TOKENFORMER_NUM_HEADS", "8"))
        self.head_dim = hidden_size // self.num_heads
        self.tokenformer_r = int(os.getenv("TOKENFORMER_R", "4"))

        self.tokenformer_k = nn.Parameter(
            torch.zeros(self.num_heads, self.hidden_size, device=device)
        )
        self.tokenformer_v = nn.Parameter(
            torch.zeros(
                self.num_heads, self.hidden_size * self.tokenformer_r, device=device
            )
        )

        self.tokenformer_p = nn.Parameter(
            torch.zeros(self.tokenformer_r, self.hidden_size, device=device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        k_gain = 3.0 / math.sqrt(self.hidden_size / self.num_heads)
        v_gain = 3.0 / math.sqrt(self.hidden_size)

        nn.init.normal_(self.tokenformer_k, std=k_gain)
        nn.init.uniform_(self.tokenformer_v, a=-v_gain, b=v_gain)
        nn.init.zeros_(self.tokenformer_p)

    # Call layer with all inputs and kwargs
    def forward(self, query: torch.Tensor):
        base_layer_results = self.layer(query)

        tokenformer_results = self.tokenformer_op_1(query)

        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results
        return layer_and_adaptor_sum

    def tokenformer_op(self, query):

        return query @ self.tokenformer_k.transpose(0, 1) @ self.tokenformer_v

    def tokenformer_op_1(self, query):

        q = query.view(
            -1, self.num_heads, self.hidden_size // self.num_heads
        ).transpose(0, 1)
        k = self.tokenformer_k.view(
            -1, self.num_heads, self.hidden_size // self.num_heads
        ).transpose(0, 1)
        v = self.tokenformer_v.view(
            -1, self.num_heads, self.hidden_size * self.tokenformer_r // self.num_heads
        ).transpose(0, 1)

        result = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,  # should be false for tokenformer
        )

        proj_down = (
            result.transpose(0, 1)
            .contiguous()
            .view([-1, self.hidden_size, self.tokenformer_r])
        )

        # tokenformer_p dims are [tokenformer_r, hidden_size]
        # query dims are [batch size, length, 1, hidden_size]
        # proj_down are [batch size, length, hidden_size, tokenformer_r]

        query_batch = query.view([-1, 1, self.hidden_size])

        result = torch.bmm(query_batch, proj_down) @ self.tokenformer_p

        return result.view(query.shape)

    # Visualize the size of the parameters
    def __repr__(self):
        return (
            f"TokenformerMLPAdapter(\nhidden_size={self.hidden_size}\n(layer): "
            + self.layer.__repr__()
            + "\n)"
        )


class TokenformerAttentionAdapter(nn.Module):
    def __init__(self, layer, hidden_size, device):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size

        self.tokenformer_k = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size, device=device)
        )
        self.tokenformer_v = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size, device=device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = 3.0 / math.sqrt(self.hidden_size)

        nn.init.zeros_(self.tokenformer_k)
        nn.init.normal_(self.tokenformer_v, std=gain)

        nn.init.normal_(self.tokenformer_k[0:1, :], std=gain)
        nn.init.zeros_(self.tokenformer_v[0:1, :])

    def forward(self, query, base_layer_results) -> torch.Tensor:

        tokenformer_results = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=self.tokenformer_k,
            value=self.tokenformer_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,  # should be false for tokenformer
        )

        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results
        return layer_and_adaptor_sum

    def __repr__(self):
        return (
            f"TokenformerAttentionAdapter(\nhidden_size={self.hidden_size}\n(layer): "
            + self.layer.__repr__()
            + "\n)"
        )


class TokenformerSurgeon(ABC):

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def _is_attn_layer(self, layer_name):
        return layer_name.split(".")[-1] == "attn"

    def _is_mlp_layer(self, layer_name):
        return "mlp" in layer_name.split(".")[-1]

    def _recursive_setattr(self, obj, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            self._recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    def update_mlp(self, name, layer):
        """Try to wrap the layer with a TokenformerMLPAdaptor."""
        if not self._is_mlp_layer(name):
            return

        logger.info(f"Wrapping layer {name} with TokenformerMLPAdaptor")

        # Wrap the layer with a TokenformerMLPAdapter
        self._recursive_setattr(
            self.model,
            name,
            TokenformerMLPAdapter(
                layer, self.model.config.hidden_size, device=self.device
            ),
        )

    @abstractmethod
    def update_attn(self, name, layer):
        pass

    def insert_adapter_modules(self):
        # Add tokenformer adapters for mlp and attention
        for name, layer in self.model.named_modules():
            self.update_mlp(name, layer)
            self.update_attn(name, layer)

        return self.model