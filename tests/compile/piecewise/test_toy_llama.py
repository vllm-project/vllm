"""
Test the piecewise compilation with a simple model, comparing the output
with and without the piecewise compilation.
"""
from typing import Optional, Tuple

import torch
from torch import nn


@torch.library.custom_op("silly::attention", mutates_args=[])
def silly_attention(q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> torch.Tensor:
    return q + k + v


@silly_attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(q)


H = 128  # hidden size
M = 256  # mlp size
V = 128  # vocab size


class LlamaMLP(nn.Module):

    def __init__(self, ) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(
            in_features=H,
            out_features=M * 2,
            bias=False,
        )
        self.down_proj = torch.nn.Linear(
            in_features=M,
            out_features=H,
            bias=False,
        )

        self.gate_up_proj.weight.data.fill_(0.1)
        self.down_proj.weight.data.fill_(0.1)

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = x[:, :x.size(1) // 2] * torch.nn.functional.relu(
            x[:, x.size(1) // 2:])
        x = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(self, ) -> None:
        super().__init__()
        self.qkv_proj = torch.nn.Linear(
            in_features=H,
            out_features=H * 3,
        )

        self.o_proj = torch.nn.Linear(
            in_features=H,
            out_features=H,
        )

        self.qkv_proj.weight.data.fill_(0.1)
        self.o_proj.weight.data.fill_(0.1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([H, H, H], dim=-1)
        # silly positional encoding
        q = q + positions.unsqueeze(1)
        k = k + positions.unsqueeze(1)
        attn_output = torch.ops.silly.attention(q, k, v)
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, ) -> None:
        super().__init__()
        self.self_attn = LlamaAttention()
        self.mlp = LlamaMLP()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            # simulate layer norm
            hidden_states = hidden_states / 2
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = hidden_states / 2
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = hidden_states / 2
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            num_embeddings=V,
            embedding_dim=H,
        )
        self.layers = nn.ModuleList([LlamaDecoderLayer() for _ in range(2)])

        self.embed_tokens.weight.data.fill_(0.1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
        return hidden_states


def run_model():
    from vllm.compilation.compile_context import set_compile_context
    from vllm.compilation.decorators import support_torch_compile
    cls = support_torch_compile(LlamaModel)
    model = cls().eval().cuda()

    B = 16  # max batch size
    input_ids = torch.randint(0, V, (B, )).cuda()
    positions = torch.arange(B).cuda()

    with set_compile_context([1, 2]):
        model(input_ids, positions)
        model(input_ids[:2], positions[:2])
        model(input_ids[:1], positions[:1])

    input_ids[:2].zero_()
    output = model(input_ids[:2], positions[:2])

    return output.cpu()


def test_toy_llama():
    # compare output with and without piecewise compilation

    from vllm.compilation.levels import CompilationLevel
    levels = [CompilationLevel.NO_COMPILATION, CompilationLevel.PIECEWISE]

    import os
    directory = os.path.dirname(__file__)
    config = os.path.join(directory, "compilation_config.json")

    outputs = []
    for level in levels:
        os.environ["VLLM_TORCH_COMPILE_CONFIG"] = config
        os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(
            CompilationLevel.PIECEWISE)
        output = run_model()
        outputs.append(output)

    assert torch.allclose(outputs[0], outputs[1])
