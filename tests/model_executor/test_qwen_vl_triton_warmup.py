# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.qwen_vl_triton_warmup import (
    _warm_mrope,
    _warm_vision,
)


def test_vision_warmup_calls_only_position_and_rotary_paths() -> None:
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    calls: list[tuple[str, object]] = []

    class FakeAttention:
        num_attention_heads_per_partition = 4
        hidden_size_per_attention_head = 8

        def apply_rotary_emb(self, qk, cos, sin):
            calls.append(("rotary", qk.shape))

    class FakeVisual(Qwen3_VisionTransformer):
        spatial_merge_size = 2

        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.blocks = [SimpleNamespace(attn=FakeAttention())]

        def fast_pos_embed_interpolate(self, grid_thw):
            calls.append(("position", grid_thw[0]))

        def rot_pos_emb(self, grid_thw):
            _, h, w = grid_thw[0]
            shape = (h * w, 4)
            return torch.empty(shape), torch.empty(shape)

        def forward(self, *args, **kwargs):
            raise AssertionError("vision warmup must not run the full tower")

    model = torch.nn.Module()
    model.visual = FakeVisual()
    _warm_vision(model)

    grids = [(1, 16, 16), (1, 16, 2), (1, 2, 16), (1, 2, 2)]
    assert [value for name, value in calls if name == "position"] == [
        list(grid) for grid in grids
    ]
    assert [value for name, value in calls if name == "rotary"] == [
        torch.Size((2, h * w, 4, 8)) for _, h, w in grids
    ]


def test_mrope_warmup_launches_triton_shapes() -> None:
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    launched: list[tuple[torch.Size, torch.Size]] = []

    class FakeRope(MRotaryEmbedding):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.head_size = 8
            self.rotary_dim = 8
            self.mrope_section = [2, 3, 3]
            self.mrope_interleaved = False
            self.is_neox_style = True

        def forward(self, positions, query, key):
            launched.append((positions.shape, query.shape))
            return query, key

    class FakeMropeModel(torch.nn.Module):
        supports_mrope = True

        def __init__(self) -> None:
            super().__init__()
            self.rotary_emb = FakeRope()

        def get_mrope_input_positions(self, input_tokens, mm_features):
            raise AssertionError("warmup must not build live M-RoPE positions")

    runner = SimpleNamespace(
        num_query_heads=4,
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda parallel_config: 2,
        ),
        parallel_config=object(),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    _warm_mrope(runner, FakeMropeModel())
    assert [shape for shape, _ in launched] == [
        torch.Size((3, 1)),
        torch.Size((3, 2)),
        torch.Size((3, 16)),
    ]


def test_mrope_warmup_skips_models_without_supports_mrope() -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("M-RoPE warmup must not run without SupportsMRoPE")

    runner = SimpleNamespace(
        num_query_heads=4,
        model_config=SimpleNamespace(get_num_kv_heads=fail),
        get_model=fail,
    )
    _warm_mrope(runner, torch.nn.Module())
