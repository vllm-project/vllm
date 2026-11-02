# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.qwen_vl_triton_warmup import (
    _warm_mrope,
    _warm_vision,
)


@torch.inference_mode()
def test_vision_warmup_matches_runtime_rotary_layout(
    default_vllm_config, dist_init
) -> None:
    import einops
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_default_torch_dtype

    config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=144,
        num_heads=2,
        intermediate_size=32,
        out_hidden_size=32,
        num_position_embeddings=16,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        deepstack_visual_indexes=[],
    )
    device = torch.device(current_platform.device_type)
    with set_default_torch_dtype(torch.float16), torch.device(device):
        visual = Qwen3_VisionTransformer(config).to(device)
    attention = visual.blocks[0].attn
    attention.qkv.quant_method.process_weights_after_loading(attention.qkv)
    layouts: list[tuple[torch.Size, tuple[int, ...]]] = []

    def record_layout(module, args):
        qk, _, _ = args
        layouts.append((qk.shape, qk.stride()))

    def reject_tower_forward(module, args):
        raise AssertionError("vision warmup must not run the full tower")

    expected = []
    for seq_len in (256, 32, 32, 4):
        x = torch.zeros(
            seq_len, 1, config.hidden_size, dtype=visual.dtype, device=device
        )
        projected, _ = attention.qkv(x)
        qkv = einops.rearrange(
            projected,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=config.num_heads,
        )
        qk = qkv[:, :, :2].flatten(2, 3)
        expected.append((qk.shape, qk.stride()))

    with (
        attention.apply_rotary_emb.register_forward_pre_hook(record_layout),
        visual.register_forward_pre_hook(reject_tower_forward),
        visual.blocks[0].register_forward_pre_hook(reject_tower_forward),
    ):
        _warm_vision(visual, device)
        assert layouts == expected

        visual.cpu()
        layouts.clear()
        _warm_vision(visual, torch.device("cuda"))
        assert layouts == []


@pytest.mark.parametrize(
    "owners_fused, skipped", [((), False), ((True,), True), ((True, False), False)]
)
def test_mrope_warmup_launches_triton_shapes(owners_fused, skipped) -> None:
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
            rope = FakeRope()
            if not owners_fused:
                self.rotary_emb = rope
            for i, fused in enumerate(owners_fused):
                owner = torch.nn.Module()
                owner.rotary_emb, owner.use_fused_qk_norm_rope_gate = rope, fused
                self.add_module(f"attn{i}", owner)

        def get_mrope_input_positions(self, input_tokens, mm_features):
            raise AssertionError("warmup must not build live M-RoPE positions")

    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda parallel_config: 4,
            get_num_kv_heads=lambda parallel_config: 2,
        ),
        parallel_config=object(),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    _warm_mrope(runner, FakeMropeModel())
    expected = [torch.Size((3, n)) for n in (1, 2, 16)]
    assert [shape for shape, _ in launched] == ([] if skipped else expected)


def test_mrope_warmup_skips_models_without_supports_mrope() -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("M-RoPE warmup must not run without SupportsMRoPE")

    runner = SimpleNamespace(
        model_config=SimpleNamespace(get_num_attention_heads=fail),
        get_model=fail,
    )
    _warm_mrope(runner, torch.nn.Module())
