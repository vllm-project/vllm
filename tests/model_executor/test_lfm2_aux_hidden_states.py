# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LFM2 DSpark capture boundaries and speculative convolution state."""

from contextlib import ExitStack
from math import prod
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import (
    Lfm2Config,
    Lfm2VlConfig,
    PreTrainedConfig,
    Qwen3Config,
    Siglip2VisionConfig,
)

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM
from vllm.model_executor.models.lfm2_vl import Lfm2VLForConditionalGeneration
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head


def _make_config(tmp_path: Path, multimodal: bool, num_spec: int) -> VllmConfig:
    text_config = Lfm2Config(
        architectures=["Lfm2ForCausalLM"],
        vocab_size=256,
        hidden_size=128,
        conv_dim=128,
        block_dim=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        block_auto_adjust_ff_dim=False,
        layer_types=["full_attention", "conv", "full_attention"],
    )
    config: PreTrainedConfig = text_config
    if multimodal:
        config = Lfm2VlConfig(
            architectures=["Lfm2VlForConditionalGeneration"],
            text_config=text_config,
            vision_config=Siglip2VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_patches=16,
                patch_size=4,
            ),
            projector_hidden_size=128,
        )
    config.save_pretrained(tmp_path / "target")
    target = ModelConfig(
        model=str(tmp_path / "target"),
        skip_tokenizer_init=True,
        enforce_eager=True,
        dtype="float32",
    )
    parallel = ParallelConfig()
    speculative = None
    if num_spec:
        Qwen3Config(
            architectures=["Qwen3DSparkModel"],
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            block_size=num_spec,
            dflash_config={"target_layer_ids": [0, 1], "mask_token_id": 3},
        ).save_pretrained(tmp_path / "draft")
        speculative = SpeculativeConfig(
            model=str(tmp_path / "draft"),
            method="dspark",
            target_model_config=target,
            target_parallel_config=parallel,
        )
    return VllmConfig(
        model_config=target,
        parallel_config=parallel,
        speculative_config=speculative,
        cache_config=CacheConfig(
            block_size=16, mamba_block_size=128, enable_prefix_caching=False
        ),
        attention_config=AttentionConfig(backend="TRITON_ATTN"),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            max_num_seqs=1,
            max_num_batched_tokens=128,
        ),
    )


@pytest.mark.parametrize("multimodal", [False, True], ids=["dense", "vl"])
def test_lfm2_captures_full_streams_at_helper_indices(tmp_path, dist_init, multimodal):
    """Capture real attention/conv residual streams without altering final output."""
    config = _make_config(tmp_path, multimodal, num_spec=9)
    device = torch.device("cuda")
    with set_current_vllm_config(config), torch.device(device):
        target = (
            Lfm2VLForConditionalGeneration(vllm_config=config)
            if multimodal
            else Lfm2ForCausalLM(vllm_config=config)
        )
    language_model = (
        target.get_language_model()
        if isinstance(target, Lfm2VLForConditionalGeneration)
        else target
    )
    assert isinstance(language_model, Lfm2ForCausalLM)
    backbone = language_model.model
    torch.manual_seed(0)
    for parameter in target.parameters():
        nn.init.normal_(parameter, std=0.02)

    num_tokens = 12
    query_start_loc_cpu = torch.tensor([0, num_tokens], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.to(device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_cpu.to(device),
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        is_prefilling=torch.tensor([True]),
        num_reqs=1,
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        max_seq_len=num_tokens,
        block_table_tensor=torch.tensor([[1]], dtype=torch.int32, device=device),
        slot_mapping=torch.arange(num_tokens, device=device) + 16,
    )
    metadata = {}
    slot_mapping = {}
    with set_current_vllm_config(config):
        for name, layer in config.compilation_config.static_forward_context.items():
            spec = layer.get_kv_cache_spec(config)
            assert spec is not None
            cache = torch.zeros(
                compute_layer_kv_cache_shape_bytes(spec, num_blocks=2),
                dtype=torch.uint8,
                device=device,
            )
            layer.bind_kv_cache(
                cache if isinstance(layer, ShortConv) else cache.view(spec.dtype)
            )
            builder = layer.get_attn_backend().get_builder_cls()(
                spec, [name], config, device
            )
            metadata[name] = builder.build(0, common)
            slot_mapping[name] = common.slot_mapping

    boundaries: list[torch.Tensor] = []
    hidden_only: list[torch.Tensor] = []

    def observe_layer(module: nn.Module, args, output):
        hidden, residual = output
        boundaries.append((hidden + residual).clone())
        hidden_only.append(hidden.clone())

    input_ids = torch.arange(num_tokens, device=device)
    positions = torch.arange(num_tokens, device=device)
    with (
        ExitStack() as hooks,
        torch.inference_mode(),
        set_forward_context(metadata, config, slot_mapping=slot_mapping),
    ):
        for layer in backbone.layers:
            hooks.callback(layer.register_forward_hook(observe_layer).remove)
        baseline = target(input_ids=input_ids, positions=positions)
        assert isinstance(baseline, torch.Tensor)
        boundaries.clear()
        hidden_only.clear()
        assert config.speculative_config is not None
        set_eagle3_aux_hidden_state_layers(target, config.speculative_config)
        assert backbone.aux_hidden_state_layers == (1, 2)
        result = target(
            input_ids=None,
            positions=positions,
            inputs_embeds=language_model.embed_input_ids(input_ids),
        )
        assert isinstance(result, tuple)
        output, aux_hidden_states = result

    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    assert len(aux_hidden_states) == 2
    assert len(boundaries) == len(hidden_only) == 3
    for captured, full, hidden in zip(aux_hidden_states, boundaries, hidden_only):
        torch.testing.assert_close(captured, full, rtol=0, atol=0)
        assert not torch.equal(captured, hidden)
    if isinstance(target, Lfm2VLForConditionalGeneration):
        assert target.get_language_model() is target.language_model
        torch.testing.assert_close(
            target.embed_input_ids(input_ids),
            language_model.embed_input_ids(input_ids),
            rtol=0,
            atol=0,
        )
        assert get_target_lm_head(target, language_model) is language_model.lm_head
        assert language_model.lm_head.weight is backbone.embed_tokens.weight


@pytest.mark.parametrize("num_spec", [0, 9])
@pytest.mark.parametrize("multimodal", [False, True], ids=["dense", "vl"])
def test_lfm2_cache_planning_includes_speculative_conv_slots(
    tmp_path, dist_init, multimodal, num_spec
):
    """Planned state must fit the runtime convolution's speculative history."""
    config = _make_config(tmp_path, multimodal, num_spec)
    hf_config = config.model_config.hf_text_config
    with set_current_vllm_config(config):
        conv = ShortConv(
            config=hf_config,
            dim=hf_config.hidden_size,
            layer_idx=1,
            model_config=config.model_config,
            cache_config=config.cache_config,
            prefix="model.layers.1.short_conv",
        )
    model_class = Lfm2VLForConditionalGeneration if multimodal else Lfm2ForCausalLM
    planned = model_class.get_mamba_state_shape_from_config(config)
    assert planned == conv.get_state_shape()
    assert prod(planned[0]) == hf_config.hidden_size * (2 + num_spec)
