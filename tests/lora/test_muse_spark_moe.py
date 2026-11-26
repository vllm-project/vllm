# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical regression for LoRA before the Muse Spark post-expert norm."""

import pytest
import torch

from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

from ..utils import multi_gpu_marks


def _save_checkpoints(tmp_path, *, bf16_experts=False):
    from peft import LoraConfig, get_peft_model
    from transformers import MuseSparkConfig, MuseSparkForConditionalGeneration

    config = MuseSparkConfig(
        text_config=dict(
            vocab_size=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            hidden_size=256,
            num_hidden_layers=2,
            head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            num_local_experts=16,
            moe_hidden_size=128,
            moe_intermediate_size=128,
            num_shared_experts=0,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            qk_scale_factor=2.0,
            output_multiplier=1.0,
            sliding_window=32,
        ),
        vision_config=dict(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            pos_emb_height=8,
            pos_emb_width=8,
            layer_types=["full_attention"],
        ),
        audio_config=dict(
            hidden_sizes=[16, 32, 64, 128], depths=[2] * 4, out_hidden_size=256
        ),
        image_token_id=124,
        video_token_id=125,
        audio_token_id=126,
        out_hidden_size=512,
        projector_hidden_size=128,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        model = MuseSparkForConditionalGeneration(config)
        if bf16_experts:
            # PEFT uses the base expert dtype for both merge rounding steps.
            for layer in model.model.language_model.layers:
                layer.mlp.experts.gate_up_proj.data = (
                    layer.mlp.experts.gate_up_proj.data.bfloat16()
                )
                layer.mlp.experts.down_proj.data = (
                    layer.mlp.experts.down_proj.data.bfloat16()
                )
        model.save_pretrained(tmp_path / "base")
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=[],
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            ),
        )
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_" in name:
                    param.normal_(std=0.25)
        model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
        model.merge_and_unload().save_pretrained(tmp_path / "merged")


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="Muse Spark unquantized Triton experts run on Hopper",
)
@pytest.mark.parametrize(
    "tp_size, factor_dtype",
    [
        (1, torch.bfloat16),
        pytest.param(2, torch.float32, marks=multi_gpu_marks(num_gpus=2)),
    ],
)
def test_expert_lora_matches_merged_weights(
    vllm_runner, tmp_path, tp_size, factor_dtype
):
    merge_order = factor_dtype == torch.float32
    _save_checkpoints(tmp_path, bf16_experts=merge_order)
    options = dict(
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        language_model_only=True,
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=32,
        max_num_seqs=4,
        max_logprobs=128,
        kv_cache_memory_bytes=64 << 20,
    )
    prompts = [{"prompt_token_ids": ids} for ids in ([4, 9, 15, 26], [5, 11, 19])]
    sampling = SamplingParams(temperature=0, max_tokens=1, logprobs=128)

    def probabilities(llm, request=None):
        outputs = llm.generate(prompts, sampling, lora_request=request, use_tqdm=False)
        return torch.tensor(
            [
                [out.outputs[0].logprobs[0][i].logprob for i in range(128)]
                for out in outputs
            ]
        ).exp()

    with vllm_runner(str(tmp_path / "merged"), **options) as runner:
        merged = probabilities(runner.llm)
    with vllm_runner(
        str(tmp_path / "base"),
        enable_lora=True,
        lora_dtype=factor_dtype,
        max_lora_rank=8,
        lora_target_modules=["experts"],
        **options,
    ) as runner:
        inactive = probabilities(runner.llm)
        active = probabilities(
            runner.llm, LoRARequest("experts", 1, str(tmp_path / "adapter"))
        )
        torch.testing.assert_close(probabilities(runner.llm), inactive, atol=0, rtol=0)

    # Total variation over the entire vocabulary, at identical prefixes.
    assert ((active - merged).abs().sum(dim=1) / 2 < 0.01).all()
    if merge_order:
        assert torch.equal(active.argmax(dim=1), merged.argmax(dim=1))
        torch.testing.assert_close(active, merged, atol=0.0001, rtol=0)
    assert ((inactive - merged).abs().sum(dim=1) / 2 > 0.05).all()


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="Muse Spark BF16 expert LoRA uses Hopper",
)
@pytest.mark.parametrize("input_group", [1, 2])
def test_merge_order_projection_mixed_routes(input_group):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.models.muse_spark.nvidia.merge_order_experts import _muse_tile_projection

    # One-hot activations expose the actual rounded weight column in the output.
    # Odd dimensions also exercise the last tile's load and store masks.
    experts, outputs, inputs, rank = 3, 35, 37, 8
    rng = torch.Generator().manual_seed(970094)
    base = torch.randn(experts, outputs, inputs, generator=rng).bfloat16()
    a = torch.randn(1, experts, rank, inputs, generator=rng)
    b = torch.randn(1, experts, outputs, rank, generator=rng)
    merged = base + torch.bmm(b[0], a[0]).bfloat16()
    ids = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1], [1, 2]], dtype=torch.int32)
    active = torch.tensor([True, False, False, True, True])
    groups = ids + active.int()[:, None] * experts
    schedule = moe_align_block_size(groups.cuda(), 16, 2 * experts)
    rows = torch.arange(ids.numel())
    columns = (rows // input_group * 7) % inputs
    x = torch.nn.functional.one_hot(columns[::input_group], inputs).bfloat16().cuda()
    result = torch.empty(ids.numel(), outputs, dtype=torch.bfloat16, device="cuda")
    _muse_tile_projection(
        x,
        base.cuda(),
        (a.cuda(),),
        (b.cuda(),),
        result,
        schedule,
        ids.numel(),
        input_group,
    )
    expected = torch.where(
        active[rows // ids.size(1), None],
        merged[ids.flatten().long(), :, columns],
        base[ids.flatten().long(), :, columns],
    )
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
