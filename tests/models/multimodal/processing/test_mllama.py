# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mllama's multimodal preprocessing and profiling."""
import pytest
from transformers import MllamaConfig

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.profiling import MultiModalProfiler

from ...utils import build_model_context


@pytest.mark.parametrize("model_id",
                         ["meta-llama/Llama-3.2-11B-Vision-Instruct"])
@pytest.mark.parametrize("max_model_len", [4096, 8192, 25600, 131072])
@pytest.mark.parametrize("max_num_seqs", [1, 2, 8])
def test_profiling(
    model_id: str,
    max_model_len: int,
    max_num_seqs: int,
):
    # regression test for https://github.com/vllm-project/vllm/issues/13929
    from vllm.model_executor.models.mllama import calc_token_per_chunk

    model_config_kwargs = {
        "max_model_len": max_model_len,
    }
    ctx = build_model_context(
        model_id,
        model_config_kwargs=model_config_kwargs,
        limit_mm_per_prompt={"image": 1},
    )

    mm_config = ctx.get_mm_config()
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    profiler = MultiModalProfiler(processor)

    dummy_encoder_data = profiler.get_encoder_dummy_data(
        max_model_len,
        mm_counts=mm_config.limit_per_prompt,
    )
    dummy_mm_data = processor.dummy_inputs.get_dummy_processor_inputs(
        max_model_len,
        mm_counts=mm_config.limit_per_prompt,
    )

    hf_config = ctx.get_hf_config(MllamaConfig)
    image_size = hf_config.vision_config.image_size
    encoder_seq_lens = [len(dummy_encoder_data.prompt_token_ids)
                        ] * max_num_seqs

    mm_kwargs = processor.apply(
        prompt=dummy_mm_data.prompt,
        mm_data=dummy_mm_data.mm_data,
        hf_processor_mm_kwargs=dict(),
    )["mm_kwargs"]

    # Get the actual number of encoder tokens for each sample.
    # Because attn_metadata.encoder_seq_lens only counts the last
    # group of images for each sample, which is used to cheat the
    # block manager to allocate blocks for those images only.
    # See MllamaMultiModalProcessor for more details.
    num_tiles = [[t] for t in mm_kwargs.pop("num_tiles")]
    num_tokens_per_tile = calc_token_per_chunk(image_size)
    actual_encoder_seq_lens = [
        sum(num_tile) * num_tokens_per_tile for num_tile in num_tiles
    ]

    # simulate mllama image-present prefill.
    for actual_len, last_group_len in zip(actual_encoder_seq_lens,
                                          encoder_seq_lens):
        assert actual_len >= last_group_len
