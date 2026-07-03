# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config.speculative import SpeculativeConfig


def test_dspark_standard_rejection_uses_probabilistic_draft_sampling(monkeypatch):
    draft_model_config = SimpleNamespace(
        model="DeepSeek-V4-Flash-DSpark-hybrid",
        architectures=["Qwen3DSparkModel"],
        hf_config=SimpleNamespace(model_type="deepseek_v4"),
        max_model_len=4096,
        verify_with_parallel_config=lambda parallel_config: None,
    )
    target_model_config = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V4-Flash",
        quantization=None,
        tokenizer="deepseek-ai/DeepSeek-V4-Flash",
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype="auto",
        seed=0,
        tokenizer_revision=None,
        max_model_len=4096,
        enforce_eager=False,
        max_logprobs=20,
        config_format="auto",
    )
    target_parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        distributed_executor_backend=None,
        max_parallel_loading_workers=None,
        disable_custom_all_reduce=False,
        ray_workers_use_nsight=False,
        placement_group=None,
    )
    monkeypatch.setattr(
        "vllm.config.speculative.ModelConfig",
        lambda **kwargs: draft_model_config,
    )

    speculative_config = SpeculativeConfig(
        method="dspark",
        num_speculative_tokens=5,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        target_model_config=target_model_config,
        target_parallel_config=target_parallel_config,
    )

    assert speculative_config.draft_sample_method == "probabilistic"
