# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


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
        hf_overrides={},
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


def _dspark_configs(block_size: int):
    draft_model_config = SimpleNamespace(
        model="DeepSeek-V4-Flash-0731",
        architectures=["DSparkDraftModel"],
        hf_config=SimpleNamespace(
            model_type="deepseek_v4", dspark_block_size=block_size
        ),
        max_model_len=4096,
        verify_with_parallel_config=lambda parallel_config: None,
    )
    target_model_config = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        quantization=None,
        tokenizer="deepseek-ai/DeepSeek-V4-Flash-0731",
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype="auto",
        seed=0,
        tokenizer_revision=None,
        hf_overrides={},
        max_model_len=4096,
        enforce_eager=False,
        max_logprobs=20,
        config_format="auto",
        hf_config=SimpleNamespace(
            model_type="deepseek_v4", dspark_block_size=block_size
        ),
    )
    target_parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        distributed_executor_backend="mp",
        max_parallel_loading_workers=None,
        disable_custom_all_reduce=False,
        ray_workers_use_nsight=False,
        placement_group=None,
        decode_context_parallel_size=1,
    )
    return draft_model_config, target_model_config, target_parallel_config


def test_dspark_warns_num_speculative_tokens_above_block_size(caplog_vllm):
    """nst > block_size works but wastes drafts, so it warns instead of raising.

    The validator briefly required ==, which would have rejected configs users
    demonstrably run (nst=7 against block_size=5 on vllm-project/vllm#41834,
    normal accept curve). Measured on 2x GB10 with DeepSeek-V4-Flash-0731
    (block size 5): positions 5 and 6 accepted 0.000 in every sample and mean
    acceptance fell 2.19 -> 2.03 (probabilistic), 2.11 -> 1.75 (greedy) — a
    footgun worth a warning, not a startup error. nst < block_size stays an
    error (garbled output, covered by the companion test)."""
    import logging

    _, target_model_config, target_parallel_config = _dspark_configs(5)
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        config = SpeculativeConfig(
            method="dspark",
            num_speculative_tokens=7,
            target_model_config=target_model_config,
            target_parallel_config=target_parallel_config,
        )
    assert config.num_speculative_tokens == 7
    assert "never be accepted" in caplog_vllm.text


def test_dspark_sequential_sampling_writes_persistent_draft_logits(monkeypatch):
    num_reqs = 2
    num_speculative_steps = 3
    vocab_size = 4
    max_num_reqs = 5
    state_ids = torch.tensor([3, 1], dtype=torch.int32)

    speculator = object.__new__(DSparkSpeculator)
    speculator.num_speculative_steps = num_speculative_steps
    speculator.sample_indices = torch.arange(num_reqs * num_speculative_steps)
    speculator.sample_idx_mapping = state_ids.repeat_interleave(num_speculative_steps)
    speculator.sample_pos = torch.arange(10, 10 + num_reqs * num_speculative_steps)
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(max_num_reqs * num_speculative_steps)
    )
    speculator._anchor_idx = torch.tensor([0, num_speculative_steps])
    speculator.temperature = torch.ones(max_num_reqs)
    speculator.seeds = torch.arange(max_num_reqs, dtype=torch.int64)
    speculator.use_fp64_gumbel = False
    speculator._step_cols = torch.arange(num_speculative_steps, dtype=torch.int32)
    speculator._draft_topk = None
    # object.__new__ skips __init__, so every attribute _sample_sequential reads
    # has to be set here. Upstream #47808 added a read of this one; without it
    # the test fails with AttributeError from production code that is in fact
    # correct -- __init__ always sets it. False alarm, not a defect.
    speculator.enable_adaptive_verification = False
    speculator._d2t_scatter_index = None
    speculator.draft_tokens = torch.empty(
        max_num_reqs,
        num_speculative_steps,
        dtype=torch.int64,
    )
    speculator.draft_logits = torch.full(
        (max_num_reqs, num_speculative_steps, vocab_size),
        float("nan"),
    )

    class FakeModel:

        def compute_logits(self, hidden_states):
            return torch.arange(
                hidden_states.shape[0] * vocab_size,
                dtype=torch.float32,
            ).view(hidden_states.shape[0], vocab_size)

        def markov_embed(self, previous_tokens):
            return previous_tokens.to(torch.float32).unsqueeze(-1)

        def markov_bias(self, markov_embed):
            return torch.ones(markov_embed.shape[0], vocab_size)

    sampled_by_step: list[torch.Tensor] = []

    def fake_gumbel_sample(
        logits,
        idx_mapping,
        temperature,
        seeds,
        pos,
        apply_temperature,
        logits_cache=None,
        logits_cache_col=None,
        use_fp64=False,
    ):
        assert logits_cache is speculator.draft_logits
        assert logits_cache_col is not None
        col = int(logits_cache_col.item())
        logits_cache[idx_mapping.long(), col] = logits
        sampled = idx_mapping.to(torch.int64) * 10 + col
        sampled_by_step.append(sampled)
        return sampled

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.gumbel_sample",
        fake_gumbel_sample,
    )
    speculator.model = FakeModel()

    DSparkSpeculator._sample_sequential(
        speculator,
        num_reqs,
        torch.zeros(num_reqs * num_speculative_steps, 1),
    )

    for col in range(num_speculative_steps):
        row_logits = torch.arange(
            num_reqs * num_speculative_steps * vocab_size,
            dtype=torch.float32,
        ).view(num_reqs, num_speculative_steps, vocab_size)[:, col]
        torch.testing.assert_close(
            speculator.draft_logits[state_ids.long(), col],
            row_logits + 1.0,
        )
        torch.testing.assert_close(
            speculator.draft_tokens[:num_reqs, col],
            sampled_by_step[col],
        )

    # The property this asserts -- the draft-logits buffer is never replaced --
    # is now guaranteed by construction rather than by a clearing call.
    # 7d9970dec3 replaced the reuse-and-clear scheme (assign base_logits, then
    # clear_runtime_draft_logits) with a persistent preallocated buffer, because
    # DSpark drafting is CUDA-graph replayed and a Python-side reassignment does
    # not run per replay. The method it called is intentionally gone; this
    # re-checks the same invariant against a second sampling pass instead.
    draft_logits = speculator.draft_logits
    DSparkSpeculator._sample_sequential(
        speculator,
        num_reqs,
        torch.zeros(num_reqs * num_speculative_steps, 1),
    )
    assert speculator.draft_logits is draft_logits
