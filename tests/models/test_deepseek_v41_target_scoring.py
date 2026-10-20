# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-token scores must survive DeepSeek-V4.1 decoder replay."""

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ),
    reason="DeepSeek-V4.1 FlashMLA requires SM90 or SM100",
)

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
LOAD_FORMAT = "dsv41_target_scoring_random"
SMALL_CONFIG = {
    "vocab_size": 4096,
    "hidden_size": 1024,
    "moe_intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 16,
    "q_lora_rank": 256,
    "o_lora_rank": 128,
    "o_groups": 4,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "compress_ratios": [0, 0, 2, 1, 1, 1],
    "kv_source_layer_ids": [2, 3],
    "index_source_layer_ids": [2, 3],
    "candidate_source_layer_id": 3,
    "candidate_topk_blocks": 128,
    "index_topk": 512,
    "index_n_heads": 8,
    "engram_layer_ids": [],
    "engram_num_embeddings": [],
    "num_nextn_predict_layers": 0,
    "dspark_target_layer_ids": [],
    "dspark_noise_token_id": 4094,
    "quantization_config": None,
    "image_token_id": 4095,
}


@register_model_loader(LOAD_FORMAT)
class ScoringWeightLoader(DummyModelLoader):
    def load_weights(self, model, model_config) -> None:
        super().load_weights(model, model_config)
        # Tiny dummy norm gains can hide missing rows behind uniform scores.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, RMSNorm) and module.weight is not None:
                    module.weight.fill_(1)
            head = model.language_model.lm_head.weight
            generator = torch.Generator(device=head.device).manual_seed(20261020)
            head.normal_(0, 0.02, generator=generator)


class ScoringWorker:
    """Import the loader in workers and confirm the real replay path is enabled."""

    def replay_window(self) -> int:
        runner = self.model_runner  # type: ignore[attr-defined]
        replay = runner.model_state.decoder_replay_layers
        assert replay is not None
        return replay.window


@pytest.mark.parametrize("budget", [300, 150])
def test_target_scores_match_full_row_reference(vllm_runner, monkeypatch, budget):
    """Check discarded and retained rows, with and without chunked prefill."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    token_ids = [10 + (i * 37) % 4000 for i in range(300)]
    prompt = {"prompt_token_ids": token_ids}

    with vllm_runner(
        MODEL,
        revision=REVISION,
        hf_overrides=SMALL_CONFIG,
        load_format=LOAD_FORMAT,
        worker_extension_cls=f"{__name__}.ScoringWorker",
        skip_tokenizer_init=True,
        limit_mm_per_prompt={"image": 0},
        dtype="bfloat16",
        seed=1234,
        enforce_eager=True,
        block_size=None,
        max_model_len=1024,
        max_num_batched_tokens=budget,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.3,
        kv_cache_memory_bytes=1024**3,
        attention_config={"backend": "FLASHMLA_SPARSE_DSV41"},
    ) as runner:
        assert runner.llm.collective_rpc("replay_window") == [128]
        for start in (0, 200):
            actual_tokens = np.asarray(token_ids[start + 1 :], dtype=np.int32)
            candidates = np.column_stack(
                (
                    actual_tokens,
                    np.full_like(actual_tokens, 3),
                    np.full_like(actual_tokens, 3),
                    np.full_like(actual_tokens, -1),
                )
            )
            outputs = []
            for reference in (True, False):
                params = SamplingParams(
                    max_tokens=1,
                    temperature=0,
                    ignore_eos=True,
                    prompt_logprobs=0 if reference else None,
                    prompt_logprob_start=start,
                    prompt_logprob_token_ids=candidates,
                )
                outputs.append(runner.llm.generate(prompt, params, use_tqdm=False)[0])
            ref, target = outputs
            reference_scores = ref.prompt_token_id_logprobs
            target_scores = target.prompt_token_id_logprobs
            assert reference_scores is not None and target_scores is not None
            for scores in (reference_scores, target_scores):
                assert scores.shape == (299 - start, 4)
                assert np.isfinite(scores[:, :3]).all()
                assert np.isneginf(scores[:, 3]).all()
                np.testing.assert_array_equal(scores[:, 1], scores[:, 2])
            ordinary = np.asarray(
                [
                    ref.prompt_logprobs[i + 1][token_ids[i + 1]].logprob
                    for i in range(start, 299)
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                reference_scores[:, 0], ordinary, rtol=0, atol=5e-4
            )
            np.testing.assert_allclose(
                target_scores[:, :3], reference_scores[:, :3], rtol=0, atol=5e-4
            )
