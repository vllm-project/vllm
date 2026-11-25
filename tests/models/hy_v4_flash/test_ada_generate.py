# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.models.hy_v4_flash.configs import (
    SLIDING_ATTENTION,
    SPARSE_ATTENTION,
    HYV4FlashConfig,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(89),
    reason="HY4 Flash Ada integration requires SM89",
)


def _write_model(path, *, dfly=False):
    layers = [SPARSE_ATTENTION, SLIDING_ATTENTION]
    if dfly:
        # Draft layers must reuse a sparse target layer after layer zero.
        layers.insert(1, SPARSE_ATTENTION)
    config = HYV4FlashConfig(
        vocab_size=2048,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=len(layers),
        num_attention_heads=32,
        num_key_value_heads=32,
        swa_num_attention_heads=64,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        max_position_embeddings=4096,
        layer_types=layers,
        mlp_layer_types=["dense"] * len(layers),
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        learnable_sink=True,
        gated_mla=True,
        gating_type="elementwise",
        engram_layer_ids=None,
        enable_ihc=False,
    )
    config.architectures = ["HYV4FlashForCausalLM"]
    if dfly:
        config.update(
            {
                "dfly_num_hidden_layers": 1,
                "dfly_mask_token_id": config.vocab_size - 1,
                "dfly_target_index_layer": [1],
                "dfly_target_kv_layers": [1],
                "dfly_block_size": 4,
            }
        )
    config.save_pretrained(path)


def _ada_llm(path, **kwargs):
    return LLM(
        model=str(path),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        enforce_eager=True,
        attention_backend="TRITON_MLA_SPARSE",
        kv_cache_dtype="auto",
        block_size=64,
        max_model_len=3072,
        max_num_batched_tokens=512,
        long_prefill_token_threshold=256,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        **kwargs,
    )


def test_ada_hybrid_layers_generate_after_chunked_prefill(tmp_path):
    """Compare isolated requests to mixed prefill/decode across both KV types."""
    _write_model(tmp_path)
    llm = _ada_llm(tmp_path, max_num_seqs=2)
    try:
        prompts = [
            TokensPrompt(prompt_token_ids=[3 + offset + i % 97 for i in range(n)])
            for offset, n in [(0, 701), (257, 2305)]
        ]
        params = SamplingParams(
            max_tokens=3, temperature=0, seed=17, ignore_eos=True, logprobs=5
        )

        def results(requests):
            output = []
            for request in llm.generate(requests, params, use_tqdm=False):
                result = request.outputs[0]
                assert len(result.token_ids) == 3 and result.finish_reason == "length"
                assert result.logprobs is not None
                assert all(
                    math.isfinite(scores[token].logprob)
                    for scores, token in zip(result.logprobs, result.token_ids)
                )
                output.append((result.token_ids, result.text, result.logprobs))
            return output

        isolated = [results([prompt])[0] for prompt in prompts]
        # Distinct contexts should give distinguishable token candidates.
        assert set(isolated[0][2][0]) != set(isolated[1][2][0])
        check_logprobs_close(
            outputs_0_lst=isolated,
            outputs_1_lst=results(prompts),
            name_0="isolated",
            name_1="mixed-length batch",
            always_check_logprobs=True,
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


def test_ada_dfly_verifies_drafts_like_target_only(tmp_path):
    """DFly must actually propose and verify tokens without changing the target."""
    _write_model(tmp_path, dfly=True)
    prompt = TokensPrompt(prompt_token_ids=[3 + i % 97 for i in range(701)])
    params = SamplingParams(
        max_tokens=7, temperature=0, seed=17, ignore_eos=True, logprobs=5
    )

    def run(spec_config):
        llm = _ada_llm(
            tmp_path,
            max_num_seqs=1,
            speculative_config=spec_config,
            disable_log_stats=False,
        )
        try:
            output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
            assert len(output.token_ids) == 7 and output.finish_reason == "length"
            assert output.logprobs is not None
            if spec_config:
                metrics = llm.get_metrics()
                for kind in ("draft", "verified"):
                    assert any(
                        counter.name == f"vllm:spec_decode_num_{kind}_tokens"
                        and counter.value > 0
                        for counter in metrics
                    )
            return [(output.token_ids, output.text, output.logprobs)]
        finally:
            llm.llm_engine.engine_core.shutdown()

    actual = run({"method": "dfly", "num_speculative_tokens": 3})
    check_logprobs_close(
        outputs_0_lst=run(None),
        outputs_1_lst=actual,
        name_0="target-only",
        name_1="DFly",
        always_check_logprobs=True,
    )
