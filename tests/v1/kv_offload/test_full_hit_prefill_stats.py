# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from vllm import SamplingParams, TokensPrompt, envs
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter, Histogram

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)


def _counter(metrics, name, source=None):
    return sum(
        metric.value
        for metric in metrics
        if isinstance(metric, Counter)
        and metric.name == f"vllm:{name}"
        and (source is None or metric.labels.get("source") == source)
    )


def _prefill_computed(metrics):
    return sum(
        metric.sum
        for metric in metrics
        if isinstance(metric, Histogram)
        and metric.name == "vllm:request_prefill_kv_computed_tokens"
    )


def test_full_async_offload_hit_accounts_for_last_token(
    vllm_runner, tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "0")
    envs.disable_envs_cache()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=512,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=128,
            )
        )
        model.save_pretrained(tmp_path)
        del model

    with vllm_runner(
        str(tmp_path),
        skip_tokenizer_init=True,
        disable_log_stats=False,
        dtype="float16",
        enforce_eager=True,
        attention_config={"backend": "TRITON_ATTN"},
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        block_size=16,
        num_gpu_blocks_override=64,
        kv_cache_memory_bytes=16 << 20,
        enable_prefix_caching=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"cpu_bytes_to_use": 16 << 20},
        ),
    ) as runner:
        llm = runner.get_llm()
        params = SamplingParams(
            temperature=0, max_tokens=1, ignore_eos=True, detokenize=False
        )

        for length, loaded in [(64, 64), (63, 48)]:
            # Different first blocks keep the two cases independent.
            prompt = TokensPrompt(
                prompt_token_ids=[
                    3 + (length * 7 + i * 37) % 500 for i in range(length)
                ]
            )
            store_before = _counter(llm.get_metrics(), "kv_offload_store_bytes")
            cold = llm.generate(prompt, params, use_tqdm=False)[0]
            assert cold.num_cached_tokens == 0

            deadline = time.monotonic() + 30
            while not (
                _counter(llm.get_metrics(), "kv_offload_store_bytes") > store_before
                and llm.reset_prefix_cache(reset_connector=False)
            ):
                assert time.monotonic() < deadline, "KV offload store did not finish"
                llm.llm_engine.step()

            before = llm.get_metrics()
            repeat = llm.generate(prompt, params, use_tqdm=False)[0]
            after = llm.get_metrics()
            expected_compute = max(1, length - loaded)

            assert _counter(after, "kv_offload_load_bytes") > _counter(
                before, "kv_offload_load_bytes"
            )
            for name, source, expected in [
                ("external_prefix_cache_hits", None, loaded),
                ("prompt_tokens_cached_by_source", "host", loaded),
                ("prompt_tokens_cached_by_source", "device", 0),
                ("prompt_tokens_by_source", "local_compute", expected_compute),
                (
                    "prompt_tokens_by_source",
                    "external_kv_transfer",
                    length - expected_compute,
                ),
            ]:
                assert _counter(after, name, source) - _counter(
                    before, name, source
                ) == (expected), (name, source)
            assert repeat.num_cached_tokens == length - expected_compute
            assert (
                _prefill_computed(after) - _prefill_computed(before) == expected_compute
            )
            # Neither the one-token rewind nor the partial tail creates a full block.
            assert repeat.num_cache_creation_tokens == 0
