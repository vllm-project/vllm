# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight regression for NVFP4 P16 -> D64 KV transfers."""

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import openai

from tests.models.utils import check_logprobs_close
from tests.utils import RemoteOpenAIServer
from tests.v1.kv_connector.rocm_pd_accuracy_utils import (
    QWEN_MODEL,
    _metric_total,
    _proxy,
    _side_channel_ports,
)
from vllm.tokenizers import get_tokenizer

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def test_nvfp4_heterogeneous_block_sizes():
    """Transferred KV must preserve candidate tokens for the same prefix."""
    args = [
        "--revision",
        REVISION,
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--attention-backend",
        "FLASHINFER",
        "--kv-cache-dtype",
        "nvfp4",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "4",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.2",
        "--kv-cache-memory-bytes",
        str(512 * 1024**2),
    ]
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_KV_CACHE_LAYOUT": "LBHNC",
    }
    with contextlib.ExitStack() as stack:
        # Both small-model servers fit on one allocated GPU; preserve its selector.
        servers: list[RemoteOpenAIServer] = []
        stack.callback(RemoteOpenAIServer.shutdown_many, servers)
        for role, block_size, port in zip(
            ("kv_producer", "kv_consumer"), (16, 64), _side_channel_ports(2)
        ):
            transfer = {
                "kv_connector": "NixlConnector",
                "kv_role": role,
                "kv_load_failure_policy": "fail",
            }
            servers.append(
                RemoteOpenAIServer(
                    QWEN_MODEL,
                    [
                        *args,
                        "--block-size",
                        str(block_size),
                        "--kv-transfer-config",
                        json.dumps(transfer),
                    ],
                    env_dict={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(port)},
                )
            )
        prefill, decode = servers
        proxy_url = stack.enter_context(_proxy("nixl", prefill, decode, 0))
        direct = stack.enter_context(decode.get_client())
        remote = stack.enter_context(
            openai.OpenAI(
                api_key="EMPTY", base_url=proxy_url, timeout=120, max_retries=0
            )
        )
        tokenizer = get_tokenizer(QWEN_MODEL, revision=REVISION)
        tokens = tokenizer.encode(
            "The capital of France is Paris. Two plus two equals four. " * 32,
            add_special_tokens=False,
        )
        before = _metric_total(decode, "vllm:nixl_bytes_transferred_sum")
        for length in (63, 64, 65, 112, 129, 193):
            outputs = []
            for client in (direct, remote):
                choice = client.completions.create(
                    model=QWEN_MODEL,
                    prompt=tokens[:length],
                    temperature=0,
                    max_tokens=80,
                    logprobs=5,
                    extra_body={"return_tokens_as_token_ids": True, "ignore_eos": True},
                ).choices[0]
                assert choice.logprobs is not None
                assert choice.logprobs.top_logprobs is not None
                assert len(choice.logprobs.tokens) == 80
                # Wrap positions separately: the helper stops after its first
                # logprob check even if the generated tokens still match.
                outputs.append(
                    [
                        ([token], "", [logprobs])
                        for token, logprobs in zip(
                            choice.logprobs.tokens,
                            choice.logprobs.top_logprobs,
                            strict=True,
                        )
                    ]
                )
            for local_output, remote_output in zip(*outputs, strict=True):
                check_logprobs_close(
                    outputs_0_lst=[local_output],
                    outputs_1_lst=[remote_output],
                    name_0=f"local NVFP4, prompt length {length}",
                    name_1=f"NIXL P16 -> D64, prompt length {length}",
                    always_check_logprobs=True,
                )
                if local_output[0] != remote_output[0]:
                    # Later positions have different autoregressive histories.
                    break

        # Counters are published asynchronously; require transport, not recomputation.
        deadline = time.monotonic() + 15
        while _metric_total(decode, "vllm:nixl_bytes_transferred_sum") <= before:
            assert time.monotonic() < deadline, "No NVFP4 KV bytes were transferred"
            time.sleep(0.2)
        assert _metric_total(decode, "vllm:nixl_num_failed_transfers_total") == 0
