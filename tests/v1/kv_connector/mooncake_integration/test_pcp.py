# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-only PCP over real Mooncake TCP transport; requires three GPUs."""

import concurrent.futures
import json
import os
import uuid

import pytest
import requests

from tests.models.utils import check_logprobs_close
from tests.utils import RemoteOpenAIServer
from vllm.utils.network_utils import get_open_ports_list


@pytest.mark.distributed
def test_pcp_producer_transfers_replicated_kv():
    """One sender must serve chunked requests without falling back to prefill."""
    model = os.environ.get(
        "MOONCAKE_PCP_TEST_MODEL", "hmellor/tiny-random-DeepseekV2ForCausalLM"
    )
    prefill_port, decode_port, bootstrap_port = get_open_ports_list(3)
    args = [
        "--served-model-name",
        "pcp-test",
        "--host",
        "127.0.0.1",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--max-num-seqs",
        "8",
        "--block-size",
        "64",
        "--num-gpu-blocks-override",
        "128",
        "--gpu-memory-utilization",
        "0.65",
        "--no-enable-prefix-caching",
    ]
    if model == "hmellor/tiny-random-DeepseekV2ForCausalLM":
        args += ["--revision", "c0146ef3550ef77c6520211184cf41184b92346a"]
    env = {"VLLM_USE_V2_MODEL_RUNNER": "1", "MC_FORCE_TCP": "1"}
    producer = json.dumps(
        {
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"mooncake_protocol": "tcp"},
        }
    )
    consumer = json.dumps(
        {
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_consumer",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"mooncake_protocol": "tcp"},
        }
    )
    with (
        RemoteOpenAIServer(
            model,
            args
            + [
                "--port",
                str(prefill_port),
                "--prefill-context-parallel-size",
                "2",
                "--kv-transfer-config",
                producer,
            ],
            auto_port=False,
            env_dict=env
            | {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(bootstrap_port),
            },
        ) as prefill,
        RemoteOpenAIServer(
            model,
            args + ["--port", str(decode_port), "--kv-transfer-config", consumer],
            auto_port=False,
            env_dict=env | {"CUDA_VISIBLE_DEVICES": "2"},
        ) as decode,
        prefill.get_client() as p_client,
        decode.get_client() as d_client,
    ):
        bootstrap = f"http://127.0.0.1:{bootstrap_port}"
        response = requests.get(f"{bootstrap}/query", timeout=10)
        response.raise_for_status()
        info = response.json()
        assert list(info) == ["0"]
        assert set(info["0"]["worker_addr"]) == {"0"}
        assert set(info["0"]["worker_addr"]["0"]) == {"0"}

        def complete(client, prompt, params=None, max_tokens=16):
            return client.completions.create(
                model="pcp-test",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0,
                seed=42,
                logprobs=5,
                extra_body={
                    "ignore_eos": True,
                    "return_tokens_as_token_ids": True,
                    "kv_transfer_params": params,
                },
            )

        def disaggregated(prompt):
            transfer_id = str(uuid.uuid4())
            complete(
                p_client,
                prompt,
                {"do_remote_decode": True, "transfer_id": transfer_id},
                max_tokens=1,
            )
            return complete(
                d_client,
                prompt,
                {
                    "do_remote_prefill": True,
                    "remote_bootstrap_addr": bootstrap,
                    "remote_engine_id": info["0"]["engine_id"],
                    "transfer_id": transfer_id,
                },
            )

        def hits():
            response = requests.get(decode.url_for("metrics"), timeout=10)
            response.raise_for_status()
            return sum(
                float(line.split()[-1])
                for line in response.text.splitlines()
                if line.startswith("vllm:external_prefix_cache_hits_total{")
            )

        prompts = [
            "The capital of France is",
            "Context: "
            + "The river flows through the quiet valley. " * 160
            + "\nSummarize this context:",
        ]
        reference = [complete(d_client, prompt) for prompt in prompts]
        before = hits()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(disaggregated, prompts * 2))
        assert hits() - before == sum(r.usage.prompt_tokens for r in results)
        for index, result in enumerate(results):
            assert result.usage.completion_tokens == 16
            outputs = []
            for response in (reference[index % len(prompts)], result):
                choice = response.choices[0]
                outputs.append(
                    (choice.logprobs.tokens, choice.text, choice.logprobs.top_logprobs)
                )
            check_logprobs_close(
                outputs_0_lst=[outputs[0]],
                outputs_1_lst=[outputs[1]],
                name_0="standalone",
                name_1="PCP2 P/D",
            )
