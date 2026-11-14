# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving regressions for the Ray data-parallel backend."""

from ipaddress import ip_address

import pytest
import ray

from tests.utils import RemoteOpenAIServer


@pytest.mark.parametrize("explicit_address", [False, True], ids=["derived", "explicit"])
def test_ray_dp_size_one_serves(explicit_address: bool) -> None:
    """DP=1 must retain the Ray node address through startup and generation."""
    model = "hmellor/tiny-random-LlamaForCausalLM"
    max_tokens = 4

    with ray.init(address="local", include_dashboard=False) as cluster:
        node_ip = ray.util.get_node_ip_address()
        assert not ip_address(node_ip).is_loopback, (
            "This regression needs a non-loopback Ray node address."
        )
        cluster_address = cluster.address_info["address"]
        assert cluster_address is not None

        args = [
            "--data-parallel-backend",
            "ray",
            "--data-parallel-size",
            "1",
            "--enforce-eager",
            "--no-async-scheduling",
            "--max-model-len",
            "128",
            "--max-num-seqs",
            "1",
        ]
        if explicit_address:
            args.extend(["--data-parallel-address", node_ip])

        # The old DP=1 fallback overwrites both CLI-selected addresses with
        # loopback, which is absent from this real Ray cluster's node resources.
        env_dict = {
            "RAY_ADDRESS": cluster_address,
            "VLLM_HOST_IP": node_ip,
            "VLLM_DP_MASTER_IP": "127.0.0.1",
        }
        with RemoteOpenAIServer(model, args, env_dict=env_dict) as server:
            with server.get_client() as client:
                completion = client.completions.create(
                    model=model,
                    prompt="Hello",
                    max_tokens=max_tokens,
                    temperature=0,
                    extra_body={"ignore_eos": True, "skip_special_tokens": False},
                )

            assert len(completion.choices) == 1
            assert completion.choices[0].text
            assert completion.choices[0].finish_reason == "length"
            assert completion.usage is not None
            assert completion.usage.completion_tokens == max_tokens
