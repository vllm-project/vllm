# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.utils import RemoteOpenAIServer


def test_remote_decode_without_bidirectional_transfer_completes():
    """One-way serving must finish even with D-to-P block metadata."""
    model = "Qwen/Qwen3-0.6B"
    args = [
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}',
    ]
    with (
        RemoteOpenAIServer(
            model, args, env_dict={"VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "0"}
        ) as server,
        server.get_client(timeout=30, max_retries=0) as client,
    ):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hello " * 128}],
            max_tokens=1,
            temperature=0,
            extra_body={
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "remote_block_ids": [[1, 2, 3]],
                    "remote_engine_id": "x",
                    "remote_request_id": "x",
                    "remote_host": "127.0.0.1",
                    "remote_port": 1,
                    "remote_num_tokens": 4096,
                    "tp_size": 1,
                }
            },
        )
        assert response.usage.completion_tokens == 1
        ordinary = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0,
        )
        assert ordinary.usage.completion_tokens == 1
