import json
import os

import openai
import pytest

from ...utils import RemoteOpenAIServer

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.mark.asyncio
async def test_shutdown_on_engine_failure(tmp_path):
    # Use a bad adapter to crash the engine
    # (This test will fail when that bug is fixed)
    adapter_path = tmp_path / "bad_adapter"
    os.mkdir(adapter_path)
    with open(adapter_path / "adapter_model_config.json", "w") as f:
        json.dump({"not": "real"}, f)
    with open(adapter_path / "adapter_model.safetensors", "wb") as f:
        f.write(b"this is fake")

    # dtype, max-len etc set so that this can run in CI
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        "--enable-lora",
        "--lora-modules",
        f"bad-adapter={tmp_path / 'bad_adapter'}",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        async with remote_server.get_async_client() as client:

            with pytest.raises(
                (openai.APIConnectionError, openai.InternalServerError)):
                # This crashes the engine
                await client.completions.create(model="bad-adapter",
                                                prompt="Hello, my name is")

            # Now the server should shut down
            return_code = remote_server.proc.wait(timeout=3)
            assert return_code is not None
