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
    os.mkdir(tmp_path / "bad_adapter")
    with open(tmp_path / "bad_adapter" / "adapter_model_config.json",
              "w") as f:
        json.dump({"not": "real"}, f)
    with open(tmp_path / "bad_adapter" / "adapter_model.safetensors",
              "wb") as f:
        f.write(b"this is fake")

    args = [
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--enable-lora",
        "--lora-modules",
        f"bad-adapter={tmp_path / 'bad_adapter'}",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.APIConnectionError):
            # This crashes the engine
            await client.completions.create(model="bad-adapter",
                                            prompt="Hello, my name is")

        # Now the server should shut down
        rc = remote_server.proc.wait(timeout=1)
        assert rc is not None
