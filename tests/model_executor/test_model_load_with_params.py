import os

import torch

MAX_MODEL_LEN = 128
MODEL_NAME = os.environ.get("MODEL_NAME",
                            "sentence-transformers/all-MiniLM-L12-v2")
REVISION = os.environ.get("REVISION", "main")


def test_model_loading_with_params(vllm_runner):
    """
    Test parameter weight loading with tp>1.
    """
    with vllm_runner(model_name=MODEL_NAME,
                     revision=REVISION,
                     dtype=torch.half if QUANTIZATION == "gptq" else "auto",
                     max_model_len=MAX_MODEL_LEN,
                     tensor_parallel_size=2) as model:

        output = model.encode("Write a short story about a robot that dreams for the first time.\n")
        print(output)
        assert output
