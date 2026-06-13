# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that a compressed-tensors WNA16-INT quantized input embedding is
dispatched to ``CompressedTensorsEmbeddingWNA16Int`` and loads/runs.

This guards the model-side plumbing: an input ``VocabParallelEmbedding`` is only
quantized if the model passes ``quant_config`` (and ``prefix``) to it. Without
that, a checkpoint with a quantized ``embed_in``/``embed_tokens`` silently falls
back to an unquantized embedding and fails to load.

Run `pytest tests/quantization/test_quantized_embedding.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import (  # noqa: E501
    CompressedTensorsEmbeddingWNA16Int,
)

# Tiny GPTNeoX checkpoint with `embed_in` quantized to WNA16-INT (W4 group64,
# class-based "Embedding" target), produced with llm-compressor.
MODEL_ID = "kkothuri/pythia-70m-emb-w4g64-ct"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="quantized embedding kernel requires CUDA"
)
def test_quantized_embedding_dispatch(vllm_runner, monkeypatch) -> None:
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        MODEL_ID, dtype=torch.float16, max_model_len=2048, enforce_eager=True
    ) as vllm_model:

        def check_model(model):
            embed = model.gpt_neox.embed_in
            assert isinstance(embed.quant_method, CompressedTensorsEmbeddingWNA16Int)

        vllm_model.apply_model(check_model)

        # Smoke test: the dequant-gather embedding path runs end-to-end.
        print(vllm_model.generate_greedy(["Hello my name is"], max_tokens=4)[0][1])
