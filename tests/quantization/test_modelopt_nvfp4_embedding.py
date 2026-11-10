# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4EmbeddingMethod,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import weights_already_processed
from vllm.utils.torch_utils import set_default_torch_dtype


def _mixed_precision_config() -> ModelOptMixedPrecisionConfig:
    config = ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
                "group_size": 16,
                "quantized_layers": {
                    "model.embed_tokens": {
                        "quant_algo": "W4A16_NVFP4",
                        "group_size": 16,
                    }
                },
            }
        }
    )
    assert isinstance(config, ModelOptMixedPrecisionConfig)
    return config


def test_nvfp4_embedding_matches_reference_dequant(dist_init):
    torch.manual_seed(0)
    # The layer pads the vocabulary, so a size that is not a multiple of the
    # padding also covers the loader's narrowing.
    vocab, hidden = 37, 64
    layer = VocabParallelEmbedding(
        vocab,
        hidden,
        params_dtype=torch.bfloat16,
        quant_config=_mixed_precision_config(),
        prefix="model.embed_tokens",
    )
    assert isinstance(layer.quant_method, ModelOptNvFp4EmbeddingMethod)

    weight = torch.randint(0, 256, (vocab, hidden // 2), dtype=torch.uint8)
    scale = torch.rand(vocab, hidden // 16).mul(4).to(torch.float8_e4m3fn)
    # ModelOpt checkpoints store the global scale as a 0-dim tensor.
    global_scale = torch.tensor(0.03)
    for name, value in (
        ("weight", weight),
        ("weight_scale", scale),
        ("weight_scale_2", global_scale),
    ):
        layer.weight_loader(getattr(layer, name), value)
    # The loader post-processes under the model dtype; reloads do not.
    with set_default_torch_dtype(torch.bfloat16):
        layer.quant_method.process_weights_after_loading(layer)

    ids = torch.tensor([[0, 5, 36], [5, 5, 1]])
    table = break_fp4_bytes(weight, torch.float32).view(vocab, -1, 16)
    table = table * (scale.float() * global_scale).unsqueeze(-1)
    expected = table.view(vocab, hidden)[ids].to(torch.bfloat16)
    torch.testing.assert_close(layer(ids), expected, rtol=0, atol=0)

    # A reload reruns post-processing; captured CUDA graphs keep the old address.
    lut = layer.e2m1_lut.data_ptr()
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.e2m1_lut.data_ptr() == lut

    # Fast-start loading hands over already-processed weights.
    assert layer.quant_method.supports_pre_processed_weights
    with weights_already_processed():
        layer.quant_method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer(ids), expected, rtol=0, atol=0)

    # A model that reuses the embedding as its LM head gets the tie error.
    with pytest.raises(NotImplementedError, match="cannot be tied"):
        layer.quant_method.apply(layer, torch.zeros(1, hidden, dtype=torch.bfloat16))


def test_nvfp4_embedding_rejects_tied_lm_head(dist_init):
    # OPT ties its word embeddings to the LM head, which would read packed bytes.
    vllm_config = VllmConfig(model_config=ModelConfig("facebook/opt-125m"))
    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(ValueError, match="cannot be tied"),
    ):
        VocabParallelEmbedding(
            64,
            64,
            params_dtype=torch.bfloat16,
            quant_config=_mixed_precision_config(),
            prefix="model.embed_tokens",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton path needs a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [64, 2080])
def test_nvfp4_embedding_triton_matches_torch(dist_init, dtype, hidden):
    torch.manual_seed(0)
    vocab = 300
    layer = VocabParallelEmbedding(
        vocab,
        hidden,
        params_dtype=dtype,
        quant_config=_mixed_precision_config(),
        prefix="model.embed_tokens",
    )
    weight = torch.randint(0, 256, (vocab, hidden // 2), dtype=torch.uint8)
    # Every finite non-negative e4m3 code, subnormals included.
    scale = torch.randint(0, 0x7F, (vocab, hidden // 16), dtype=torch.uint8)
    scale = scale.view(torch.float8_e4m3fn)
    for name, value in (
        ("weight", weight),
        ("weight_scale", scale),
        ("weight_scale_2", torch.tensor(0.03)),
    ):
        layer.weight_loader(getattr(layer, name), value)
    with set_default_torch_dtype(dtype):
        layer.quant_method.process_weights_after_loading(layer)

    ids = torch.randint(0, vocab, (3, 41))
    expected = layer(ids)  # torch path on CPU
    layer.cuda()
    torch.testing.assert_close(layer(ids.cuda()).cpu(), expected, rtol=0, atol=0)
    assert layer(ids[:0].cuda()).shape == (0, 41, hidden)
