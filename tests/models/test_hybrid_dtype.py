# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: SIM117
import pytest
import torch

from tests.models.language.pooling.mteb_utils import mteb_test_embed_models
from tests.models.utils import DTypeInfo, EmbedModelInfo
from vllm.config import _STR_DTYPE_TO_TORCH_DTYPE

high_precision_data_types = [
    DTypeInfo(dtype="auto"),
    DTypeInfo(dtype="float32"),
    DTypeInfo(dtype="float32", attn_dtype="float16"),
    DTypeInfo(dtype="float32", attn_dtype="bfloat16")
]
low_precision_data_types = [
    DTypeInfo(dtype="float16"),
    DTypeInfo(dtype="bfloat16")
]
data_types = high_precision_data_types + low_precision_data_types
embed_model = "intfloat/e5-small"
generate_model = "EleutherAI/pythia-70m"


@pytest.mark.parametrize("dtype", data_types)
def test_dtype(vllm_runner, dtype: DTypeInfo):
    with vllm_runner(embed_model,
                     dtype=dtype.dtype,
                     max_model_len=None,
                     attn_dtype=dtype.attn_dtype) as vllm_model:
        model_config = vllm_model.model.llm_engine.model_config
        if dtype.dtype == "auto":  # hybrid dtype
            assert model_config.dtype == torch.float32
            assert model_config.attn_dtype == torch.float16
        elif dtype.attn_dtype == "auto":
            assert model_config.dtype == model_config.attn_dtype
        else:
            assert model_config.dtype == _STR_DTYPE_TO_TORCH_DTYPE[dtype.dtype]
            assert model_config.attn_dtype == _STR_DTYPE_TO_TORCH_DTYPE[
                dtype.attn_dtype]


@pytest.mark.parametrize("dtype", data_types)
def test_embed_models_mteb(hf_runner, vllm_runner, dtype: DTypeInfo):
    model_info = EmbedModelInfo(embed_model,
                                architecture="BertModel",
                                dtype=dtype)
    st_main_score, st_dtype = None, None
    if model_info.dtype in high_precision_data_types:
        _, st_main_score, _, st_dtype = mteb_test_embed_models(
            hf_runner,
            vllm_runner,
            model_info,
            st_main_score=st_main_score,
            st_dtype=st_dtype)
    else:
        with pytest.raises(AssertionError):
            _, st_main_score, _, st_dtype = mteb_test_embed_models(
                hf_runner,
                vllm_runner,
                model_info,
                st_main_score=st_main_score,
                st_dtype=st_dtype)


@pytest.mark.parametrize("model", [generate_model])
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [4])
def test_generate_models(hf_runner, vllm_runner, example_prompts, model: str,
                         dtype: DTypeInfo, max_tokens: int,
                         num_logprobs: int) -> None:
    if dtype.attn_dtype == "auto":
        with vllm_runner(model, dtype=dtype.dtype,
                         attn_dtype=dtype.attn_dtype):
            pass
    else:
        with pytest.raises(ValueError):
            with vllm_runner(model,
                             dtype=dtype.dtype,
                             attn_dtype=dtype.attn_dtype):
                pass
