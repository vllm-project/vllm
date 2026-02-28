# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.utils.torch_utils import resolve_kv_cache_dtype_string


def test_resolve_kv_cache_dtype_auto_prefers_model_arch_quant_cfg():
    # Simulate ModelOpt quantization config where kv_cache_quant_algo is "fp8"
    quant_cfg = {"quant_method": "modelopt", "kv_cache_quant_algo": "fp8"}

    model_config = SimpleNamespace(
        model_arch_config=SimpleNamespace(quantization_config=quant_cfg),
        hf_config=SimpleNamespace(quantization_config=None),
    )

    assert resolve_kv_cache_dtype_string("auto", model_config) == "fp8_e4m3"


def test_resolve_kv_cache_dtype_auto_fallbacks_to_hf_quant_cfg():
    quant_cfg = {"quant_method": "modelopt", "kv_cache_quant_algo": "fp8"}

    model_config = SimpleNamespace(
        model_arch_config=SimpleNamespace(quantization_config=None),
        hf_config=SimpleNamespace(quantization_config=quant_cfg),
    )

    assert resolve_kv_cache_dtype_string("auto", model_config) == "fp8_e4m3"


def test_resolve_kv_cache_dtype_explicit_override_wins():
    quant_cfg = {"quant_method": "modelopt", "kv_cache_quant_algo": "fp8"}

    model_config = SimpleNamespace(
        model_arch_config=SimpleNamespace(quantization_config=quant_cfg),
        hf_config=SimpleNamespace(quantization_config=None),
    )

    assert resolve_kv_cache_dtype_string("bfloat16", model_config) == "bfloat16"
