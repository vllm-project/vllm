# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SDSA's settings and torch.compile rules, on CPU."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationConfig, CompilationMode
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.nemotron_shortcut.nvidia.sdsa import (
    SdsaAttention,
    SdsaConfig,
    get_sdsa_config,
)
from vllm.models.nemotron_shortcut.nvidia.sdsa.attention import (
    SDSA_INDEXER_OP,
    _split_at_indexer,
)

from ..configuration_nemotron4 import Nemotron4TextConfig as TextConfig

# Stand-ins for the indexer's key projection: BF16, quantized or not.
UNQUANTIZED = SimpleNamespace(
    quant_method=UnquantizedLinearMethod(), params_dtype=torch.bfloat16
)
QUANTIZED = SimpleNamespace(quant_method=object(), params_dtype=torch.bfloat16)


def test_sdsa_config_comes_from_the_model_config() -> None:
    """K is index_topk and the sdsa_* fields set the rest, "auto" when the
    config does not have them, as checkpoints' configs do not. An "auto" k cache
    dtype is NVFP4 if the checkpoint quantizes the indexer's key projection,
    BF16 if not; a set one wins."""
    config = TextConfig(index_topk=512, sdsa_prefill_core="token")
    expected = SdsaConfig(top_k=512, indexer_dtype="bf16", prefill_core="token")
    assert get_sdsa_config(config, UNQUANTIZED) == expected
    config = TextConfig(index_topk=2048)
    assert get_sdsa_config(config, QUANTIZED).indexer_dtype == "nvfp4"
    config = TextConfig(index_topk=2048, sdsa_indexer_dtype="bf16")
    assert get_sdsa_config(config, QUANTIZED).indexer_dtype == "bf16"


def test_sdsa_config_refuses_what_sdsa_does_not_have() -> None:
    """A K other than 512 or 2048, a core that SDSA does not have, and a BF16 k
    cache for a model that is not BF16."""
    with pytest.raises(ValueError, match="index_topk"):
        get_sdsa_config(TextConfig(index_topk=1024), UNQUANTIZED)
    config = TextConfig(index_topk=512, sdsa_prefill_core="fa5")
    with pytest.raises(ValueError, match="prefill_core"):
        get_sdsa_config(config, UNQUANTIZED)
    float16 = SimpleNamespace(
        quant_method=UnquantizedLinearMethod(), params_dtype=torch.float16
    )
    with pytest.raises(ValueError, match="BF16 model"):
        get_sdsa_config(TextConfig(index_topk=512), float16)


def compilation(mode, splitting_ops, use_inductor_graph_partition=False):
    """A stand-in vLLM config with the compilation fields the split rule reads."""
    fields = dict(mode=mode, splitting_ops=splitting_ops)
    fields["use_inductor_graph_partition"] = use_inductor_graph_partition
    return SimpleNamespace(compilation_config=SimpleNamespace(**fields))


def test_torch_compile_splits_at_the_indexer() -> None:
    """VLLM's default splitting ops have the indexer op, and an eager config
    needs none. A torch.compile config that cannot split at it is refused: no
    splitting ops, a list without the op, Inductor graph partition, and stock
    torch.compile."""
    defaults = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    defaults.set_splitting_ops_for_v1(all2all_backend="naive")
    assert SDSA_INDEXER_OP in defaults.splitting_ops
    _split_at_indexer(SimpleNamespace(compilation_config=defaults))
    _split_at_indexer(compilation(CompilationMode.NONE, None))
    for config in [
        compilation(CompilationMode.VLLM_COMPILE, []),
        compilation(CompilationMode.VLLM_COMPILE, ["vllm::unified_attention"]),
        compilation(CompilationMode.VLLM_COMPILE, [SDSA_INDEXER_OP], True),
        compilation(CompilationMode.STOCK_TORCH_COMPILE, None),
    ]:
        with pytest.raises(NotImplementedError, match="splitting op"):
            _split_at_indexer(config)


def test_kv_cache_scales_load_into_the_core() -> None:
    """The checkpoint's KV cache scales load into the layer's core. Without a
    query scale, an FP8 cache takes the key scale for Q; a query scale that
    loads later replaces it."""
    core = torch.nn.Module()
    core.kv_cache_dtype = "fp8"
    for name in ("q_scale", "k_scale", "v_scale"):
        unset = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        setattr(core, name, unset)
    layer = SimpleNamespace(core=core)
    scales = [("k_scale", torch.tensor(0.025)), ("v_scale", torch.tensor(0.035))]
    loaded = SdsaAttention.load_weights(layer, scales)
    assert loaded == {"core.k_scale", "core.v_scale"}
    assert core.q_scale.item() == pytest.approx(0.025)
    SdsaAttention.load_weights(layer, [("q_scale", torch.tensor(0.017))])
    assert core.q_scale.item() == pytest.approx(0.017)
