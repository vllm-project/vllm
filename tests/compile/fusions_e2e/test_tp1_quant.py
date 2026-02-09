# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import pytest

from vllm.config import PassConfig

from .common import (
    INDUCTOR_GRAPH_PARTITION,
    AttentionBackendCase,
    Matches,
    custom_ops_combos,
    is_blackwell,
)
from .models import (
    FLASHINFER_ATTN,
    TRITON_ATTN,
    llama3_8b_fp4,
    llama3_8b_fp8,
    llama4_scout_fp4,
    llama4_scout_fp8,
    qwen3_a3b_fp8,
)


@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides, use_deepgemm",
    [
        (*llama3_8b_fp8, False),
        (*llama4_scout_fp8, False),
        (*qwen3_a3b_fp8, False),
        (*qwen3_a3b_fp8, True),
    ],
)
@pytest.mark.parametrize("attn_backend", [TRITON_ATTN, FLASHINFER_ATTN])
@pytest.mark.parametrize("n_layers", [6])
@pytest.mark.parametrize("custom_ops", custom_ops_combos("quant_fp8", "rms_norm"))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
def test_tp1_fp8_fusions(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    custom_ops: str,
    inductor_graph_partition: bool,
    use_deepgemm: bool,
    run_e2e_fusion_test,
    monkeypatch,
):
    if use_deepgemm:
        # TODO(luka/eliza) DeepGEMM uses different quants, matching not supported
        #  - on Blackwell, uses a special quant fp8, currently not supported
        #  - on Hopper, tma-aligned scales inhibit matching (fix WIP)
        pytest.skip("DeepGEMM & quant matching not currently supported")

    matches = matches_fn(n_layers)

    if "qwen" in model_name.lower() and "-quant_fp8" in custom_ops:
        # This is why config forces +quant_fp8 by default
        pytest.skip("native QuantFP8 matching not supported for group quant")

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_attn_quant=True,
            enable_qk_norm_rope_fusion=True,
        ),
    )

    matches_check = [
        "rms_quant_fusion",
        "act_quant_fusion",
        "norm_rope_fusion",
        "attn_quant_fusion",
    ]

    run_e2e_fusion_test(
        model_name,
        matches,
        model_kwargs,
        attn_backend,
        compilation_config,
        matches_check,
        use_deepgemm=use_deepgemm,
    )


@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides",
    [llama3_8b_fp4, llama4_scout_fp4],
)
@pytest.mark.parametrize("attn_backend", [FLASHINFER_ATTN])
@pytest.mark.parametrize("n_layers", [6])
@pytest.mark.parametrize("custom_ops", custom_ops_combos("rms_norm"))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.skipif(not is_blackwell(), reason="Blackwell required for fp4")
def test_tp1_fp4_fusions(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    custom_ops: str,
    inductor_graph_partition: bool,
    run_e2e_fusion_test,
):
    matches = matches_fn(n_layers)

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_attn_quant=True,
            enable_qk_norm_rope_fusion=True,
        ),
    )

    matches_check = ["act_quant_fusion", "attn_quant_fusion", "norm_rope_fusion"]

    run_e2e_fusion_test(
        model_name,
        matches,
        model_kwargs,
        attn_backend,
        compilation_config,
        matches_check,
    )
