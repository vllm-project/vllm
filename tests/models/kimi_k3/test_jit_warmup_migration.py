# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import torch

from vllm.models.kimi_k3.common.mtp import _FUSED_MTP_INPUT_KERNEL
from vllm.models.kimi_k3.nvidia.kda import _STORE_CACHE_CHECKPOINTS_KERNEL
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    _ALIGNED_STATE_INDICES_KERNEL,
    _STAGE_SPEC_DECODE_KERNEL,
)
from vllm.models.kimi_k3.nvidia.ops.attn_res import _ATTN_RES_KERNEL
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    _COMMIT_KDA_STATE_KERNEL,
    _COMPACT_CONV_STATE_KERNEL,
    _PREPARE_COMMIT_PLAN_KERNEL,
    _RECOVERSSM_VERIFY_KERNEL,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda.chunk import (
    _CHUNK_GLA_FWD_O_KERNEL,
    _GATE_CHUNK_CUMSUM_KERNEL,
    _RECOMPUTE_WU_KERNEL,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda.chunk_intra import (
    _CHUNK_KDA_INTER_SOLVE_KERNEL,
    _CHUNK_KDA_SUB_CHUNK_KERNEL,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda.chunk_intra_token_parallel import (  # noqa: E501
    _CHUNK_KDA_TOKEN_PARALLEL_KERNEL,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
    _FUSED_KDA_GATE_BETA_KERNEL,
    _FUSED_RECURRENT_KDA_FWD_KERNEL,
    _FUSED_RECURRENT_KDA_PACKED_DECODE_KERNEL,
)
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    _CHUNK_GATED_DELTA_RULE_FWD_H_KERNEL,
)
from vllm.third_party.flash_linear_attention.ops.kda import (
    _LAYER_NORM_GATED_FWD_KERNEL,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import (
    _L2NORM_FWD_KERNEL2,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

_NUM_HEADS = 16
_HEAD_DIM = 128
_PROJECTION_SIZE = _NUM_HEADS * _HEAD_DIM


def _vllm_config():
    text_config = SimpleNamespace(
        attn_res_block_size=8,
        hidden_size=7168,
        num_hidden_layers=64,
        rms_norm_eps=1e-5,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=text_config,
        )
    )


def test_attn_res_warmup_covers_runtime_dispatch_classes(monkeypatch):
    # ops/__init__ shadows the attn_res submodule with a re-exported function,
    # so a dotted-string monkeypatch target can't resolve current_platform;
    # import the real module and patch the platform instance directly.
    attn_res_module = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.attn_res")
    monkeypatch.setattr(
        attn_res_module.current_platform, "is_arch_support_pdl", lambda: False
    )
    vllm_config = _vllm_config()
    config = vllm_config.model_config.hf_text_config
    block_size = config.attn_res_block_size
    max_blocks = triton.cdiv(config.num_hidden_layers, block_size)
    warmed_keys = set(_ATTN_RES_KERNEL.get_warmup_keys(vllm_config))

    for num_tokens in (1, 256):
        for layer_idx in range(config.num_hidden_layers):
            is_block_write = layer_idx % block_size == 0
            previous_blocks = triton.cdiv(layer_idx, block_size)
            pre_key = _ATTN_RES_KERNEL.dispatch(
                dtype=vllm_config.model_config.dtype,
                num_tokens=num_tokens,
                num_blocks=previous_blocks,
                hidden_size=config.hidden_size,
                max_blocks=max_blocks,
                block_write_idx=layer_idx // block_size if is_block_write else -1,
                eps=config.rms_norm_eps,
                output_norm_eps=config.rms_norm_eps,
                has_delta=layer_idx > 0,
                apply_output_norm=True,
                launch_pdl=False,
            )
            assert pre_key in warmed_keys

            post_key = _ATTN_RES_KERNEL.dispatch(
                dtype=vllm_config.model_config.dtype,
                num_tokens=num_tokens,
                num_blocks=previous_blocks + is_block_write,
                hidden_size=config.hidden_size,
                max_blocks=max_blocks,
                block_write_idx=-1,
                eps=config.rms_norm_eps,
                output_norm_eps=config.rms_norm_eps,
                has_delta=not is_block_write,
                apply_output_norm=True,
                launch_pdl=False,
            )
            assert post_key in warmed_keys

        for has_delta in (False, True):
            final_key = _ATTN_RES_KERNEL.dispatch(
                dtype=vllm_config.model_config.dtype,
                num_tokens=num_tokens,
                num_blocks=max_blocks,
                hidden_size=config.hidden_size,
                max_blocks=max_blocks,
                block_write_idx=-1,
                eps=config.rms_norm_eps,
                output_norm_eps=0.0,
                has_delta=has_delta,
                apply_output_norm=False,
                launch_pdl=False,
            )
            assert final_key in warmed_keys


def test_fused_mtp_input_warmup_matches_runtime_dispatch():
    vllm_config = _vllm_config()
    config = vllm_config.model_config.hf_text_config
    warmed_keys = _FUSED_MTP_INPUT_KERNEL.get_warmup_keys(vllm_config)
    runtime_key = _FUSED_MTP_INPUT_KERNEL.dispatch(
        positions_dtype=torch.int64,
        dtype=vllm_config.model_config.dtype,
        inputs_embeds_stride=config.hidden_size,
        previous_hidden_states_stride=config.hidden_size,
        output_stride=2 * config.hidden_size,
        hidden_size=config.hidden_size,
    )

    assert warmed_keys == [runtime_key]


def test_kda_metadata_warmup_matches_runtime_dispatch(monkeypatch):
    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.kda_metadata._metadata_launch_pdl",
        lambda: False,
    )

    aligned_registration = dict(
        max_num_blocks_per_req=17,
        num_state_slots=4,
        cache_block_size=64,
    )
    assert _ALIGNED_STATE_INDICES_KERNEL.get_warmup_keys(
        **aligned_registration
    ) == [
        _ALIGNED_STATE_INDICES_KERNEL.dispatch(
            block_table_stride_0=17,
            block_table_stride_1=1,
            seq_lens_stride=1,
            state_indices_stride_0=4,
            state_indices_stride_1=1,
            cache_block_size=64,
            num_state_slots=4,
            launch_pdl=False,
        )
    ]

    stage_registration = dict(
        source_state_indices_stride_0=17,
        spec_state_slots=4,
    )
    assert _STAGE_SPEC_DECODE_KERNEL.get_warmup_keys(**stage_registration) == [
        _STAGE_SPEC_DECODE_KERNEL.dispatch(
            state_indices_stride_0=17,
            state_indices_stride_1=1,
            staged_state_indices_stride_0=4,
            staged_state_indices_stride_1=1,
            num_state_slots=4,
            null_state_id=NULL_BLOCK_ID,
            launch_pdl=False,
        )
    ]


def test_store_checkpoints_warmup_covers_runtime_strides():
    common = dict(
        x_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
        recurrent_state_dtype=torch.float32,
        state_stride_0=4096,
        state_stride_1=3,
        state_stride_2=1,
        checkpoint_stride_0=_NUM_HEADS * _HEAD_DIM * _HEAD_DIM,
        recurrent_state_stride_0=_NUM_HEADS * _HEAD_DIM * _HEAD_DIM + 64,
        state_len=3,
        width=3 * _PROJECTION_SIZE,
        recurrent_row_size=_NUM_HEADS * _HEAD_DIM * _HEAD_DIM,
    )
    runtime_strides = (
        4 * _PROJECTION_SIZE + _HEAD_DIM + _NUM_HEADS,
        3 * _PROJECTION_SIZE,
    )
    warmed = set(
        _STORE_CACHE_CHECKPOINTS_KERNEL.get_warmup_keys(
            x_stride_0=runtime_strides,
            **common,
        )
    )
    expected = {
        _STORE_CACHE_CHECKPOINTS_KERNEL.dispatch(
            x_stride_0=x_stride,
            x_stride_1=1,
            checkpoint_offset_stride=1,
            **common,
        )
        for x_stride in runtime_strides
    }
    assert warmed == expected


def _recurrent_registration() -> dict:
    return dict(
        io_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        a_log_dtype=torch.float32,
        dt_bias_dtype=torch.float32,
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        stride_state_token=_NUM_HEADS * _HEAD_DIM * _HEAD_DIM + 64,
        use_lower_bound=True,
        launch_pdl=False,
    )


def test_chunk_gla_fwd_o_autotune_warmup_matches_runtime_dispatch():
    registration = {
        "q_dtype": torch.bfloat16,
        "v_dtype": torch.bfloat16,
        "g_dtype": torch.float32,
        "h_dtype": torch.bfloat16,
        "out_dtype": torch.bfloat16,
        "a_dtype": torch.bfloat16,
        "num_heads": _NUM_HEADS,
        "qk_head_dim": _HEAD_DIM,
        "v_head_dim": _HEAD_DIM,
        "block_t": 64,
        "is_varlen": True,
    }

    assert _CHUNK_GLA_FWD_O_KERNEL.get_warmup_keys(**registration) == [
        _CHUNK_GLA_FWD_O_KERNEL.dispatch(**registration)
    ]


def test_chunk_gla_fwd_o_runtime_launch_is_independent_from_dispatch(monkeypatch):
    owner = type(_CHUNK_GLA_FWD_O_KERNEL)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    q = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    v = torch.empty_like(q)
    g = torch.empty_like(q)
    A = torch.empty(1, 64, _NUM_HEADS, 64)
    h = torch.empty(1, 1, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM)
    o = torch.empty_like(q)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs, output = call(
        owner,
        q,
        v,
        g,
        A,
        h,
        o,
        _HEAD_DIM**-0.5,
        chunk_size=64,
    )

    assert grid({"BV": 128}) == (1, 1, _NUM_HEADS)
    assert launch_kwargs == {
        "T": 64,
        "H": _NUM_HEADS,
        "K": _HEAD_DIM,
        "V": _HEAD_DIM,
        "BT": 64,
    }
    assert output is o


def test_fused_recurrent_warmup_covers_runtime_dispatch():
    common = dict(
        **_recurrent_registration(),
        scale=_HEAD_DIM**-0.5,
        stride_qkv_token=3 * _PROJECTION_SIZE,
        stride_g_token=_PROJECTION_SIZE,
        stride_out_token=_PROJECTION_SIZE,
        stride_indices_seq=4,
        has_dt_bias=True,
    )
    warmed = set(_FUSED_RECURRENT_KDA_FWD_KERNEL.get_warmup_keys(**common))
    runtime_common = {
        key: value
        for key, value in common.items()
        if key not in {
            "a_log_dtype",
            "dt_bias_dtype",
            "has_dt_bias",
            "use_lower_bound",
        }
    }
    runtime_variants = (
        dict(
            gate_dtype=torch.bfloat16,
            beta_dtype=torch.bfloat16,
            a_log_dtype=torch.float32,
            dt_bias_dtype=torch.float32,
            use_gate_in_kernel=True,
            apply_beta_sigmoid=True,
            has_a_log=True,
            has_dt_bias=True,
            use_lower_bound=True,
        ),
        dict(
            gate_dtype=torch.float32,
            beta_dtype=torch.float32,
            a_log_dtype=torch.float32,
            dt_bias_dtype=torch.float32,
            use_gate_in_kernel=False,
            apply_beta_sigmoid=False,
            has_a_log=False,
            has_dt_bias=False,
            use_lower_bound=False,
        ),
    )

    for n_sequences in range(1, 24):
        for variant in runtime_variants:
            runtime_key = _FUSED_RECURRENT_KDA_FWD_KERNEL.dispatch(
                n_sequences=n_sequences,
                is_spec_decoding=True,
                use_qk_l2norm=True,
                **variant,
                **runtime_common,
            )
            assert runtime_key in warmed


def test_packed_decode_warmup_matches_runtime_dispatch():
    registration = dict(
        **_recurrent_registration(),
        stride_mixed_token=3 * _PROJECTION_SIZE,
        stride_g_token=_PROJECTION_SIZE,
    )
    expected = _FUSED_RECURRENT_KDA_PACKED_DECODE_KERNEL.dispatch(
        io_dtype=registration["io_dtype"],
        state_dtype=registration["state_dtype"],
        a_log_dtype=registration["a_log_dtype"],
        dt_bias_dtype=registration["dt_bias_dtype"],
        num_heads=_NUM_HEADS,
        k_dim=_HEAD_DIM,
        v_dim=_HEAD_DIM,
        scale=_HEAD_DIM**-0.5,
        stride_mixed_token=registration["stride_mixed_token"],
        stride_g_token=registration["stride_g_token"],
        stride_state_token=registration["stride_state_token"],
        use_lower_bound=True,
        launch_pdl=False,
    )
    assert _FUSED_RECURRENT_KDA_PACKED_DECODE_KERNEL.get_warmup_keys(
        **registration
    ) == [expected]


def test_gate_beta_warmup_covers_triton_scalar_classes():
    beta_strides = (
        _NUM_HEADS,
        4 * _PROJECTION_SIZE + _HEAD_DIM + _NUM_HEADS,
    )
    registration = dict(
        io_dtype=torch.bfloat16,
        a_log_dtype=torch.float32,
        dt_bias_dtype=torch.float32,
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        max_num_tokens=64,
        stride_g_token=_PROJECTION_SIZE,
        stride_beta_token=beta_strides,
        has_dt_bias=True,
        use_lower_bound=True,
        launch_pdl=False,
    )
    warmed = set(_FUSED_KDA_GATE_BETA_KERNEL.get_warmup_keys(**registration))
    dispatch_inputs = {
        key: value
        for key, value in registration.items()
        if key not in {"max_num_tokens", "stride_beta_token"}
    }

    for num_tokens in range(1, registration["max_num_tokens"] + 1):
        for beta_stride in beta_strides:
            assert _FUSED_KDA_GATE_BETA_KERNEL.dispatch(
                num_tokens=num_tokens,
                stride_beta_token=beta_stride,
                **dispatch_inputs,
            ) in warmed


def test_recoverssm_verify_warmup_covers_beta_stride_classes():
    beta_strides = (
        _NUM_HEADS,
        4 * _PROJECTION_SIZE + _HEAD_DIM + _NUM_HEADS,
    )
    registration = dict(
        io_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        a_log_dtype=torch.float32,
        dt_bias_dtype=torch.float32,
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        spec_query_len=4,
        stride_q_token=3 * _PROJECTION_SIZE,
        stride_k_token=3 * _PROJECTION_SIZE,
        stride_v_token=3 * _PROJECTION_SIZE,
        stride_g_token=_PROJECTION_SIZE,
        stride_beta_token=beta_strides,
        stride_out_token=_PROJECTION_SIZE,
        stride_state_block=_NUM_HEADS * _HEAD_DIM * _HEAD_DIM + 64,
        stride_correction_block=_NUM_HEADS * 4 * _HEAD_DIM + 32,
        stride_kg_block=_NUM_HEADS * 4 * 2 * _HEAD_DIM + 16,
        stride_state_indices=(1, 17),
        use_lower_bound=True,
    )
    warmed = set(_RECOVERSSM_VERIFY_KERNEL.get_warmup_keys(**registration))
    dispatch_inputs = {
        key: value
        for key, value in registration.items()
        if key not in {"head_dim", "stride_beta_token", "stride_state_indices"}
    }

    for beta_stride in beta_strides:
        for state_indices_stride in registration["stride_state_indices"]:
            assert _RECOVERSSM_VERIFY_KERNEL.dispatch(
                key_dim=_HEAD_DIM,
                value_dim=_HEAD_DIM,
                stride_beta_token=beta_stride,
                stride_state_indices=state_indices_stride,
                **dispatch_inputs,
            ) in warmed


def test_recoverssm_commit_warmup_matches_runtime_dispatch():
    plan_registration = dict(
        spec_query_len=4,
        align_mode=True,
        mamba_block_size=64,
        block_table_width=17,
        stride_state_indices=(1, 17),
    )
    warmed_plans = set(
        _PREPARE_COMMIT_PLAN_KERNEL.get_warmup_keys(**plan_registration)
    )
    plan_dispatch_inputs = {
        key: value
        for key, value in plan_registration.items()
        if key != "stride_state_indices"
    }
    assert warmed_plans == {
        _PREPARE_COMMIT_PLAN_KERNEL.dispatch(
            has_request_indices=has_request_indices,
            stride_state_indices=state_indices_stride,
            **plan_dispatch_inputs,
        )
        for has_request_indices in (False, True)
        for state_indices_stride in plan_registration["stride_state_indices"]
    }

    compact_registration = dict(
        conv_state_dtype=torch.bfloat16,
        conv_dim=3 * _PROJECTION_SIZE,
        conv_history_len=3,
        align_mode=True,
        stride_state_indices=(1, 17),
    )
    compact_dispatch_inputs = {
        key: value
        for key, value in compact_registration.items()
        if key != "stride_state_indices"
    }
    assert set(
        _COMPACT_CONV_STATE_KERNEL.get_warmup_keys(**compact_registration)
    ) == {
        _COMPACT_CONV_STATE_KERNEL.dispatch(
            stride_state_indices=state_indices_stride,
            **compact_dispatch_inputs,
        )
        for state_indices_stride in compact_registration["stride_state_indices"]
    }

    commit_registration = dict(
        state_dtype=torch.float32,
        kg_dtype=torch.bfloat16,
        a_log_dtype=torch.float32,
        dt_bias_dtype=torch.float32,
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        spec_query_len=4,
        use_lower_bound=True,
        align_mode=True,
        stride_state_indices=(1, 17),
    )
    runtime_registration = {
        key: value
        for key, value in commit_registration.items()
        if key not in {"head_dim", "stride_state_indices"}
    }
    expected = {
        _COMMIT_KDA_STATE_KERNEL.dispatch(
            key_dim=_HEAD_DIM,
            value_dim=_HEAD_DIM,
            stride_state_indices=state_indices_stride,
            **runtime_registration,
        )
        for state_indices_stride in commit_registration["stride_state_indices"]
    }
    assert set(
        _COMMIT_KDA_STATE_KERNEL.get_warmup_keys(**commit_registration)
    ) == expected


def test_recompute_wu_autotune_warmup_matches_runtime_dispatch():
    registration = {
        "k_dtype": torch.bfloat16,
        "kg_dtype": torch.bfloat16,
        "v_dtype": torch.bfloat16,
        "beta_dtype": torch.float32,
        "w_dtype": torch.bfloat16,
        "u_dtype": torch.bfloat16,
        "a_dtype": torch.bfloat16,
        "gk_dtype": torch.float32,
        "num_heads": _NUM_HEADS,
        "qk_head_dim": _HEAD_DIM,
        "v_head_dim": _HEAD_DIM,
        "block_t": 64,
        "block_k": 64,
        "block_v": 64,
        "store_qg": False,
        "store_kg": True,
        "is_varlen": True,
        "dot_precision": "ieee",
    }

    assert _RECOMPUTE_WU_KERNEL.get_warmup_keys(**registration) == [
        _RECOMPUTE_WU_KERNEL.dispatch(**registration)
    ]


def test_recompute_wu_runtime_launch_is_independent_from_dispatch(monkeypatch):
    owner = type(_RECOMPUTE_WU_KERNEL)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    k = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    kg = torch.empty_like(k)
    v = torch.empty_like(k)
    beta = torch.empty(1, 64, _NUM_HEADS)
    w = torch.empty_like(k)
    u = torch.empty_like(k)
    A = torch.empty(1, 64, _NUM_HEADS, 64)
    gk = torch.empty_like(k)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs = call(
        owner,
        None,
        k,
        None,
        kg,
        v,
        beta,
        w,
        u,
        A,
        gk,
    )

    assert grid == (1, _NUM_HEADS)
    assert launch_kwargs == {
        "T": 64,
        "H": _NUM_HEADS,
        "K": _HEAD_DIM,
        "V": _HEAD_DIM,
        "BT": 64,
        "BK": 64,
        "BV": 64,
        "DOT_PRECISION": "ieee",
    }


def test_gate_chunk_cumsum_autotune_warmup_matches_runtime_dispatch():
    registration = {
        "s_dtype": torch.bfloat16,
        "raw_beta_dtype": torch.bfloat16,
        "a_log_dtype": torch.float32,
        "g_bias_dtype": torch.float32,
        "o_dtype": torch.float32,
        "beta_out_dtype": torch.float32,
        "num_heads": _NUM_HEADS,
        "gate_dim": _HEAD_DIM,
        "block_t": 64,
        "has_bias": True,
        "is_varlen": True,
        "use_lower_bound": True,
    }

    assert _GATE_CHUNK_CUMSUM_KERNEL.get_warmup_keys(**registration) == [
        _GATE_CHUNK_CUMSUM_KERNEL.dispatch(**registration)
    ]


def test_gate_chunk_cumsum_runtime_launch_is_independent_from_dispatch(
    monkeypatch,
):
    owner = type(_GATE_CHUNK_CUMSUM_KERNEL)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    s = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    raw_beta = torch.empty(1, 64, _NUM_HEADS)
    A_log = torch.empty(_NUM_HEADS)
    g_bias = torch.empty(_NUM_HEADS * _HEAD_DIM)
    o = torch.empty_like(s)
    beta_out = torch.empty(1, 64, _NUM_HEADS)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs = call(
        owner,
        s,
        raw_beta,
        A_log,
        g_bias,
        o,
        beta_out,
        use_lower_bound=True,
    )

    assert grid({"BS": 64}) == (3, 1, _NUM_HEADS)
    assert launch_kwargs == {
        "T": 64,
        "stride_beta_batch": 64 * _NUM_HEADS,
        "stride_beta_token": _NUM_HEADS,
        "stride_beta_head": 1,
        "H": _NUM_HEADS,
        "S": _HEAD_DIM,
        "BT": 64,
        "USE_LOWER_BOUND": True,
    }


def test_chunk_kda_inter_solve_autotune_warmup_matches_runtime_dispatch():
    registration = {
        "q_dtype": torch.bfloat16,
        "k_dtype": torch.bfloat16,
        "g_dtype": torch.float32,
        "beta_dtype": torch.float32,
        "aqk_dtype": torch.bfloat16,
        "akkd_dtype": torch.float32,
        "akk_dtype": torch.bfloat16,
        "num_heads": _NUM_HEADS,
        "hv": _NUM_HEADS,
        "qk_head_dim": _HEAD_DIM,
        "block_t": 64,
        "block_c": 16,
        "is_varlen": True,
        "use_safe_gate": True,
        "solve_tril_dot_precision": "tf32",
    }

    assert _CHUNK_KDA_INTER_SOLVE_KERNEL.get_warmup_keys(**registration) == [
        _CHUNK_KDA_INTER_SOLVE_KERNEL.dispatch(**registration)
    ]


def test_chunk_kda_inter_solve_runtime_launch_is_independent_from_dispatch(
    monkeypatch,
):
    owner = type(_CHUNK_KDA_INTER_SOLVE_KERNEL)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    k = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    q = torch.empty_like(k)
    g = torch.empty_like(k)
    beta = torch.empty(1, 64, _NUM_HEADS)
    Aqk = torch.empty(1, 64, _NUM_HEADS, 64)
    Akkd = torch.empty(1, 64, _NUM_HEADS, 16)
    Akk = torch.empty(1, 64, _NUM_HEADS, 64)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs = call(
        owner,
        q,
        k,
        g,
        beta,
        Aqk,
        Akkd,
        Akk,
        1.0,
        use_safe_gate=True,
        solve_tril_dot_precision="tf32",
    )

    assert grid == (1, _NUM_HEADS)
    assert launch_kwargs == {
        "T": 64,
        "H": _NUM_HEADS,
        "HV": _NUM_HEADS,
        "K": _HEAD_DIM,
        "BT": 64,
        "BC": 16,
        "USE_SAFE_GATE": True,
        "SOLVE_TRIL_DOT_PRECISION": "tf32",
    }


def test_chunk_kda_sub_chunk_autotune_warmup_matches_runtime_dispatch():
    registration = {
        "q_dtype": torch.bfloat16,
        "k_dtype": torch.bfloat16,
        "g_dtype": torch.float32,
        "beta_dtype": torch.float32,
        "aqk_dtype": torch.bfloat16,
        "akk_dtype": torch.float32,
        "num_heads": _NUM_HEADS,
        "hv": _NUM_HEADS,
        "qk_head_dim": _HEAD_DIM,
        "block_t": 64,
        "block_c": 16,
        "block_k": 128,
        "is_varlen": True,
        "use_gather": True,
    }

    assert _CHUNK_KDA_SUB_CHUNK_KERNEL.get_warmup_keys(**registration) == [
        _CHUNK_KDA_SUB_CHUNK_KERNEL.dispatch(**registration)
    ]


def test_chunk_kda_sub_chunk_runtime_launch_is_independent_from_dispatch(
    monkeypatch,
):
    owner = type(_CHUNK_KDA_SUB_CHUNK_KERNEL)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    k = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    q = torch.empty_like(k)
    g = torch.empty_like(k)
    beta = torch.empty(1, 64, _NUM_HEADS)
    Aqk = torch.empty(1, 64, _NUM_HEADS, 64)
    Akk = torch.empty(1, 64, _NUM_HEADS, 16)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs = call(
        owner,
        q,
        k,
        g,
        beta,
        Aqk,
        Akk,
        1.0,
        use_gather=True,
    )

    assert grid == (1, 4, _NUM_HEADS)
    assert launch_kwargs == {
        "T": 64,
        "H": _NUM_HEADS,
        "HV": _NUM_HEADS,
        "K": _HEAD_DIM,
        "BT": 64,
        "BC": 16,
        "BK": 128,
        "USE_GATHER": True,
    }


def test_chunk_kda_token_parallel_autotune_warmup_matches_runtime_dispatch():
    registration = {
        "q_dtype": torch.bfloat16,
        "k_dtype": torch.bfloat16,
        "g_dtype": torch.float32,
        "beta_dtype": torch.float32,
        "aqk_dtype": torch.bfloat16,
        "akk_dtype": torch.float32,
        "num_heads": _NUM_HEADS,
        "hv": _NUM_HEADS,
        "qk_head_dim": _HEAD_DIM,
        "block_t": 64,
        "block_c": 16,
        "is_varlen": True,
    }

    assert _CHUNK_KDA_TOKEN_PARALLEL_KERNEL.get_warmup_keys(**registration) == [
        _CHUNK_KDA_TOKEN_PARALLEL_KERNEL.dispatch(**registration)
    ]


def test_chunk_kda_token_parallel_runtime_launch_is_independent_from_dispatch(
    monkeypatch,
):
    owner = type(_CHUNK_KDA_TOKEN_PARALLEL_KERNEL)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    q = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    k = torch.empty_like(q)
    g = torch.empty_like(q)
    beta = torch.empty(1, 64, _NUM_HEADS)
    Aqk = torch.empty(1, 64, _NUM_HEADS, 64)
    Akk = torch.empty(1, 64, _NUM_HEADS, 16)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs = call(
        owner,
        q,
        k,
        g,
        beta,
        Aqk,
        Akk,
        1.0,
    )

    assert grid({"BH": 8}) == (64, 2)
    assert launch_kwargs == {
        "N": 1,
        "T": 64,
        "H": _NUM_HEADS,
        "HV": _NUM_HEADS,
        "K": _HEAD_DIM,
        "BT": 64,
        "BC": 16,
    }


def test_l2norm_warmup_covers_triton_scalar_classes():
    registration = dict(
        x_dtype=torch.bfloat16,
        y_dtype=torch.bfloat16,
        eps=1e-6,
        n=_HEAD_DIM,
        bd=128,
        mblock=32,
    )
    warmed = set(_L2NORM_FWD_KERNEL2.get_warmup_keys(**registration))
    expected = {
        _L2NORM_FWD_KERNEL2.dispatch(m=m, **registration) for m in (1, 2, 16)
    }
    assert warmed == expected


def test_l2norm_runtime_launch_is_independent_from_dispatch(monkeypatch):
    owner = type(_L2NORM_FWD_KERNEL2)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    x = torch.empty(64, _HEAD_DIM)
    y = torch.empty_like(x)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs = call(owner, x, y, 1e-6, 64, _HEAD_DIM, 128, 32)

    assert grid == (triton.cdiv(64, 32),)
    assert launch_kwargs["X"] is x
    assert launch_kwargs["Y"] is y
    assert {k: v for k, v in launch_kwargs.items() if k not in ("X", "Y")} == {
        "eps": 1e-6,
        "M": 64,
        "N": _HEAD_DIM,
        "BD": 128,
        "MBLOCK": 32,
    }


def test_delta_h_autotune_warmup_matches_runtime_dispatch():
    registration = {
        "k_dtype": torch.bfloat16,
        "v_dtype": torch.bfloat16,
        "w_dtype": torch.bfloat16,
        "gk_dtype": torch.float32,
        "h0_dtype": torch.float32,
        "ht_dtype": torch.float32,
        "num_heads": _NUM_HEADS,
        "num_k_heads": _NUM_HEADS,
        "qk_head_dim": _HEAD_DIM,
        "v_head_dim": _HEAD_DIM,
        "block_t": 64,
        "use_g": False,
        "use_gk": True,
        "use_initial_state": True,
        "store_final_state": True,
        "save_new_value": True,
        "is_varlen": True,
        "use_exp2": True,
    }

    keys = _CHUNK_GATED_DELTA_RULE_FWD_H_KERNEL.get_warmup_keys(**registration)
    assert keys == [_CHUNK_GATED_DELTA_RULE_FWD_H_KERNEL.dispatch(**registration)]


def test_delta_h_runtime_launch_is_independent_from_dispatch(monkeypatch):
    owner = type(_CHUNK_GATED_DELTA_RULE_FWD_H_KERNEL)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    k = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    v = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    w = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    v_new = torch.empty_like(v)
    gk = torch.empty(1, 64, _NUM_HEADS, _HEAD_DIM)
    h = torch.empty(1, 1, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM)
    h0 = torch.empty(1, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM)
    ht = torch.empty(1, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM)
    cu_seqlens = torch.tensor([0, 64], dtype=torch.int32)
    chunk_offsets = torch.zeros(1, dtype=torch.int32)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs = call(
        owner,
        k,
        v,
        w,
        v_new,
        None,
        gk,
        h,
        h0,
        ht,
        cu_seqlens,
        chunk_offsets,
        chunk_size=64,
        use_exp2=True,
    )

    assert grid({"BV": 64}) == (triton.cdiv(_HEAD_DIM, 64), _NUM_HEADS)
    assert launch_kwargs == {
        "T": 64,
        "H": _NUM_HEADS,
        "Hg": _NUM_HEADS,
        "K": _HEAD_DIM,
        "V": _HEAD_DIM,
        "BT": 64,
        "USE_EXP2": True,
    }


def test_o_norm_warmup_covers_triton_scalar_classes():
    registration = dict(
        x_dtype=torch.bfloat16,
        y_dtype=torch.bfloat16,
        g_dtype=torch.bfloat16,
        w_dtype=torch.bfloat16,
        eps=1e-5,
        num_heads=_NUM_HEADS,
        g_stride_n=_NUM_HEADS * _HEAD_DIM,
        d=_HEAD_DIM,
        block_t=16,
        block_d=128,
        activation="sigmoid",
        is_rms_norm=True,
        store_residual_out=False,
        has_residual=False,
        has_weight=True,
        has_bias=False,
    )
    warmed = set(_LAYER_NORM_GATED_FWD_KERNEL.get_warmup_keys(**registration))
    expected = {
        _LAYER_NORM_GATED_FWD_KERNEL.dispatch(t=t, **registration)
        for t in (1, 2, 16)
    }
    assert warmed == expected


def test_o_norm_runtime_launch_is_independent_from_dispatch(monkeypatch):
    owner = type(_LAYER_NORM_GATED_FWD_KERNEL)()

    def fail_dispatch(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(owner, "dispatch", fail_dispatch)
    x = torch.empty(32, _HEAD_DIM)
    g = torch.empty(32, _NUM_HEADS, _HEAD_DIM)
    y = torch.empty_like(x)
    weight = torch.empty(_HEAD_DIM)
    rstd = torch.empty(32)

    call = type(owner).__call__.__wrapped__
    grid, launch_kwargs = call(
        owner,
        x,
        g,
        y,
        weight,
        None,
        None,
        None,
        None,
        rstd,
        1e-5,
        32,
        _NUM_HEADS,
        _NUM_HEADS * _HEAD_DIM,
        _HEAD_DIM,
        128,
        16,
        "sigmoid",
        True,
    )

    assert grid == (triton.cdiv(32, 16),)
    assert launch_kwargs["x"] is x
    assert launch_kwargs["g"] is g
    assert launch_kwargs["y"] is y
    assert launch_kwargs["w"] is weight
    assert launch_kwargs["rstd"] is rstd
    assert {
        k: v
        for k, v in launch_kwargs.items()
        if k not in ("x", "g", "y", "w", "rstd")
    } == {
        "b": None,
        "residual": None,
        "residual_out": None,
        "mean": None,
        "eps": 1e-5,
        "T": 32,
        "H": _NUM_HEADS,
        "g_stride_n": _NUM_HEADS * _HEAD_DIM,
        "D": _HEAD_DIM,
        "BD": 128,
        "BT": 16,
        "ACTIVATION": "sigmoid",
        "IS_RMS_NORM": True,
        "num_warps": 8,
    }
