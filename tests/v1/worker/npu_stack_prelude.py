# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import-order shims for running this tree's tests on the NPU stack.

The NPU validation stack pairs torch 2.10 + torch_npu + the vllm-ascend
plugin (vllm 0.31-era API) with this tree's newer vllm core. Two gaps
must be bridged before any VllmConfig is constructed in that process:

1. vllm_ascend.mrv2_utils reads envs.VLLM_USE_V2_MODEL_RUNNER at import
   time; the field was removed from this tree's envs. Stub it to False
   (Model Runner V1) so the plugin import succeeds.
2. vllm_ascend.platform.apply_config_platform_defaults reads
   PassConfig.enable_sp / fuse_gemm_comms / sp_min_token_num (removed
   upstream by vLLM #60517, absent here). The executor-level tests need
   no compilation-config defaulting, so neutralize the hook.

This module exists only for the NPU device-validation harness; the CPU
CI never imports it.
"""

import contextlib

import vllm.envs as _envs

if not hasattr(_envs, "VLLM_USE_V2_MODEL_RUNNER"):
    _envs.VLLM_USE_V2_MODEL_RUNNER = False

with contextlib.suppress(Exception):  # plugin absent on CPU hosts
    import vllm_ascend.mrv2_utils  # noqa: F401  # reads the stubbed env

try:
    from vllm.platforms import current_platform

    current_platform.__class__.apply_config_platform_defaults = lambda self, cfg: None
    # init_ascend_config's derive_and_validate also reads the 0.31-era
    # PassConfig.enable_sp (vllm_ascend/ascend_config.py:949);
    # neutralize the AscendConfig derivation too.
    current_platform.__class__.check_and_update_config = lambda self, cfg: None

    # The Ascend attention backends import 0.31-era kv_cache_interface
    # symbols (CrossAttentionSpec, FullAttentionSpec) absent in this
    # tree. These tests bind (K, V) caches themselves and drive CPU
    # attention metadata, so resolve the CPU backend instead.
    def _cpu_attn_backend_cls(
        self, backend, attn_selector_config=None, num_heads=None, **kw
    ):
        return "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"

    current_platform.__class__.get_attn_backend_cls = _cpu_attn_backend_cls

    # vllm_ascend's AscendCompiler implements the 0.31-era compute_hash
    # API; this tree's CompilerInterface rejects it at class-definition
    # time. These tests run eager, so use the default compile backend.
    current_platform.__class__.get_compile_backend = lambda cls=None: (
        current_platform.__class__.simple_compile_backend
    )

    # set_additional_forward_context reads the 0.31-era
    # vllm_config.use_v2_model_runner field; the harness supplies its
    # own forward context, so make the hook a no-op context manager.
    import contextlib as _contextlib

    current_platform.__class__.set_additional_forward_context = classmethod(
        lambda cls, *a, **kw: _contextlib.nullcontext()
    )
except Exception:  # pragma: no cover - defensive; never expected
    pass
