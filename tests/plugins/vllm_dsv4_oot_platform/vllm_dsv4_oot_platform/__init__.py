# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dummy out-of-tree platform plugin for validating the DeepSeek V4
``hw_agnostic/`` model branch.

The plugin reuses CUDA's runtime (so the model can actually execute on a
real GPU) but reports ``PlatformEnum.OOT``. This causes
``vllm.models.deepseek_v4.__init__`` to dispatch to the hardware-agnostic
implementation, exercising exactly the path an OOT vendor would take.
"""


def dsv4_oot_platform_plugin() -> str | None:
    # Activate unconditionally: simply having this package installed in
    # the environment is the opt-in. ``pip uninstall vllm_dsv4_oot_platform``
    # disables it. Out-of-tree plugins take precedence over built-in
    # platforms (see ``vllm.platforms.resolve_current_platform_cls_qualname``).

    # Breakable is a CUDA-only feature; Force the env var to ``"0"``.
    import os

    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

    return "vllm_dsv4_oot_platform.platform.DSv4OOTPlatform"
