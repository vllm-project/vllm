# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform wrapper for the meta-device CI/CD mode (``VLLM_META_MODE=1``).

Rather than sprinkling ``if envs.VLLM_META_MODE`` branches through the concrete
platform classes (``CudaPlatform`` etc.), this mixin overrides only the handful
of device-capability methods that must report a GPU-free, zero-memory device.
It is mixed OVER the detected platform class in ``vllm/platforms/__init__.py``
only when meta mode is on (see ``wrap_meta_platform``), so the concrete platform
classes stay untouched and off-mode has zero impact (this module is not even
imported).
"""

import torch


class MetaPlatformMixin:
    """Meta-mode overrides, mixed in front of the detected platform class.

    Because the mixin is only installed when ``VLLM_META_MODE`` is on, the
    overrides are unconditional -- the presence of the wrapper IS the gate.
    """

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # Pinned host memory needs a CUDA context.
        return False

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        # No real device to query.
        return 1

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        # Weights + KV live on the meta device (zero real memory) and there is
        # no device to poll.
        return 0.0

    @classmethod
    def current_device(cls, *args, **kwargs) -> torch.device:
        # Layers place small buffers via current_platform.current_device();
        # route them to the meta device.
        return torch.device("meta")

    @classmethod
    def opaque_attention_op(cls) -> bool:
        # Route attention through the vllm::unified_attention_with_output custom
        # op (use_direct_call=False) so its registered fake (meta) kernel is
        # used instead of the real backend kernel.
        return True

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config):
        # The vllm_c (_C) IR impls have no meta kernel; force pure-torch native.
        from vllm.config.kernel import IrOpPriorityConfig

        return IrOpPriorityConfig.with_default(["native"])


def wrap_meta_platform(platform_cls: type) -> type:
    """Return a subclass mixing ``MetaPlatformMixin`` over ``platform_cls``."""
    return type(
        f"Meta{platform_cls.__name__}",
        (MetaPlatformMixin, platform_cls),
        {},
    )
