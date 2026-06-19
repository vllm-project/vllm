# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dummy OOT platform that piggybacks on CUDA.

The class derives from CUDA's ``NvmlCudaPlatform`` so all device handling,
attention-backend selection, etc. continue to work as on a stock NVIDIA
host. The only behavioral change is ``_enum``: by reporting ``OOT`` we
trigger the ``hw_agnostic/`` dispatch in ``vllm.models.deepseek_v4``.
"""

from typing import TYPE_CHECKING

from vllm.platforms.cuda import NvmlCudaPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class DSv4OOTPlatform(NvmlCudaPlatform):
    """A test-only OOT platform that piggybacks on CUDA.

    A real OOT vendor would supply its own attention backend, memory
    allocator, kernel registrations, etc. We don't have any of that here;
    we just want a Platform whose ``is_out_of_tree()`` returns True so the
    DeepSeek V4 model selector picks the ``hw_agnostic/`` branch, while the
    rest of vLLM's CUDA infrastructure keeps working underneath.

    To do that we keep ``_enum = CUDA`` (so the kernel choosers, memory
    allocator, sleep mode, etc. all behave as on CUDA) and override only
    ``is_out_of_tree()``.
    """

    # Keep ``_enum`` as CUDA so kernel registries / allocator dispatch
    # continue to work. ``device_name`` is also inherited (= "cuda") so
    # that ``torch.device(f"{device_name}:{rank}")`` resolves.

    def is_out_of_tree(self) -> bool:
        return True

    @classmethod
    def support_deep_gemm(cls) -> bool:
        # The CUDA base class would return True on Hopper/Blackwell here,
        # but DeepGEMM's JIT compiler cannot produce kernels for the OOT
        # device (we're running on a foreign accelerator that piggybacks
        # on CUDA's enum). Disabling at the platform level skips:
        #   * the kernel-warmup pass (kernel_warmup.py:62 gate);
        #   * `is_deep_gemm_supported()` checks throughout vLLM,
        #     including the FlashInfer/DeepGEMM linear-kernel
        #     registry's runtime decisions.
        # Symptom when this is enabled on OOT: DeepGEMM JIT
        # ``runtime != nullptr`` assertion at compiler.hpp:147.
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)

        # Force the Triton linear-kernel backend. The CUDA priority list
        # (vllm/model_executor/kernels/linear/__init__.py:_POSSIBLE_FP8_BLOCK_KERNELS)
        # picks FlashInfer/DeepGEMM/Cutlass first, all of which JIT-compile
        # CUDA-binary-coded kernels that the OOT host (which inherits
        # _enum=CUDA) cannot run — surfaces as e.g. DeepGEMM JIT
        # ``runtime != nullptr`` or ``cudaErrorInvalidAddressSpace``. The
        # Triton kernel is portable and serves as the agnostic fallback.
        if vllm_config.kernel_config.linear_backend == "auto":
            vllm_config.kernel_config.linear_backend = "triton"

        # Force the Triton MoE backend. The Fp8 MoE oracle
        # (vllm/model_executor/layers/fused_moe/oracle/fp8.py) probes
        # FlashInfer/DeepGEMM/AITER/vLLM-Cutlass first; on the OOT host
        # none of them resolves a valid kernel and the oracle would
        # raise ``NotImplementedError: No FP8 MoE backend supports the
        # deployment configuration.``. ``moe_backend="triton"`` skips
        # the auto-priority list and goes straight to ``TritonExperts``
        # (BatchedTritonExperts when use_batched_activation_format=True),
        # which is portable.
        if vllm_config.kernel_config.moe_backend == "auto":
            vllm_config.kernel_config.moe_backend = "triton"

