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

