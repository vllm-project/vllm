# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ZenCpuPlatform(CpuPlatform):
    """CPU platform with AMD Zen (ZenDNN/zentorch) optimizations.

    Model-load time (dispatch_cpu_unquantized_gemm in layers/utils.py):
      - Routes linear ops to zentorch_linear_unary.
      - When VLLM_ZENTORCH_WEIGHT_PREPACK=1 (default), eagerly prepacks
        weights via zentorch_weight_prepack_for_linear.
    """

    device_name: str = "cpu"
    device_type: str = "cpu"

    def is_zen_cpu(self) -> bool:
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)
        cls._apply_pytorch_backports()

    @classmethod
    def _apply_pytorch_backports(cls):
        """Backport PyTorch mainline fixes missing in 2.10.

        PyTorch 2.10 has a bug in FxGraphCachePickler.dumps that doesn't
        catch ValueError, causing torch.compile cache misses. Remove this
        once we drop PyTorch 2.10 support. PT mainline already has this fix.
        """
        from torch.torch_version import TorchVersion

        if (TorchVersion(torch.__version__) < (2, 10)
                or TorchVersion(torch.__version__) >= (2, 11)):
            return

        cls._patch_fxgraphcache_pickle()

    @classmethod
    def _patch_fxgraphcache_pickle(cls):
        """Backport mainline ValueError fix to FxGraphCachePickler.dumps()."""
        from torch._inductor.codecache import BypassFxGraphCache, FxGraphCachePickler

        original_dumps = FxGraphCachePickler.dumps
        if hasattr(original_dumps, "_zen_patched"):
            return

        def patched_dumps(self, obj):
            try:
                return original_dumps(self, obj)
            except ValueError as e:
                raise BypassFxGraphCache("Failed to pickle cache key") from e

        patched_dumps._zen_patched = True  # type: ignore[attr-defined]
        FxGraphCachePickler.dumps = patched_dumps
        logger.info("[zen_cpu] Patched FxGraphCachePickler.dumps (ValueError fix)")
