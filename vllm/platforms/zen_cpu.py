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

        if TorchVersion(torch.__version__) < (2, 10):
            return

        cls._patch_fxgraphcache_pickle()

    @classmethod
    def _patch_fxgraphcache_pickle(cls):
        """Backport mainline ValueError fix to FxGraphCachePickler.dumps()."""
        import pickle

        from torch._inductor.codecache import FxGraphCachePickler

        original_dumps = FxGraphCachePickler.dumps
        if hasattr(original_dumps, "_zen_patched"):
            return

        def patched_dumps_method(self, obj):
            from torch._inductor.codecache import BypassFxGraphCache
            import logging as _logging
            _logger = _logging.getLogger("torch._inductor.codecache")
            try:
                self.dump(obj)
                return self._stream.getvalue()
            except (TypeError, AttributeError, pickle.PicklingError, ValueError) as e:
                _logger.warning("Failed to pickle cache key", exc_info=True)
                raise BypassFxGraphCache("Failed to pickle cache key") from e
            finally:
                self._stream.seek(0)
                self._stream.truncate(0)

        patched_dumps_method._zen_patched = True
        FxGraphCachePickler.dumps = patched_dumps_method
        logger.info("[zen_cpu] Patched FxGraphCachePickler.dumps (ValueError fix)")
