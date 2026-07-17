# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    ops.qr_max_size()
    quick_ar = True
except Exception:
    # For CPUs and CUDA
    quick_ar = False


from vllm.distributed.utils import is_weak_contiguous  # noqa: E402, F401


class QuickReduceRegime(Enum):
    # Keep integer ids aligned with csrc/quickreduce/quick_reduce.h
    FP = 0
    INT8 = 1
    INT6 = 2
    INT4 = 3
    INT3 = 4
    NONE = 5


KB = 1024
MB = 1024 * KB


class QuickAllReduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    # The following data is based on kernel tests.
    # In this order [FP, INT8, INT6, INT4, INT3].
    _QR_MIN_SIZE = {
        (torch.float16, 2): [1 * MB, 2 * MB, 2 * MB, 1 * MB, 1 * MB],
        (torch.float16, 4): [1 * MB, 16 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.float16, 8): [16 * MB, 4 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.bfloat16, 2): [2 * MB, 8 * MB, 8 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 4): [8 * MB, 64 * MB, 64 * MB, 16 * MB, 16 * MB],
        (torch.bfloat16, 8): [
            16 * MB,
            2048 * MB,
            2048 * MB,
            2048 * MB,
            2048 * MB,
        ],
    }

    def __init__(self, group: ProcessGroup, device: int | str | torch.device) -> None:
        """
        Custom allreduce provides non-destructive acceleration and is
        available for CUDA and ROCm MI300 series.

        Custom quick allreduce leverages quantization for further
        acceleration on ROCm. It currently supports Q8, Q6, Q4, and Q3
        quantization formats and FP(float16, bfloat16). Q3 (INT3) is
        restricted to TP2 (world_size == 2) due to poor performance on
        larger world sizes.

        Quick allreduce is designed as a complement to custom allreduce.
        Its initialization requires even stricter conditions.

        Only the ROCm MI300 series is supported for quick allreduce at
        this time.

        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self.disabled = True
        if not self._rocm_arch_available():
            logger.debug(
                "Custom quick allreduce is only supported on ROCm MI300 series."
            )
            return

        if not quick_ar:
            # disable because of missing quick reduce library
            # e.g. in a cuda environment
            logger.info(
                "Custom quick allreduce is disabled because "
                "of missing custom quick allreduce library"
            )
            return

        self.group = group
        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "Custom quick allreduce should be attached to a non-NCCL group."
        )
        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom quick allreduce for
            # multi-node case.
            logger.warning(
                "Custom quick allreduce is disabled because this "
                "process group spans across nodes."
            )
            return
        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        self.rank = rank
        self.world_size = world_size
        if world_size == 1:
            # No need to initialize QuickReduce for single GPU case.
            return

        if world_size not in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom quick allreduce is disabled due to an "
                "unsupported world size: %d. Supported world sizes: %s.",
                world_size,
                str(QuickAllReduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        # device.index is a visible ordinal, not a logical local ID.
        physical_device_id = current_platform.visible_device_id_to_physical_device_id(
            device.index
        )
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(self.world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom quick allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        self.fully_connected = current_platform.is_fully_connected(physical_device_ids)
        if self.world_size > 2 and not self.fully_connected:
            logger.debug(
                "Custom quick allreduce is disabled because it's not supported "
                "on more than two PCIe-only GPUs. "
            )
            return

        self.init_quick_all_reduce()

    def init_quick_all_reduce(self):
        # On RocM, bfloat16 kernels are slower than fp16
        # due to slower match operations
        # If environment variable is set to 1, we convert input to fp16
        self.use_fp16_kernels = envs.VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16
        regime_str = envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION
        if regime_str not in QuickReduceRegime.__members__:
            logger.warning(
                "Custom quick allreduce:",
                f"Invalid quantization level: {regime_str}. "
                "Supported levels: "
                f"{list(QuickReduceRegime.__members__.keys())}",
            )
            return

        if regime_str == "NONE":
            logger.debug(
                "Custom quick allreduce is disabled based "
                "on env variable "
                "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION='NONE'"
            )
            return
        self.qr_quant_level = QuickReduceRegime[regime_str]

        # INT3 is only enabled for TP2 (world_size == 2).
        # Kernel benchmarks show INT3 all-reduce on TP4/TP8 has poor
        # performance (the extra ranks make the 3-bit codec's pack/unpack
        # overhead outweigh the reduced communication volume), so INT3 is
        # restricted to 2-GPU tensor parallelism. For TP4/TP8 use a wider
        # codec (e.g. INT4) or NONE instead.
        if self.qr_quant_level == QuickReduceRegime.INT3 and self.world_size != 2:
            logger.warning(
                "Custom quick allreduce is disabled: INT3 quantization is "
                "only supported for TP2 (world_size == 2), but world_size "
                "is %d. INT3 on TP4/TP8 is disabled due to poor kernel "
                "performance. Use INT4/NONE for this world size.",
                self.world_size,
            )
            return

        self.qr_quantization_min_size = self._get_qr_quantization_min_size()
        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and hasattr(vllm_config, "model_config")
            and hasattr(vllm_config.model_config, "dtype")
        ):
            dtype = vllm_config.model_config.dtype
            if dtype not in [torch.float16, torch.bfloat16]:
                logger.debug(
                    "Custom quick allreduce disabled: only supports "
                    "float16 and float16, but get %s.",
                    dtype,
                )
                return

            if dtype == torch.bfloat16 and self.use_fp16_kernels:
                logger.info(
                    "Custom quick allreduce: BF16 inputs will be converted "
                    "to FP16 to improve performance. set "
                    "envs.VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0 "
                    "to turn off."
                )

        # VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB is specified in MB
        qr_max_size = envs.VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB
        if qr_max_size is not None:
            if qr_max_size < 1:
                logger.info(
                    "You should not set a max_size smaller than 1MB, which can "
                    "lead to error or degradation to custom allreduce or rccl."
                )
            qr_max_size = qr_max_size * MB
        effective_qr_max_size = (
            qr_max_size if qr_max_size is not None else ops.qr_max_size()
        )
        qr_min_size = self._get_qr_min_size(effective_qr_max_size)
        self._ptr = ops.init_custom_qr(self.rank, self.world_size, qr_max_size)
        self.qr_max_size = effective_qr_max_size
        self.qr_min_size = qr_min_size
        if qr_min_size is not None:
            logger.info(
                "Custom quick allreduce: min size override = %d MB",
                qr_min_size // MB,
            )
        if self.qr_quantization_min_size is not None:
            logger.info(
                "Custom quick allreduce: quantization codec threshold = %d KB",
                self.qr_quantization_min_size // KB,
            )
        self.create_shared_buffer()
        self.disabled = False

    @staticmethod
    def _get_qr_min_size(qr_max_size: int | None) -> int | None:
        qr_min_size = envs.VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB
        if qr_min_size is None:
            return None
        if qr_min_size < 0:
            raise ValueError(
                "VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB must be non-negative, "
                f"got {qr_min_size}"
            )
        qr_min_size *= MB
        if qr_max_size is not None and qr_min_size > qr_max_size:
            raise ValueError(
                "VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB must be less than or "
                "equal to the effective QuickReduce max size"
            )
        return qr_min_size

    @staticmethod
    def _get_qr_quantization_min_size() -> int | None:
        quantization_min_size = envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB
        if quantization_min_size is None:
            return None
        if quantization_min_size < 0:
            raise ValueError(
                "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB must be "
                f"non-negative, got {quantization_min_size}"
            )
        return quantization_min_size * KB

    def _rocm_arch_available(self):
        if not current_platform.is_rocm():
            return False
        try:
            props = torch.cuda.get_device_properties(0)
            gcn_arch = getattr(props, "gcnArchName", "")
            supported_archs = ["gfx94", "gfx95"]
            return any(gfx in gcn_arch for gfx in supported_archs)
        except Exception as e:
            logger.warning("Failed to determine ROCm for quick allreduce: %s", e)
            return False

    def create_shared_buffer(self):
        """
        Creates a shared buffer for quickreduce.
        Has to be called after init_custom_qr
        """
        handle = ops.qr_get_handle(self._ptr)
        world_size = dist.get_world_size(group=self.group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=self.group)
        ops.qr_open_handles(self._ptr, handles)

    def should_quick_allreduce(self, inp: torch.Tensor):
        """
        Check if quickreduce is available
        """
        if self.disabled:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom quick allreduce requires input byte size to be
        # multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        dtype = inp.dtype
        if self.use_fp16_kernels:
            dtype = torch.float16
        min_size = self.qr_min_size
        if min_size is None:
            min_size = self._QR_MIN_SIZE[(dtype, self.world_size)][
                self.qr_quant_level.value
            ]
        return inp_size <= self.qr_max_size and inp_size >= min_size

    def quick_all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """Performs an out-of-place custom quick all reduce."""
        # quick allreduce doesn't require a separate graph mode,
        # as QR uses static IPC buffer.
        if out is None:
            out = torch.empty_like(inp)
        ops.qr_all_reduce(
            self._ptr, inp, out, self._get_qr_quant_level(inp), self.use_fp16_kernels
        )
        return out

    def _get_qr_quant_level(self, inp: torch.Tensor) -> int:
        quantization_min_size = self.qr_quantization_min_size
        if (
            quantization_min_size is not None
            and inp.numel() * inp.element_size() < quantization_min_size
        ):
            return QuickReduceRegime.FP.value
        return self.qr_quant_level.value

    def close(self):
        if not self.disabled and getattr(self, "_ptr", None):
            if ops is not None:
                ops.qr_destroy(self._ptr)
            self._ptr = 0
            self.disabled = True

    def __del__(self):
        self.close()
