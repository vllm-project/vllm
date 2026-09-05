# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm.distributed import get_tp_group
from vllm.triton_utils import tl, triton


@triton.jit
def _reduce_published_shared_kernel(
    workspace_ptr,
    flags_ptr,
    output_ptr,
    max_m: tl.constexpr,
    hidden_size: tl.constexpr,
    num_sources: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * block_n + tl.arange(0, block_n)
    mask = columns < hidden_size
    generation = tl.load(flags_ptr)
    generation_offset = generation * max_m * num_sources * hidden_size
    row_offset = row * num_sources * hidden_size
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    for source in range(num_sources):
        source_offset = source * hidden_size
        values = tl.load(
            workspace_ptr + generation_offset + row_offset + source_offset + columns,
            mask=mask,
            other=0.0,
        )
        accumulator += values.to(tl.float32)
    tl.store(output_ptr + row * hidden_size + columns, accumulator, mask=mask)


def reduce_published_shared(
    workspace: torch.Tensor,
    flags: torch.Tensor,
    output: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Reduce token-destination TP partials into a local BF16 residual."""
    if workspace.ndim != 4 or not workspace.is_contiguous():
        raise ValueError("workspace must be contiguous [generation, M, source, H]")
    if workspace.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError("workspace and output must be BF16")
    if workspace.device != output.device or flags.device != output.device:
        raise ValueError("workspace, flags, and output must share one device")
    if flags.shape != (12,) or flags.dtype != torch.int32:
        raise ValueError("flags must be int32 [12]")
    if not 1 <= num_tokens <= workspace.shape[1]:
        raise ValueError("num_tokens exceeds the published workspace")
    hidden_size = workspace.shape[3]
    if output.shape != (workspace.shape[1], hidden_size) or not output.is_contiguous():
        raise ValueError("output must be contiguous [max_M, H]")

    block_n = 256
    _reduce_published_shared_kernel[(num_tokens, triton.cdiv(hidden_size, block_n))](
        workspace,
        flags,
        output,
        workspace.shape[1],
        hidden_size,
        workspace.shape[2],
        block_n,
    )
    return output[:num_tokens]


@dataclass(frozen=True)
class KimiK3SPPublishedTailContract:
    tp_group_id: int
    tp_size: int
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    latent_size: int
    max_num_tokens: int


class KimiK3SPPublishedTailOp:
    """Own the SP token-destination workspace and local beta-add tail."""

    _instances: ClassVar[
        dict[KimiK3SPPublishedTailContract, "KimiK3SPPublishedTailOp"]
    ] = {}

    @classmethod
    def initialize(
        cls,
        *,
        hidden_size: int,
        latent_size: int,
        max_num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "KimiK3SPPublishedTailOp":
        tp = get_tp_group()
        group = tp.device_group
        device = torch.device(device)
        contract = KimiK3SPPublishedTailContract(
            tp_group_id=id(group),
            tp_size=dist.get_world_size(group),
            device=device,
            dtype=dtype,
            hidden_size=hidden_size,
            latent_size=latent_size,
            max_num_tokens=max_num_tokens,
        )
        op = cls._instances.get(contract)
        if op is None:
            op = cls(contract, group)
            cls._instances[contract] = op
        return op

    def __init__(
        self,
        contract: KimiK3SPPublishedTailContract,
        group: dist.ProcessGroup,
    ) -> None:
        if contract.tp_size not in (8, 16):
            raise ValueError("K3 SP published tail requires TP 8 or 16")
        if contract.device.type != "cuda":
            raise ValueError("K3 SP published tail requires CUDA")
        if torch.cuda.get_device_capability(contract.device)[0] != 10:
            raise ValueError("K3 SP published tail requires SM100")
        if contract.dtype != torch.bfloat16:
            raise ValueError("K3 SP published tail requires BF16")
        if (contract.hidden_size, contract.latent_size) != (7168, 3584):
            raise ValueError("K3 SP published tail requires H=7168 and K=3584")
        if contract.max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be positive")

        self.contract = contract
        device = contract.device
        with torch.accelerator.device_index(device.index):
            # One generation is sufficient: producer and consumer are ordered
            # on the same stream, and MegaMoE's final NVLink barrier completes
            # all remote writes before this consumer starts.
            self._workspace = symm_mem.empty(
                (
                    1,
                    contract.max_num_tokens,
                    contract.tp_size,
                    contract.hidden_size,
                ),
                dtype=contract.dtype,
                device=device,
            )
            self._symm_mem = symm_mem.rendezvous(self._workspace, group)
            self._flags = torch.zeros(12, dtype=torch.int32, device=device)
            self._flags[1] = 1
            self._flags[2] = self._workspace[0].nbytes
            self._output = torch.empty(
                (contract.max_num_tokens, contract.hidden_size),
                dtype=contract.dtype,
                device=device,
            )
            peer_ptrs = [
                self._symm_mem.get_buffer(
                    peer,
                    self._workspace.shape,
                    contract.dtype,
                ).data_ptr()
                for peer in range(contract.tp_size)
            ]
            if any(pointer == 0 for pointer in peer_ptrs):
                raise RuntimeError("K3 SP shared LSA peer mapping is unavailable")
            self._peer_ptrs = torch.tensor(peer_ptrs, dtype=torch.int64, device=device)

        torch.accelerator.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])

    def published_workspace(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._workspace, self._flags, self._peer_ptrs

    def __call__(
        self,
        routed_output: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        contract = self.contract
        if routed_output.ndim != 2:
            raise ValueError("routed_output must be rank-2")
        num_tokens = routed_output.shape[0]
        expected = (
            (routed_output, (num_tokens, contract.latent_size), "routed_output"),
            (
                up_weight,
                (contract.hidden_size, contract.latent_size),
                "up_weight",
            ),
        )
        for tensor, shape, name in expected:
            if (
                tensor.shape != shape
                or tensor.dtype != contract.dtype
                or tensor.device != contract.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA {contract.dtype} {list(shape)}"
                )
        shared_output = reduce_published_shared(
            self._workspace,
            self._flags,
            self._output,
            num_tokens,
        )
        return shared_output.addmm_(routed_output, up_weight.t())
