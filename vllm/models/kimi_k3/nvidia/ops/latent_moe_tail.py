# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)

_MAX_NUM_TOKENS = 128
_SKINNY_MAX_NUM_TOKENS = 5
_MMA_TILER_MN = (64, 32)
_GEMM_CLUSTER_MN = (1, 8)
_B_PRIME_STAGES = 2
_COLLECTIVE_TOKEN_CTAS = 32
_LAMPORT_COPY_CTAS = 32
_LAMPORT_COPY_THREADS = 224
_SUPPORTED_TP_SIZES = (8, 16)


@dataclass(frozen=True)
class KimiK3LatentMoETailContract:
    tp_group_id: int
    tp_size: int
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    latent_size: int
    max_num_tokens: int
    rms_eps: float
    experts_per_token: int


class KimiK3LatentMoETailOp:
    """Process-wide cached K3 latent-MoE tail implementation."""

    _instances: ClassVar[
        dict[KimiK3LatentMoETailContract, "KimiK3LatentMoETailOp"]
    ] = {}

    @classmethod
    def _contract_and_group(
        cls,
        *,
        hidden_size: int,
        latent_size: int,
        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
        experts_per_token: int = 0,
    ) -> tuple[KimiK3LatentMoETailContract, dist.ProcessGroup]:
        tp = get_tp_group()
        group = tp.device_group
        device = torch.device(device)
        tp_device = torch.device(tp.device)
        if device != tp_device:
            raise ValueError(
                f"Input device {device} does not match TP device {tp_device}."
            )
        return (
            KimiK3LatentMoETailContract(
                tp_group_id=id(group),
                tp_size=dist.get_world_size(group),
                device=device,
                dtype=dtype,
                hidden_size=hidden_size,
                latent_size=latent_size,
                max_num_tokens=_MAX_NUM_TOKENS,
                rms_eps=float(rms_eps),
                experts_per_token=experts_per_token,
            ),
            group,
        )

    @classmethod
    def initialize(
        cls,
        *,
        hidden_size: int,
        latent_size: int,
        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
        experts_per_token: int = 0,
    ) -> "KimiK3LatentMoETailOp":
        contract, group = cls._contract_and_group(
            hidden_size=hidden_size,
            latent_size=latent_size,
            experts_per_token=experts_per_token,
            dtype=dtype,
            device=device,
            rms_eps=rms_eps,
        )
        op = cls._instances.get(contract)
        if op is None:
            op = cls(contract, group)
            cls._instances[contract] = op
        return op

    def __init__(
        self,
        contract: KimiK3LatentMoETailContract,
        group: dist.ProcessGroup,
    ) -> None:
        if contract.tp_size not in _SUPPORTED_TP_SIZES:
            raise ValueError(
                "K3 latent-MoE tail fusion requires TP in "
                f"{_SUPPORTED_TP_SIZES}, got {contract.tp_size}."
            )
        if contract.device.type != "cuda":
            raise ValueError("K3 latent-MoE tail fusion requires a CUDA device.")
        if torch.cuda.get_device_capability(contract.device)[0] != 10:
            raise ValueError("K3 latent-MoE tail fusion requires SM100.")
        if contract.dtype != torch.bfloat16:
            raise ValueError("K3 latent-MoE tail fusion requires bfloat16.")
        if (contract.hidden_size, contract.latent_size) != (7168, 3584):
            raise ValueError(
                "K3 latent-MoE tail fusion requires hidden_size=7168 and "
                "latent_size=3584."
            )

        self.contract = contract
        self.rank = dist.get_rank(group)
        from .cute_dsl.latent_moe_tail import (
            AdaptiveUpProjectionKernel,
            CollectiveKernel,
            LamportCopyKernel,
        )

        with torch.accelerator.device_index(contract.device.index):
            self._collective = CollectiveKernel(
                group=group,
                rank=self.rank,
                tp_size=contract.tp_size,
                latent_dim=contract.latent_size,
                hidden_dim=contract.hidden_size,
                max_m=contract.max_num_tokens,
                max_token_ctas=_COLLECTIVE_TOKEN_CTAS,
                rms_eps=contract.rms_eps,
                fp32_internal=False,
                top_k=contract.experts_per_token,
            )
            self._up_projection = AdaptiveUpProjectionKernel(
                group=group,
                rank=self.rank,
                tp_size=contract.tp_size,
                latent_dim=contract.latent_size,
                hidden_dim=contract.hidden_size,
                max_m=contract.max_num_tokens,
                skinny_max_m=_SKINNY_MAX_NUM_TOKENS,
                mma_tiler_mn=_MMA_TILER_MN,
                cluster_shape_mn=_GEMM_CLUSTER_MN,
                b_prime_stages=_B_PRIME_STAGES,
            )
            self._lamport_copy = LamportCopyKernel(
                hidden_dim=contract.hidden_size,
                max_m=contract.max_num_tokens,
                ctas=_LAMPORT_COPY_CTAS,
                threads=_LAMPORT_COPY_THREADS,
            )
        register_cutedsl_warmup_provider(self)

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        contract = self.contract
        skinny_units = tuple(
            CuTeDSLCompileUnit(
                name="K3 latent MoE tail Skinny up-projection",
                key=(
                    "k3-latent-moe-tail-skinny-up-projection",
                    contract,
                    m,
                ),
                compile=partial(self._up_projection.compile_skinny, m),
            )
            for m in range(1, self._up_projection.skinny_max_m + 1)
        )
        return skinny_units + (
            CuTeDSLCompileUnit(
                name="K3 latent MoE tail dynamic up-projection",
                key=(
                    "k3-latent-moe-tail-dynamic-up-projection",
                    contract,
                    _MMA_TILER_MN,
                    _GEMM_CLUSTER_MN,
                    _B_PRIME_STAGES,
                ),
                compile=self._up_projection.compile_dynamic,
            ),
        )

    def __call__(
        self,
        routed_output: torch.Tensor | UnfinalizedMoEOutput,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            routed_output,
            shared_output,
            rms_weight,
            up_weight,
        )
        num_tokens = (
            routed_output.expanded_idx_to_permuted_idx.shape[0]
            if isinstance(routed_output, UnfinalizedMoEOutput)
            else routed_output.shape[0]
        )
        self._up_projection.ensure_compiled(num_tokens)
        latent, shared_shard = self._collective(
            routed_output,
            shared_output,
            rms_weight,
        )
        local_hidden_size = self.contract.hidden_size // self.contract.tp_size
        local_up_weight = up_weight.narrow(
            0,
            self.rank * local_hidden_size,
            local_hidden_size,
        )
        mailbox = self._up_projection(
            latent,
            local_up_weight,
            shared_shard,
        )
        return self._lamport_copy(
            mailbox,
            m=num_tokens,
        ).squeeze(0)

    def _validate_inputs(
        self,
        routed_output: torch.Tensor | UnfinalizedMoEOutput,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        contract = self.contract
        if isinstance(routed_output, UnfinalizedMoEOutput):
            num_tokens = routed_output.expanded_idx_to_permuted_idx.shape[0]
        else:
            if routed_output.ndim != 2:
                raise ValueError("routed_output must be a 2D tensor.")
            num_tokens = routed_output.shape[0]
            if routed_output.shape != (num_tokens, contract.latent_size):
                raise ValueError(
                    f"routed_output must have shape [M, {contract.latent_size}]."
                )
        if shared_output.shape != (num_tokens, contract.hidden_size):
            raise ValueError(
                f"shared_output must have shape [M, {contract.hidden_size}]."
            )
        if rms_weight.shape != (contract.latent_size,):
            raise ValueError(f"rms_weight must have shape [{contract.latent_size}].")
        if up_weight.shape != (contract.hidden_size, contract.latent_size):
            raise ValueError(
                "up_weight must have shape "
                f"[{contract.hidden_size}, {contract.latent_size}]."
            )
        if not 1 <= num_tokens <= contract.max_num_tokens:
            raise ValueError(
                "K3 latent-MoE tail fusion requires between 1 and "
                f"{contract.max_num_tokens} tokens."
            )

        tensors = (shared_output, rms_weight, up_weight)
        if any(tensor.device != contract.device for tensor in tensors):
            raise ValueError("All inputs must be on the contract device.")
        if any(tensor.dtype != contract.dtype for tensor in tensors):
            raise ValueError("All inputs must use the contract dtype.")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("All inputs must be contiguous.")
