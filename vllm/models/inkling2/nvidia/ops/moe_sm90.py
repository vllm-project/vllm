# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Hopper: Humming expert GEMMs (bf16
activations, NVFP4 / MXFP8 weights) in place of the TRT-LLM-gen ones.

Rows stay in the flat (token * topk + slot) order: the gate/up GEMM
gathers each row's token, the down GEMM reads and writes row by row, and
`ops.moe.combine` sums them with the identity `src2dst` (-1: not local).
"""

import json
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.models.inkling2.common.ops.moe import swiglu

if TYPE_CHECKING:
    from vllm.utils.humming import HummingInputSchema

# Row block size when no Humming tuning range covers the row count.
_DEFAULT_BLOCK_M = 64


class HummingExperts:
    """Humming tensors and kernel configs of the two expert projections.

    Args:
        weights: Per projection (`w13`, `w2`), the row-major checkpoint
            weights `[E, N, K / pack]`, block scales `[E, N, K / group]`
            and, for NVFP4, the per-expert global scales `[E]` (else None).
        shapes: `(N, K)` of each projection.
        nvfp4: NVFP4 (else MXFP8) weights.
        input_schema: Quantized input contract, or None for BF16 activations.

    """

    def __init__(
        self,
        weights: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        input_schema: "HummingInputSchema | None" = None,
    ) -> None:
        from vllm.utils.humming import (
            GemmType,
            HummingInputSchema,
            ModeloptMxfp8WeightSchema,
            ModeloptNvfp4WeightSchema,
            get_heuristics_config,
            prepare_and_transform_humming_tensors,
        )

        self.configs = {}
        self.tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.tunings: dict[str, list] = {}
        self.tuning_strs: dict[str, str] = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            weight, scale, global_scale = weights[name]
            num_experts = weight.shape[0]
            tensors = {"weight": weight, "weight_scale": scale}
            if nvfp4:
                source = ModeloptNvfp4WeightSchema()
                tensors["weight_scale_2"] = global_scale
            else:
                source = ModeloptMxfp8WeightSchema()
            schema, tensors = source.convert_humming(
                tensors=tensors,
                shape_n_stacks=[n],
                shape_k_stacks=[k],
                param_dtype=torch.bfloat16,
                num_experts=num_experts,
            )
            config, tensors = prepare_and_transform_humming_tensors(
                tensors=tensors,
                shape_n=n,
                shape_k=k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema or HummingInputSchema(),
                weight_schema=schema,
                num_experts=num_experts,
                torch_dtype=torch.bfloat16,
                device=weight.device,
            )
            assert config.shape_n == n and config.shape_k == k, (n, k)
            tuning = get_heuristics_config(
                layer_config=config,
                use_f16_accum=False,
                use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
                gemm_type=GemmType.INDEXED,
            )
            # Stream-K adds the K slices into the output with bf16 atomics,
            # whose order (and so the result) changes from run to run.
            for _, _, kernel_config in tuning:
                kernel_config["use_stream_k"] = False
            self.configs[name] = config
            self.tensors[name] = tensors
            self.tunings[name] = tuning
            self.tuning_strs[name] = json.dumps(tuning)
        self.num_experts = num_experts
        self.compute_config = json.dumps(
            {
                "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
                "use_f16_accum": False,
                "gemm_type": GemmType.INDEXED.value,
            }
        )
        self.locks = torch.zeros(2048, dtype=torch.int32, device=weight.device)

    def _block_m(self, name: str, rows: int) -> int:
        for lower, upper, config in self.tunings[name]:
            if lower < rows <= upper:
                return config["block_shape"][0]
        return _DEFAULT_BLOCK_M

    def _gemm(
        self,
        name: str,
        inputs: torch.Tensor,
        rows: int,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        top_k: int,
        valid_rows: int,
    ) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        config = self.configs[name]
        tensors = self.tensors[name]
        out = torch.empty(
            (rows, config.shape_n), dtype=torch.bfloat16, device=inputs.device
        )
        sorted_ids, expert_ids, num_tokens_padded = alignment
        return humming_forward(
            config,
            inputs=inputs,
            weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"),
            weight_scale_2=tensors.get("weight_scale_2"),
            outputs=out,
            locks=self.locks,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            valid_shape_m=valid_rows,
            compute_config=self.compute_config,
            tuning_config=self.tuning_strs[name],
        )

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (`select_experts`); the other rows are left unset."""
        tokens, topk = ids.shape
        rows = tokens * topk
        # Expected local rows, which pick the kernel configs.
        valid_rows = -(-rows * self.num_experts // num_global_experts)
        block_m = {name: self._block_m(name, valid_rows) for name in ("w13", "w2")}
        alignments = {
            size: moe_align_block_size(
                ids, size, self.num_experts, ignore_invalid_experts=True
            )[:3]
            for size in set(block_m.values())
        }
        gate_up = self._gemm(
            "w13", x, rows, alignments[block_m["w13"]], topk, valid_rows
        )
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm(
            "w2", activated, rows, alignments[block_m["w2"]], 1, valid_rows
        )


def __getattr__(name: str):
    """Names that moved to `common/ops/` stay importable from this module."""
    from importlib import import_module

    for module in (
        "moe",
        "activation",
        "linear",
    ):
        moved = import_module(f"vllm.models.inkling2.common.ops.{module}")
        if hasattr(moved, name):
            return getattr(moved, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
