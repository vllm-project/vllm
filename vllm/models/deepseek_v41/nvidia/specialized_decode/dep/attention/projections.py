# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared TP1 projections, opt-in and warmed before graph capture.

Only exact-shape recipes are reused from the TP4 table; other TP1 shapes
retain FlashInfer's selection until their own recipes have been validated.
The default bank is per stream. The executor uses a forward-only bank whose
addresses also survive capture on a different stream. Its projections are
sequential; side-stream indexer projections never use that bank.
Missing recipes or layouts retain the stock fallback.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fusion.quant_activation import as_quantized_activation
from vllm.models.deepseek_v41.common.ops.query_quant import mxfp8_scale_bytes
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import mxfp8


@dataclass(frozen=True)
class Recipe:
    backend: str
    tactic: tuple


def wide_smem_bytes(tile, tactic):
    """FlashInfer's MXFP8 allocator with the wider activation tile."""
    cta_k, ab, sf_stages, split_k = tactic
    if (
        tile not in (16, 32, 48)
        or cta_k not in (128, 256)
        or split_k not in (1, 2, 4, 8)
        or tile % split_k
    ):
        raise ValueError("unvalidated wide projection layout")
    if not 1 <= ab <= 16 or sf_stages != 1:
        raise ValueError("unvalidated wide projection stages")
    cursor = 16 * ab + 16 * sf_stages + 16 + 8 + 8 + 4
    mailbox_elements = (split_k - 1) * 128 * (tile // split_k)
    cursor = (cursor + 7) // 8 * 8 + 8 + mailbox_elements * 4
    sf_k = cta_k // 32
    for size in (
        128 * cta_k * ab,
        tile * cta_k * ab,
        128 * sf_k * ab,
        tile * sf_k * ab,
        128 * sf_k * ab,
    ):
        cursor = (cursor + 127) // 128 * 128 + size
    return cursor


class PreparedProjection:
    def __init__(self, layer, qa, recipe, workspace, out):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare projections before graph capture")
        self.layer = layer
        self.recipe = recipe
        self.rows, self.k = qa.data.shape
        self.n = layer.weight.shape[1]
        self.out = out
        self.workspace = workspace
        self.runner = mxfp8._cute_dsl_runner(qa.orig_dtype)
        self.compiled = None
        if recipe.backend == "stock":
            # Preserve FlashInfer's selection for TP1 shapes without a measured
            # pinned recipe. The public API still accepts our reusable output.
            self.runner = None
        elif recipe.backend == "cute-dsl":
            if (
                self.runner is None
                or recipe.tactic
                not in self.runner.get_valid_tactics(self.inputs(qa), None)
            ):
                raise ValueError("unsupported DEP projection tactic")
        elif recipe.backend == "wide":
            tile = (self.rows + 15) // 16 * 16
            if not 8 < self.rows <= 48 or qa.orig_dtype != torch.bfloat16:
                raise ValueError("wide projection requires 9-48 BF16 output rows")
            capacity = torch.cuda.get_device_properties(
                qa.data.device
            ).shared_memory_per_block_optin
            if wide_smem_bytes(tile, recipe.tactic) > capacity:
                raise ValueError(
                    "wide projection shared memory exceeds device capacity"
                )
            self.compiled = mxfp8._wide_kernel(recipe.tactic, tile, qa.data.device)
        else:
            raise ValueError("unsupported projection backend")

    def inputs(self, qa):
        return [
            qa.data,
            self.layer.weight,
            qa.scale,
            self.layer.weight_scale,
            qa.orig_dtype,
            self.out,
            self.workspace,
        ]

    def __call__(self, qa):
        if self.recipe.backend == "stock":
            from flashinfer import mm_mxfp8

            mm_mxfp8(
                qa.data,
                self.layer.weight,
                qa.scale,
                self.layer.weight_scale,
                out=self.out,
                out_dtype=qa.orig_dtype,
                backend="cute-dsl",
            )
        elif self.compiled is not None:
            mxfp8.launch_low_latency(
                self.compiled,
                self.layer.weight,
                self.layer.weight_scale,
                qa.data,
                qa.scale,
                self.out,
            )
        else:
            assert self.runner is not None
            self.runner(inputs=self.inputs(qa), tactic=self.recipe.tactic)
        return self.out.view(*qa.orig_shape[:-1], self.n)


class ProjectionBank:
    """One bank per executor; no global runner patch or hot-path autotune."""

    def __init__(self, max_rows=384, *, forward_only=False):
        if not 9 <= max_rows <= 384:
            raise ValueError("prepared projection capacity must be within 9-384")
        self.max_rows = max_rows
        self.forward_only = forward_only
        self.plans = {}
        self.workspaces = {}
        self.outputs = {}

    @staticmethod
    def activation(layer, x):
        if not mxfp8.mxfp8_eligible(layer):
            return None
        kernel = mxfp8._mxfp8_kernel(layer)
        assert kernel is not None
        qa = as_quantized_activation(x, kernel.input_quant_key())
        if (
            qa is None
            or qa.data.ndim != 2
            or not qa.data.is_contiguous()
            or qa.data.dtype != torch.float8_e4m3fn
            or qa.scale.ndim != 1
            or qa.scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or not qa.scale.is_contiguous()
            or qa.orig_dtype != torch.bfloat16
            or qa.data.shape[1] != layer.weight.shape[0]
            or qa.data.device != layer.weight.device
            or qa.scale.device != qa.data.device
            or qa.scale.nbytes != mxfp8_scale_bytes(*qa.data.shape)
        ):
            return None
        return qa

    def key(self, layer, qa):
        return (
            id(layer),
            tuple(qa.data.shape),
            qa.data.device,
            "forward"
            if self.forward_only
            else torch.cuda.current_stream(qa.data.device).cuda_stream,
        )

    def prepare(self, layer, x, recipe=None):
        qa = self.activation(layer, x)
        if qa is None or not 8 < qa.data.shape[0] <= self.max_rows:
            raise ValueError("unsupported prepared projection input")
        n = layer.weight.shape[1]
        if (n, qa.data.shape[1]) not in ((1792, 5120), (32768, 1280), (5120, 8192)):
            raise ValueError("unvalidated TP1 attention projection shape")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare projections before graph capture")
        if self.key(layer, qa) in self.plans and (
            recipe is None or self.plans[self.key(layer, qa)].recipe == recipe
        ):
            return self.plans[self.key(layer, qa)]
        if recipe is None:
            # Reuse the exact-shape table, never TP4's WQb/WOb tactic
            # for a different TP1 shape. Unmeasured shapes use stock selection.
            wide = mxfp8.wide_tactic(n, qa.data.shape[1], qa.data.shape[0])
            ranges = mxfp8.CUTE_DSL_TACTICS.get((n, qa.data.shape[1]), ())
            pinned = next((t for end, t in ranges if qa.data.shape[0] <= end), None)
            recipe = (
                Recipe("wide", wide[1])
                if wide is not None
                else Recipe("cute-dsl", pinned)
                if pinned is not None
                else Recipe("stock", ())
            )
        from flashinfer.gemm import gemm_base

        stream_key = self.key(layer, qa)[2:]
        # The public stock API owns its scratch, and the wide kernel needs none.
        if recipe.backend == "cute-dsl" and stream_key not in self.workspaces:
            self.workspaces[stream_key] = torch.empty(
                gemm_base.DEFAULT_WORKSPACE_SIZE,
                dtype=torch.uint8,
                device=qa.data.device,
            )
        out_key = (*stream_key, n, qa.orig_dtype)
        if out_key not in self.outputs:
            self.outputs[out_key] = torch.empty(
                self.max_rows, n, dtype=qa.orig_dtype, device=qa.data.device
            )
        out = self.outputs[out_key][: qa.data.shape[0]]
        workspace = (
            self.workspaces[stream_key] if recipe.backend == "cute-dsl" else None
        )
        plan = PreparedProjection(layer, qa, recipe, workspace, out)
        plan(qa)  # Compile on this stream before it participates in a graph.
        self.plans[self.key(layer, qa)] = plan
        return plan

    def run(self, layer, x):
        qa = self.activation(layer, x)
        if qa is None:
            return None
        plan = self.plans.get(self.key(layer, qa))
        return None if plan is None else plan(qa)
