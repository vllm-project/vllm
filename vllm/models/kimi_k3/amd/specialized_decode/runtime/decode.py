# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi-K3 specialized decode executor (layer mode): one routing decision per
step, then the layer kernels inside the stock layer loop.

Built by the model state once the weights are loaded (``KimiK3ModelState``): checks
the loaded weights, builds the per-rank launch state (``runner``) and the layer
kernels of every width, and compiles every build, before memory profiling and graph
capture. ``prepare_kv`` binds it to every KV cache vLLM allocates. Each step
``plan`` returns a ``Step`` or None; ``Step.layer(i)`` is the kernel replacing
decoder layer i (None for vLLM's layers). Weight reloads are not supported."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import get_dcp_group, get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import FAILED
from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
    DCP,
    FFN,
    WHOLE,
    DecodeConfig,
    layer_kinds,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.guards import (
    FailStop,
    cache_reasons,
    mla_weight_reasons,
    weight_reasons,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import (
    FfnLayer,
    MlaOpsLayer,
    WholeLayer,
    compile_build,
    m1c_at,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import tp_agree
from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

logger = init_logger(__name__)

Kernel = WholeLayer | FfnLayer | MlaOpsLayer


class Step(NamedTuple):
    """A planned step: ``width`` rows of whole uniform spec-verify requests."""

    width: int
    attn_metadata: Mapping[str, Any]  # by layer name
    kernels: Mapping[int, Kernel]  # by decoder layer, at this width

    def layer(self, idx: int) -> Callable | None:
        """The kernel replacing decoder layer idx in this step, or None (vLLM's)."""
        kernel = self.kernels.get(idx)
        return None if kernel is None else partial(kernel.launch, self)


def step_reason(
    attn_metadata, kda_prefix: str, width: int, mla_prefix: str = ""
) -> str | None:
    """None if a step's KDA and MLA metadata (identical on every TP rank) are a
    pure spec-verify step of whole uniform requests filling ``width`` rows, else
    why not."""
    if not isinstance(attn_metadata, dict):
        return "no metadata"
    m: Any = attn_metadata.get(kda_prefix)
    if m is None or m.spec_sequence_masks is None:
        return "not spec-verify"
    if m.num_prefills or m.num_decodes or m.num_spec_decodes < 1:
        return "not pure decode"
    q = m.spec_state_indices_tensor.size(-1)
    if m.uniform_spec_sequence_length != q or m.num_spec_decodes * q != width:
        return "not uniform"
    if mla_prefix:
        a: Any = attn_metadata.get(mla_prefix)
        if a is None or a.decode is None:
            return "no MLA decode metadata"
        if a.num_prefills or a.num_decode_tokens != width:
            return "MLA: not pure decode"
        if a.decode.max_qo_len != q:
            return "MLA: not uniform"
        # Gluon serves bf16 caches only; M1a hands AITER an fp8 query
        if getattr(a.decode, "use_gluon_decode", False) or getattr(
            a.decode, "use_gluon_verify", False
        ):
            return "MLA: Gluon decode"
    return None


def agree(reasons: list[str], group, what: str) -> None:
    """Raises on every TP rank if any rank has ``reasons``: a rank that refused
    alone would leave the others spinning in-kernel."""
    if not tp_agree(not reasons, group):
        why = "; ".join(reasons) if reasons else "another TP rank refused"
        raise ValueError(
            f"Kimi-K3 specialized decode does not support {what}: {why} "
            "(decode_backend=auto runs vLLM's decode)"
        )


class SpecializedDecode:
    """The executor: builds the layer kernels, binds them to each KV cache and
    plans every step."""

    def __init__(self, backbone, config: DecodeConfig):
        tp, dcp = get_tp_group(), get_dcp_group()
        reasons = weight_reasons(backbone)
        if dcp.world_size > 1 and (
            dcp.world_size != DCP or dcp.rank_in_group != tp.rank_in_group
        ):
            reasons.append(
                f"DCP{dcp.world_size} rank {dcp.rank_in_group} on TP rank "
                f"{tp.rank_in_group} (the DCP group must be the TP group)"
            )
        agree(reasons, tp.cpu_group, "the loaded weights")
        self.backbone, self.config = backbone, config
        kinds = {
            i: kind
            for i, kind in layer_kinds(backbone.config).items()
            if backbone.start_layer <= i < backbone.end_layer
        }
        ffn = [i for i, kind in kinds.items() if kind == FFN]
        m1 = config.m1 and bool(ffn)
        if m1:
            why = mla_weight_reasons(backbone)
            m1 = tp_agree(not why, tp.cpu_group)
            if not m1:
                logger.info(
                    "Kimi-K3 specialized decode: the MLA layers run vLLM's "
                    "attention ops, then K2 (M1 does not support: %s)",
                    "; ".join(why) if why else "another TP rank's MLA layers",
                )
        runner = self.runner = MonoRunner(next(backbone.parameters()).device, m1=m1)
        runner.num_spec = config.q - 1
        runner.interleave = config.interleave
        runner.first_layer = min(kinds)
        whole = [i for i, kind in kinds.items() if kind == WHOLE]
        runner.kda_prefix = backbone.layers[whole[0]].self_attn.prefix
        if m1:
            # the M1 layers' metadata: their MLAAttention's
            wrapper = backbone.layers[ffn[0]].self_attn.mla_attn
            runner.mla_prefix = wrapper.mla_attn.layer_name
        mla_kernel = FfnLayer if m1 else MlaOpsLayer
        self.kernels: dict[int, dict[int, Kernel]] = {
            w: {
                i: (WholeLayer if kind == WHOLE else mla_kernel)(
                    backbone.layers[i], runner, w
                )
                for i, kind in kinds.items()
            }
            for w in config.widths
        }
        self.kda: list[WholeLayer] = [
            k for k in self.kernels[config.widths[0]].values()
            if isinstance(k, WholeLayer)
        ]  # fmt: skip
        self.mla: list[FfnLayer] = [
            k for k in self.kernels[config.widths[0]].values()
            if isinstance(k, FfnLayer)
        ]  # fmt: skip
        builds = {
            b for by in self.kernels.values() for k in by.values() for b in k.builds()
        }
        for build in builds:
            compile_build(build)
        self.failstop = FailStop(
            runner.health[FAILED : FAILED + 1], runner.rank, tp.world_size, tp.cpu_group
        )
        # (data_ptr, numel) of the first KDA layer's state prepare_kv bound
        self._kv: tuple[int, int] | None = None
        m1c = [w for w in config.widths if m1c_at(runner, w)]
        mla = f"M1 (M1c in K2 at widths {m1c})" if m1c else "M1"
        logger.info(
            "Kimi-K3 specialized decode: %d layers (MLA: %s), widths %s, %d "
            "builds; other steps run vLLM's decode",
            len(kinds),
            mla if m1 else "vLLM's ops",
            list(config.widths),
            len(builds),
        )

    def _kv_key(self) -> tuple[int, int] | None:
        state = self.kda[0].layer.self_attn.kv_cache
        if not isinstance(state, (list, tuple)) or len(state) != 2:
            return None
        if state[1].numel() == 0:
            return None
        return state[1].data_ptr(), state[1].numel()

    def prepare_kv(self) -> None:
        """Checks and binds the KDA states and MLA caches vLLM just bound (the
        CUDA-graph memory profiling cache, then the real one)."""
        self._kv = None
        if self._kv_key() is None:
            return
        caches = {k.layer.self_attn.prefix: k.caches() for k in self.kda}
        caches.update({k.wrapper.mla_attn.layer_name: k.caches() for k in self.mla})
        agree(cache_reasons(caches), get_tp_group().cpu_group, "the KV caches")
        self._kv = self._kv_key()

    def plan(self, positions: torch.Tensor, intermediate_tensors) -> Step | None:
        """This step's plan, or None to run vLLM's layers.

        Steps that run vLLM's layers:
        - pipeline-parallel stage inputs, and steps before a KV cache is bound
          (profiling, dummy runs);
        - PIECEWISE steps (captured without attention metadata);
        - widths without kernels (more than MAX_ROWS rows);
        - steps that are not pure spec-verify steps of whole uniform requests
          (``step_reason``).

        Eager steps run the kernels too. Every predicate is the same on all TP
        ranks: the kernels' mailboxes need every rank on one path. Under FULL
        graphs this runs at capture only and the plan is baked in; a replay with
        fewer requests than its graph pads whole requests (NULL_BLOCK_ID state,
        one accepted token), which the kernels serve like real ones.
        """
        if intermediate_tensors is not None or self._kv is None:
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        width = positions.size(-1)
        kernels = self.kernels.get(width)
        if kernels is None:
            return None
        md = context.attn_metadata
        runner = self.runner
        if step_reason(md, runner.kda_prefix, width, runner.mla_prefix) is not None:
            return None
        assert isinstance(md, dict)
        if self._kv_key() != self._kv:
            raise RuntimeError(
                "Kimi-K3 specialized decode: the KV cache was re-bound without "
                "prepare_kv (ModelState.initialize_kv_cache)"
            )
        return Step(width, md, kernels)

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (eager on every rank, also
        under graph replay). Raises on every rank once a rank gave up an in-kernel
        wait."""
        self.failstop.after_step()
