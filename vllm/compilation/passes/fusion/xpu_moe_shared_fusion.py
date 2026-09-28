# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replace routed + shared MoE with one fused XPU op for small token counts.

Matches, per MoE layer::

    out = vllm.moe_forward_shared(hidden, hidden, hidden, None, layer, 0)
    shared, routed = out[0], out[1]
    y = shared + routed

and rewrites it to ``y = vllm.xpu_moe_shared_fused(hidden, hidden, layer)``.
The TP all-reduce consuming ``y`` is kept. Only applied for compile ranges
whose end is <= ``pass_config.xpu_moe_shared_fusion_max_token_num``, and only
when every MoE layer of the model supports the fused op; anything that does
not match the pattern is left unchanged.
"""

import operator

import torch
from torch import fx

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


def _is_fp8_per_tensor(config: VllmConfig) -> bool:
    quant = config.quant_config
    if quant is None or quant.get_name() != "fp8":
        return False
    return getattr(quant, "weight_block_size", None) is None


def _xpu_moe_shared_fused_norm_available() -> bool:
    try:
        from vllm._xpu_ops import xpu_moe_shared_fused_norm_available
    except ImportError:
        return False
    return xpu_moe_shared_fused_norm_available(2048)


def _xpu_moe_shared_fused_available() -> bool:
    # vllm._xpu_ops needs vllm_xpu_kernels; treat an import failure as "not
    # available" so compilation never fails because of this pass.
    try:
        from vllm._xpu_ops import xpu_moe_shared_fused_available
    except ImportError:
        return False
    return xpu_moe_shared_fused_available()


class XpuMoESharedFusionPass(VllmInductorPass):
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        pass_config = config.compilation_config.pass_config
        self.max_token_num = pass_config.xpu_moe_shared_fusion_max_token_num
        mc = config.model_config
        hf = mc.hf_text_config if mc is not None else None
        pc = config.parallel_config
        self.enabled = (
            hf is not None
            and mc.dtype == torch.float16
            and getattr(hf, "hidden_size", None) == 2048
            and getattr(hf, "num_experts", None) == 256
            and getattr(hf, "num_experts_per_tok", None) == 8
            and getattr(hf, "shared_expert_intermediate_size", 0) > 0
            and _is_fp8_per_tensor(config)
            and config.lora_config is None
            and not pc.enable_expert_parallel
            and not pc.enable_eplb
            and _xpu_moe_shared_fused_available()
        )
        if not self.enabled:
            logger.warning_once(
                "XpuMoESharedFusionPass disabled: model, quantization, parallel "
                "config or kernel not supported."
            )
        else:
            self.enabled = self._all_moe_layers_supported(config)
        self.matched_count = 0
        self.norm_fused_count = 0
        self.fuse_input_norm = self.enabled and _xpu_moe_shared_fused_norm_available()

    @staticmethod
    def _all_moe_layers_supported(config: VllmConfig) -> bool:
        from vllm._xpu_ops import xpu_moe_shared_fused_unsupported_reason
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        runners = get_layers_from_vllm_config(config, MoERunner)
        if not runners:
            logger.warning_once("XpuMoESharedFusionPass disabled: no MoE layers.")
            return False
        for name, runner in runners.items():
            reason = xpu_moe_shared_fused_unsupported_reason(runner)
            if reason is not None:
                logger.warning_once(
                    "XpuMoESharedFusionPass disabled: MoE layer %s is not "
                    "supported (%s).",
                    name,
                    reason,
                )
                return False
        return True

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.enabled and compile_range.end <= self.max_token_num

    @staticmethod
    def _match(node: fx.Node) -> tuple[fx.Node | None, str]:
        """Return the `add` node combining shared and routed outputs."""
        hidden, router_in, shared_in, input_ids, _layer, unpadded = node.args[:6]
        if router_in is not hidden:
            return None, "router input is not the hidden states"
        if shared_in is not hidden or input_ids is not None or unpadded != 0:
            return None, "unsupported moe_forward_shared arguments"
        getitems = {}
        for user in node.users:
            if user.target is not operator.getitem or len(user.users) != 1:
                return None, "unexpected users of moe_forward_shared"
            getitems[user.args[1]] = user
        if set(getitems) != {0, 1}:
            return None, "unexpected users of moe_forward_shared"
        add = next(iter(getitems[0].users))
        if (
            add is not next(iter(getitems[1].users))
            or add.target is not torch.ops.aten.add.Tensor
            or set(add.args) != {getitems[0], getitems[1]}
        ):
            return None, "shared and routed outputs are not simply added"
        val = hidden.meta.get("val")
        if val is None or val.dtype != torch.float16 or val.shape[-1] != 2048:
            return None, "unsupported hidden states dtype/shape"
        return add, ""

    @staticmethod
    def _match_input_norm(hidden: fx.Node, moe: fx.Node):
        """If `hidden` is the normed output of a Gemma fused_add_rms_norm
        used only by `moe`, return (norm node, x, residual, weight, eps)."""
        if hidden.target is not operator.getitem or hidden.args[1] != 0:
            return None
        norm = hidden.args[0]
        if not (
            isinstance(norm, fx.Node)
            and norm.target is torch.ops.vllm_ir.fused_add_rms_norm.default
            and set(hidden.users) == {moe}
        ):
            return None
        if len(norm.args) > 4 and norm.args[4] is not None:
            return None
        if any(
            u.target is not operator.getitem or u.args[1] not in (0, 1)
            for u in norm.users
        ):
            return None
        x, residual, weight, eps = norm.args[:4]
        # GemmaRMSNorm passes weight.float() + 1.0.
        if not (
            isinstance(weight, fx.Node)
            and weight.target is torch.ops.aten.add.Tensor
            and len(weight.args) == 2
            and weight.args[1] == 1.0
        ):
            return None
        conv = weight.args[0]
        if not (
            isinstance(conv, fx.Node)
            and conv.target is torch.ops.prims.convert_element_type.default
            and conv.args[1] == torch.float32
        ):
            return None
        w = conv.args[0]
        w_val = w.meta.get("val") if isinstance(w, fx.Node) else None
        if not (
            isinstance(w_val, torch.Tensor)
            and w_val.dtype == torch.float16
            and tuple(w_val.shape) == (2048,)
        ):
            return None
        return norm, x, residual, w, eps

    def _rewrite_with_norm(self, graph, node, add, layer_name, m) -> None:
        norm, x, residual, w, eps = m
        with graph.inserting_before(add):
            fused = graph.call_function(
                torch.ops.vllm.xpu_moe_shared_fused_resadd_norm.default,
                (x, residual, w, eps, layer_name),
            )
            out = graph.call_function(operator.getitem, (fused, 0))
            new_residual = graph.call_function(operator.getitem, (fused, 1))
        out.meta["val"] = add.meta["val"]
        new_residual.meta["val"] = add.meta["val"]
        fused.meta["val"] = (add.meta["val"], add.meta["val"])
        add.replace_all_uses_with(out)
        norm_users = list(norm.users)
        for u in norm_users:
            if u.args[1] == 1:
                u.replace_all_uses_with(new_residual)
        getitems = list(node.users)
        graph.erase_node(add)
        for g in getitems:
            graph.erase_node(g)
        graph.erase_node(node)
        for u in norm_users:
            graph.erase_node(u)
        graph.erase_node(norm)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        self.norm_fused_count = 0
        skipped = 0
        target = torch.ops.vllm.moe_forward_shared.default
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target is not target:
                continue
            add, reason = self._match(node)
            if add is None:
                logger.debug("XpuMoESharedFusionPass skipped %s: %s", node, reason)
                skipped += 1
                continue
            hidden, router_in, _, _, layer_name = node.args[:5]
            norm_match = (
                self._match_input_norm(hidden, node) if self.fuse_input_norm else None
            )
            if norm_match is not None:
                self._rewrite_with_norm(graph, node, add, layer_name, norm_match)
                self.matched_count += 1
                self.norm_fused_count += 1
                continue
            with graph.inserting_before(add):
                fused = graph.call_function(
                    torch.ops.vllm.xpu_moe_shared_fused.default,
                    (hidden, router_in, layer_name),
                )
            fused.meta["val"] = add.meta["val"]
            add.replace_all_uses_with(fused)
            getitems = list(node.users)
            graph.erase_node(add)
            for g in getitems:
                graph.erase_node(g)
            graph.erase_node(node)
            self.matched_count += 1
        logger.info(
            "XpuMoESharedFusionPass replaced %d MoE layers (%d with the input "
            "RMSNorm, %d not matched)",
            self.matched_count,
            self.norm_fused_count,
            skipped,
        )

    def uuid(self) -> str:
        return (
            self.hash_source(self)
            + f"|{self.enabled}|{self.fuse_input_norm}|{self.max_token_num}"
        )
