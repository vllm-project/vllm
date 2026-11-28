# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2 checkpoints (`loop_config`).

The core layers `[core_start, core_end)` run `num_passes` times: a prelude,
then each pass from the AttnRes state after the prelude, then the coda from
the last pass's state. Each pass of a core layer has its own KV and
conv-state caches. In passes after the first, every core sublayer input
(attention q/k/v and MLP) is blended with a message: a normalized-value
AttnRes read of the previous pass's blocks, merged by a gated low-rank
controller. Each sublayer's short conv runs fused with the next AttnRes mix
(and small batches at TP > 1 with its all-reduce); without sequence parallel.
"""

from collections.abc import Sequence

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import get_pp_group, tensor_model_parallel_all_reduce
from vllm.distributed.utils import get_pp_indices
from vllm.sequence import IntermediateTensors

from ..configs import Inkling2ModelConfig
from ..quant_config import Inkling2QuantConfig
from .model import (
    Inkling2DecoderLayer,
    Inkling2ForConditionalGeneration,
    Inkling2Model,
    Inkling2RMSNorm,
)
from .ops.attn_res import attn_res
from .ops.linear import linear_k_groups
from .ops.ll_gemm_sm120 import sm12x_bf16_ll_gemm, use_sm12x_bf16_ll_gemm
from .ops.loop import blend_rms_norm, pair_rms_norm
from .sconv import conv_metadata

LOOP_STREAMS = ("attn", "k", "v", "mlp")


def _split_range(start: int, stop: int, num_stages: int) -> list[tuple[int, int]]:
    total = stop - start
    base, rem = divmod(total, num_stages)
    ranges: list[tuple[int, int]] = []
    cur = start
    for idx in range(num_stages):
        nxt = cur + base + (1 if idx < rem else 0)
        ranges.append((cur, nxt))
        cur = nxt
    return ranges


def _get_loop_pp_indices(
    num_hidden_layers: int, core: range, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """Partition a looped Inkling2 model into `pp_size` contiguous stages
    without splitting the recurrent `core` across stages."""
    if pp_size == 1:
        return 0, num_hidden_layers

    if envs.VLLM_PP_LAYER_PARTITION is not None:
        ranges = [get_pp_indices(num_hidden_layers, r, pp_size) for r in range(pp_size)]
        if not any(s <= core.start and core.stop <= e for s, e in ranges):
            raise ValueError(
                f"VLLM_PP_LAYER_PARTITION={envs.VLLM_PP_LAYER_PARTITION!r} splits "
                f"the Inkling2 loop core layers [{core.start}, {core.stop}) across "
                "pipeline stages; the recurrent core must stay on a single stage."
            )
        return ranges[pp_rank]

    prelude = core.start
    coda = num_hidden_layers - core.stop
    max_stages = prelude + 1 + coda
    if pp_size > max_stages:
        raise ValueError(
            f"pipeline_parallel_size={pp_size} exceeds the maximum {max_stages} "
            f"stages for Inkling2 loop core [{core.start}, {core.stop}) "
            f"(prelude={prelude}, coda={coda}); the recurrent core must stay on "
            "a single stage."
        )

    if pp_size == 2:
        cut = core.stop if coda > prelude else core.start
        ranges = [(0, cut), (cut, num_hidden_layers)]
        return ranges[pp_rank]

    rem_stages = pp_size - 1
    p_stages = 1 if prelude > 0 else 0
    c_stages = 1 if coda > 0 else 0
    while p_stages + c_stages < rem_stages:
        p_load = prelude / p_stages if 0 < p_stages < prelude else -1.0
        c_load = coda / c_stages if 0 < c_stages < coda else -1.0
        if p_load >= c_load:
            p_stages += 1
        else:
            c_stages += 1

    ranges = []
    if p_stages > 0:
        ranges.extend(_split_range(0, core.start, p_stages))
    ranges.append((core.start, core.stop))
    if c_stages > 0:
        ranges.extend(_split_range(core.stop, num_hidden_layers, c_stages))
    return ranges[pp_rank]


class Inkling2MessageController(nn.Module):
    """Blends a loop message into a core sublayer input: from the rank-`r`
    projection `h` of the RMS-normalized `[dest, message]`,
    `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message` in fp32.

    The weights are views of the layer's stream-stacked weights
    (`Inkling2LoopDecoderLayer`), which run every stream at once; `forward`
    is the per-stream reference.
    """

    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)

    def forward(self, dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """fp32 `(T, D)` `dest` and `message` -> fp32 `(T, D)`."""
        pair = torch.stack((dest, message), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        normalized = pair.to(torch.bfloat16).flatten(1)
        hidden = linear_k_groups(normalized, self.down_weight, torch.bfloat16)
        gain = torch.mm(hidden, self.gain_weight.t(), out_dtype=torch.float32)
        forget = torch.mm(hidden, self.forget_weight.t(), out_dtype=torch.float32)
        return (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    """A loop core layer: per-pass caches and a message controller per
    stream (`LOOP_STREAMS`), whose weights are stacked over the streams."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        streams = len(LOOP_STREAMS)
        # down (S, r, 2D) and [gain; forget] (S, 2D, r), bf16.
        self.register_buffer(
            "controller_down",
            torch.empty(streams, rank, 2 * width, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "controller_gain_forget",
            torch.empty(streams, 2 * width, rank, dtype=torch.bfloat16),
            persistent=False,
        )
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2MessageController(
                    self.controller_down[i],
                    self.controller_gain_forget[i, :width],
                    self.controller_gain_forget[i, width:],
                )
                for i, stream in enumerate(LOOP_STREAMS)
            }
        )

    def process_weights_after_loading(self) -> None:
        # The controllers' weights must still alias the stacked ones.
        for i, controller in enumerate(self.controllers.values()):
            down, gain_forget = self.controller_down, self.controller_gain_forget
            assert controller.down_weight.data_ptr() == down[i].data_ptr()
            assert controller.gain_weight.data_ptr() == gain_forget[i].data_ptr()

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The `streams` controllers on fp32 `(S, T, D)` `dest` (may
        broadcast over streams) and `message`, each output RMS-normalized
        with its row of `norm_weight` `(S, D)`: bf16 `(S, T, D)`."""
        pair = pair_rms_norm(dest, message)
        down = self.controller_down[streams]
        up = self.controller_gain_forget[streams]
        if use_sm12x_bf16_ll_gemm(pair, down):
            hidden = sm12x_bf16_ll_gemm(pair, down)
            gain_forget = sm12x_bf16_ll_gemm(hidden, up, out_dtype=torch.float32)
        else:
            hidden = torch.bmm(pair, down.transpose(1, 2))
            gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)


class Inkling2LoopModel(Inkling2Model):
    seq_parallel_paths = False

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        # Serving options set through --hf-overrides skip the constructor.
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        self.has_loop_core = (
            self.start_layer <= self.loop_core.start
            and self.loop_core.stop <= self.end_layer
        )
        if self.has_loop_core:
            # Message pseudo-queries (raw), two per core layer: attn/mlp and k/v.
            shape = (2 * len(self.loop_core), config.hidden_size)
            self.middle_cycle_cross_pass_queries = nn.Parameter(
                torch.zeros(shape, dtype=torch.float32), requires_grad=False
            )
            self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
                torch.zeros(shape, dtype=torch.float32), requires_grad=False
            )
            # Per core layer, the `LOOP_STREAMS` queries scaled by D**-0.5.
            self.register_buffer(
                "message_queries",
                torch.zeros(
                    len(self.loop_core),
                    len(LOOP_STREAMS),
                    config.hidden_size,
                    dtype=torch.float32,
                ),
                persistent=False,
            )

    def _pp_indices(self, config: Inkling2ModelConfig) -> tuple[int, int]:
        pp_group = get_pp_group()
        return _get_loop_pp_indices(
            config.num_hidden_layers,
            config.loop_core_layers,
            pp_group.rank_in_group,
            pp_group.world_size,
        )

    def process_weights_after_loading(self) -> None:
        if not self.has_loop_core:
            return
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)
        return super()._make_layer(config, layer_id, quant_config, prefix)

    def _max_loop_blocks(self) -> int:
        """Completed AttnRes blocks this stage holds at most: each pass restarts
        from the prelude's, and the coda continues the last pass."""
        return -(-self.end_layer // self.block_size)

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs: torch.Tensor | None,
        bank: torch.Tensor | None,
        next_query: torch.Tensor | None,
        next_norm: nn.Module | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
        plan: list[tuple[bool, bool]],
    ) -> tuple[int, torch.Tensor | None]:
        """Run decoder layer `i` in loop pass `pass_idx` on its `inputs`
        (the attention inputs; for a visit with messages from `bank`, the
        fp32 mixes they blend into). Returns the number of completed AttnRes
        blocks after it and the mixes of its output with `next_query` through
        `next_norm` (fp32 without it), or None at a non-last PP stage boundary."""
        layer = self.layers[i]
        assert isinstance(layer, Inkling2DecoderLayer)
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        workspace = self.allreduce_workspace
        (attn_fused, attn_two_shot), (moe_fused, moe_two_shot) = plan
        messages = None
        attn_inputs: Sequence[torch.Tensor]
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            # All four streams' messages: a mix of the bank's two blocks.
            messages = attn_res(
                bank[:1],
                bank[1],
                None,
                self.message_queries[i - self.loop_core.start],
                None,
                0.0,
                True,
            )
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            attn_inputs = layer.blend(
                slice(0, 3), dests, messages[:3], norm_weight, norm.eps
            )
        elif inputs is None and num_blocks == 0:
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        elif inputs is None:
            query = self.attn_res.attn_queries[i - 1]
            attn_inputs = self._mix(stack[:num_blocks], partial, query, layer.attn_norm)
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

        # The MLP input mixes the blocks with (partial +) attn_out.
        first = i % self.block_size == 0
        attn_out = torch.empty_like(partial)
        mix_args = (
            positions,
            attn_out,
            stack[: num_blocks + first],
            None if first else partial,
            self.attn_res.mlp_queries[i],
            None if messages is not None else layer.mlp_norm,
        )
        if attn_fused:
            assert workspace is not None
            mlp_in = layer.attn_sconv.all_reduce_mix(
                workspace,
                *mix_args,
                x=attn_partial,
                two_shot=attn_two_shot,
                owner=owner,
            )
        else:
            attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
            mlp_in = layer.attn_sconv.conv_mix(attn_reduced, *mix_args, owner=owner)
        if messages is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer)
            norm = layer.mlp_norm
            mlp_in = layer.blend(
                slice(3, 4), mlp_in, messages[3:], norm.weight, norm.eps
            )

        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        if next_query is None:
            layer.mlp_sconv(
                layer.mlp(mlp_in[0], pass_idx=pass_idx),
                positions,
                addend=attn_out,
                residual=residual,
                out=stack[num_blocks],
                owner=owner,
            )
            return num_blocks, None
        mix_args = (
            positions,
            stack[num_blocks],
            stack[:num_blocks],
            None,
            next_query,
            next_norm,
        )
        if moe_fused:
            assert workspace is not None
            outputs = layer.mlp_sconv.all_reduce_mix(
                workspace,
                *mix_args,
                combine=layer.mlp(mlp_in[0], reduce=False, pass_idx=pass_idx),
                addend=attn_out,
                residual=residual,
                two_shot=moe_two_shot,
                owner=owner,
            )
        else:
            # The MoE output is all-reduced (in bf16) inside layer.mlp.
            outputs = layer.mlp_sconv.conv_mix(
                layer.mlp(mlp_in[0], pass_idx=pass_idx),
                *mix_args,
                addend=attn_out,
                residual=residual,
                owner=owner,
            )
        return num_blocks, outputs

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """Prelude, `num_passes` core passes, then the coda. Each sublayer's
        short conv runs fused with the next AttnRes mix (and small batches
        with its all-reduce)."""
        pp_group = get_pp_group()
        is_first_rank = pp_group.is_first_rank
        is_last_rank = pp_group.is_last_rank
        if is_first_rank:
            assert inputs_embeds is not None
            num_tokens, hidden = inputs_embeds.shape
            ref_tensor = inputs_embeds
        else:
            assert intermediate_tensors is not None
            received = intermediate_tensors["attn_res_stack"]
            num_tokens, received_blocks, hidden = received.shape
            assert received_blocks == self.init_blocks + 1
            ref_tensor = received
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = ref_tensor.new_empty((self._max_loop_blocks() + 1, num_tokens, hidden))
        if is_first_rank:
            assert inputs_embeds is not None
            stack[0].copy_(inputs_embeds)
            num_blocks = 0
        else:
            stack[: self.init_blocks + 1].copy_(received.transpose(0, 1))
            num_blocks = self.init_blocks
        tau = self._log_scaling_tau(positions)
        workspace = self.allreduce_workspace
        plan = [(False, False)] * 2
        if self.start_layer < self.end_layer:
            first_layer = self.layers[self.start_layer]
            assert isinstance(first_layer, Inkling2DecoderLayer)
            if (
                workspace is not None
                and conv_metadata(first_layer.conv_state) is not None
            ):
                plan = workspace.plan(num_tokens)
        # (layer, pass, global_visit) for this stage; prelude and coda run as pass 0.
        visits: list[tuple[int, int, int]] = []
        for i in range(self.start_layer, min(self.end_layer, core.start)):
            visits.append((i, 0, i))
        if self.has_loop_core:
            for p in range(num_passes):
                for i in core:
                    visits.append((i, p, i + p * len(core)))
        coda_offset = (num_passes - 1) * len(core)
        for i in range(max(self.start_layer, core.stop), self.end_layer):
            visits.append((i, 0, i + coda_offset))
        aux_hidden_states = []
        next_query: torch.Tensor | None
        next_norm: nn.Module | None
        inputs: torch.Tensor | None = None
        bank = prelude_blocks = prelude_partial = prelude_dests = None
        for idx, (i, pass_idx, global_visit) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) and partial as
                # the messages' bank.
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        query,
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if global_visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].clone())
            if idx + 1 == len(visits):
                if is_last_rank:
                    next_query, next_norm = self.attn_res.output_query, self.norm
                else:
                    next_query, next_norm = None, None
            else:
                next_i, next_pass, _ = visits[idx + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                if next_pass:
                    next_norm = None
                else:
                    next_layer = self.layers[next_i]
                    assert isinstance(next_layer, Inkling2DecoderLayer)
                    next_norm = next_layer.attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
                plan,
            )
        if not is_last_rank:
            return IntermediateTensors(
                {"attn_res_stack": stack[: num_blocks + 1].transpose(0, 1).contiguous()}
            )
        if inputs is not None:
            out = inputs[0]
        else:
            assert isinstance(self.norm, Inkling2RMSNorm)
            out = self._mix(
                stack[:num_blocks],
                stack[num_blocks],
                self.attn_res.output_query,
                self.norm,
            )[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """The `inkling2_loop` checkpoints' architecture name; the model is
    `Inkling2ForConditionalGeneration` with an `Inkling2LoopModel`."""
