# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Balanced MoE routing for benchmarks with dummy weights.

Dummy gate weights give degenerate routing, which skews MoE and all-to-all
cost. The routing still runs; its result is replaced by the fixture at
whichever boundary the layer exposes:

- Gate output: fixed logits whose top-k is the fixture. Covers every backend
  that routes from logits, including monolithic kernels that run top-k
  internally (e.g. FlashInfer TRTLLM).
- Experts input: models that compute top-k themselves and pass
  `topk_ids` to their experts (e.g. MegaMoE with a fused gate) get the
  fixture's IDs. The routing weights are kept; they don't affect cost.

Hash-routed layers ignore logits and keep their tables unless their experts
take `topk_ids`.
"""

import inspect

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

logger = init_logger(__name__)

# Fixture logits: chosen experts score far above the rest under any monotonic
# scoring function, so top-k picks exactly them.
_CHOSEN_LOGIT = 4.0
_OTHER_LOGIT = -4.0


def balanced_topk_ids(
    num_tokens: int, num_experts: int, top_k: int, ep_size: int, rng
) -> np.ndarray:
    """Expert IDs [num_tokens, top_k] giving every EP rank equal load.

    Each token's slots go round-robin over EP ranks, to distinct experts.
    """
    assert num_experts % ep_size == 0
    local = num_experts // ep_size
    slots = np.arange(num_tokens)[:, None] * top_k + np.arange(top_k)
    ranks = slots % ep_size
    # Slots of one token that land on the same rank take consecutive experts.
    repeat = np.zeros_like(ranks)
    for j in range(1, top_k):
        repeat[:, j] = (ranks[:, :j] == ranks[:, j : j + 1]).sum(axis=1)
    assert repeat.max() < local
    base = rng.integers(local, size=(num_tokens, ep_size))
    offset = np.take_along_axis(base, ranks, axis=1)
    return ranks * local + (offset + repeat) % local


def _moe_layers(model: torch.nn.Module) -> list[tuple[torch.nn.Module, GateLinear]]:
    """(MoE module, gate) per layer, in model order."""
    layers: dict[GateLinear, torch.nn.Module] = {}
    for module in model.modules():
        gate = getattr(module, "gate", None)
        if not isinstance(gate, GateLinear):
            gate = getattr(module, "router", None)
        if not isinstance(gate, GateLinear) or gate in layers:
            continue
        if _top_k(module) is not None:
            layers[gate] = module
    return [(module, gate) for gate, module in layers.items()]


def _top_k(module: torch.nn.Module) -> int | None:
    """The MoE layer's top-k, from the module or its experts submodules."""
    for sub in module.modules():
        for name in ("top_k", "n_activated_experts"):
            top_k = getattr(sub, name, None)
            if isinstance(top_k, int):
                return top_k
    return None


class BalancedRouting:
    """Installs balanced routing for one case; `remove()` restores the model."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_tokens: int,
        dp_rank: int,
        dp_size: int,
        ep_size: int,
        seed: int,
        device: torch.device,
    ):
        layers = _moe_layers(model)
        if not layers:
            raise ValueError("No MoE gates found; use --routing real.")
        self._hooks = []
        self._biases: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._fired = [0] * len(layers)
        self._layers = layers
        # Per-rank rows, with headroom for padding; every rank generates the
        # same global fixture and takes its slice.
        rows = -(-num_tokens // 256) * 256 + 256
        for layer, (module, gate) in enumerate(layers):
            if (getattr(module, "num_expert_group", None) or 1) > 1:
                raise NotImplementedError("Balanced routing with grouped top-k.")
            num_experts = gate.weight.shape[0]
            top_k = _top_k(module)
            assert top_k is not None
            rng = np.random.default_rng([seed, layer])
            ids = balanced_topk_ids(rows * dp_size, num_experts, top_k, ep_size, rng)[
                dp_rank * rows : (dp_rank + 1) * rows
            ]
            logits = torch.full((rows, num_experts), _OTHER_LOGIT, dtype=torch.float32)
            logits.scatter_(1, torch.from_numpy(ids), _CHOSEN_LOGIT)
            # Distinct values: no ties for any top-k implementation.
            logits += torch.from_numpy(rng.uniform(0, 0.5, logits.shape)).float()
            self._hooks.append(
                gate.register_forward_hook(
                    self._substitute(logits.to(device), self._fired, layer)
                )
            )
            bias = getattr(gate, "e_score_correction_bias", None)
            if bias is None:
                bias = getattr(module, "e_score_correction_bias", None)
            if bias is None:
                bias = getattr(module, "expert_bias", None)
            if isinstance(bias, torch.Tensor):
                self._biases.append((bias, bias.detach().clone()))
                bias.data.zero_()
            experts = getattr(module, "experts", None)
            if isinstance(experts, torch.nn.Module):
                params = list(inspect.signature(experts.forward).parameters)
                if "topk_ids" in params:
                    self._hooks.append(
                        experts.register_forward_pre_hook(
                            self._substitute_ids(
                                torch.from_numpy(ids).to(device),
                                params.index("topk_ids"),
                                self._fired,
                                layer,
                            ),
                            with_kwargs=True,
                        )
                    )

    @staticmethod
    def _substitute(fixture: torch.Tensor, fired: list[int], layer: int):
        cast = [fixture]

        def hook(module, inputs, output):
            fired[layer] += 1
            logits = output[0] if isinstance(output, tuple) else output
            if cast[0].dtype != logits.dtype:
                cast[0] = fixture.to(logits.dtype)  # Once, at warmup.
            replaced = cast[0][: logits.shape[0]]
            return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

        return hook

    @staticmethod
    def _substitute_ids(fixture: torch.Tensor, index: int, fired, layer: int):
        cast = [fixture]

        def hook(module, args, kwargs):
            fired[layer] += 1
            ids = kwargs["topk_ids"] if "topk_ids" in kwargs else args[index]
            if cast[0].dtype != ids.dtype:
                cast[0] = fixture.to(ids.dtype)  # Once, at warmup.
            replaced = cast[0][: ids.shape[0]]
            if "topk_ids" in kwargs:
                return args, {**kwargs, "topk_ids": replaced}
            return (*args[:index], replaced, *args[index + 1 :]), kwargs

        return hook

    def check_applied(self) -> None:
        """Fail if a layer bypassed both the gate and the experts hooks."""
        missed = [
            type(module).__name__
            for (module, _), fired in zip(self._layers, self._fired)
            if not fired
        ]
        if missed:
            raise RuntimeError(
                f"Balanced routing didn't reach {len(missed)} MoE layers "
                f"({missed[0]}): their gate is fused away. Use --routing real."
            )

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        for bias, saved in self._biases:
            bias.data.copy_(saved)


def install_routing(
    runner, requested: str, dummy_weights: bool, num_tokens: int, seed: int
) -> tuple[str, BalancedRouting | None]:
    """Resolve `--routing` and install balanced routing if selected.

    `auto` is balanced with dummy weights and real otherwise; it falls back to
    real routing when the model doesn't support the fixture.
    """
    mode = requested
    if mode == "auto":
        mode = "balanced" if dummy_weights else "real"
    if mode == "real":
        return mode, None
    parallel = runner.parallel_config
    ep_size = (
        parallel.data_parallel_size
        * parallel.tensor_parallel_size
        * parallel.prefill_context_parallel_size
        if parallel.enable_expert_parallel
        else 1
    )
    # Without EP, MoE gathers PCP rows, including replicated decode queries.
    if not parallel.enable_expert_parallel:
        num_tokens *= parallel.prefill_context_parallel_size
    try:
        routing = BalancedRouting(
            runner.model,
            num_tokens,
            runner.dp_rank,
            runner.dp_size,
            ep_size,
            seed,
            runner.device,
        )
    except (ValueError, NotImplementedError) as e:
        if requested == "balanced":
            raise
        logger.warning("Using real routing: %s", e)
        return "real", None
    return mode, routing
