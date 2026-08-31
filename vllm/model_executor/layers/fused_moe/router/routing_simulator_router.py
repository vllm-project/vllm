# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.worker.ubatching import dbo_enabled

logger = init_logger(__name__)

# Cap for the per-step batched RNG fast path: decode/verify-sized batches
# only. Prefill-sized batches keep the per-call path (bounded memory; the
# decode FULL cudagraph is the target).
_STEP_BATCH_MAX_TOKENS = 64


def _is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        # CPU-only torch raises from the C binding rather than return False.
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


class RoutingStrategy(ABC):
    """Base class for token-to-expert routing strategies."""

    @abstractmethod
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        pass


class DistributionBasedRouting(RoutingStrategy):
    """
    Distribution-based random routing strategy with configurable distributions.

    This routing strategy randomly selects experts for each token based on
    different probability distributions. Currently supports uniform and normal
    distributions for testing different routing patterns.
    """

    def __init__(self, distribution: str = "uniform", **distribution_params: Any):
        """
        Initialize distribution-based routing.

        Args:
            distribution: Type of distribution to use for sampling
                - "uniform": Uniform distribution (default)
                - "normal": Normal/Gaussian distribution
            **distribution_params: Parameters specific to the
                chosen distribution
                For "uniform": No additional parameters needed
                For "normal": mean (default: 0.0), std (default: 1.0)
        """
        self.distribution = distribution.lower()
        self.distribution_params = distribution_params

        # Validate distribution and parameters
        self._validate_distribution_params()

        # Per-forward-pass batched RNG state ("uniform" only). One strategy
        # instance is shared by every MoE layer, so one rand+topk per pass
        # can serve all layers via slicing instead of one rand+topk each.
        #
        # Invariant preserved (vs. the per-call path):
        # (i) for a fixed torch generator state at the start of a forward
        #     pass, every layer's (topk_weights, topk_ids) is a bitwise-
        #     deterministic function of the pass's call shapes;
        # (ii) each token row remains an iid uniform top-k draw without
        #     replacement, rows independent across layers and tokens;
        # (iii) topk_weights are all-ones float32 [num_tokens, top_k],
        #     bitwise identical to the per-call path.
        # NOT preserved: per-layer value identity with the per-call path
        # (the generator is consumed in one batched call per pass).
        self.step_batched_rng_enabled = True
        self.step_batch_stats: dict[str, int] = {
            "batched_calls": 0,
            "fallback_calls": 0,
        }
        self._step_ctx: object | None = None
        self._step_key: tuple[int, int, int] | None = None
        self._step_calls = 0
        self._step_clean = False
        self._step_ids: torch.Tensor | None = None
        self._served_from_step_batch = False
        # (num_experts, top_k) -> routed-layer calls observed per pass.
        self._learned_layers: dict[tuple[int, int], int] = {}
        # Never evicted: entries may be baked into captured CUDA graphs,
        # so their storage must stay alive. Bounded by decode shapes
        # (num_tokens <= _STEP_BATCH_MAX_TOKENS).
        self._ones_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        # Keys whose ones fill has so far only been RECORDED inside CUDA
        # graph capture (never executed): the buffer holds arbitrary
        # graph-pool memory until the creating graph first replays, so
        # every other reader must fill it before use. Cleared by the
        # first executed (eager) fill.
        self._ones_pending_fill: set[tuple[int, int, torch.device]] = set()
        # Per-forward-pass dedup so one capture records at most one fill
        # per key (reset in _begin_step).
        self._step_ones_filled: set[tuple[int, int, torch.device]] = set()

    def _validate_distribution_params(self):
        """Validate distribution type and parameters."""
        valid_distributions = ["uniform", "normal"]

        if self.distribution not in valid_distributions:
            raise ValueError(
                f"Unsupported distribution: {self.distribution}. "
                f"Supported distributions: {valid_distributions}"
            )

        # Set default parameters if not provided
        if self.distribution == "normal":
            self.distribution_params.setdefault("mean", 0.0)
            self.distribution_params.setdefault("std", 1.0)

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select experts for each token using the specified distribution.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids) where:
            - topk_weights: Weights based on distribution sampling
            - topk_ids: Expert indices sampled from the distribution
        """
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]

        if indices_type is None:
            indices_type = torch.long

        self._served_from_step_batch = False

        # Generate expert IDs based on the specified distribution
        topk_ids = self._sample_expert_ids(
            num_tokens, num_experts, top_k, hidden_states.device, indices_type
        )

        # Generate weights based on the distribution
        topk_weights = self._generate_weights(num_tokens, top_k, hidden_states.device)

        return topk_weights, topk_ids

    def _sample_expert_ids(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        indices_type: torch.dtype,
    ) -> torch.Tensor:
        """Sample expert IDs based on the specified distribution."""

        if self.distribution == "uniform":
            topk_ids = self._sample_uniform_step_batched(
                num_tokens, num_experts, top_k, device, indices_type
            )
            if topk_ids is not None:
                return topk_ids
            # Generate random scores, and take the top-k to avoid duplicate topk_ids
            scores = torch.rand(num_tokens, num_experts, device=device)
            _, topk_ids = torch.topk(scores, top_k, dim=-1)
            return topk_ids.to(indices_type)

        elif self.distribution == "normal":
            # For normal distribution, sample continuous values and map to
            # expert IDs
            continuous_samples = self._sample_continuous_distribution(
                num_tokens, top_k, device
            )

            # Map continuous samples to expert indices
            # Normalize to [0, 1] range and scale to [0, num_experts)
            normalized_samples = self._normalize_samples(continuous_samples)
            expert_ids = (normalized_samples * num_experts).long()
            expert_ids = torch.clamp(expert_ids, 0, num_experts - 1)

            return expert_ids.to(dtype=indices_type)

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _begin_step(self, ctx: object, step_key: tuple[int, int, int]) -> None:
        """Start tracking a new forward pass (fresh ForwardContext object)."""
        # Learn the routed-layer call count from the pass that just
        # completed, but only if it was shape-homogeneous.
        if self._step_clean and self._step_calls > 0 and self._step_key is not None:
            self._learned_layers[self._step_key[1:]] = self._step_calls
        self._step_ctx = ctx
        self._step_key = step_key
        self._step_calls = 0
        self._step_clean = True
        self._step_ids = None
        self._step_ones_filled.clear()

    def _sample_uniform_step_batched(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        indices_type: torch.dtype,
    ) -> torch.Tensor | None:
        """Serve uniform expert ids from one per-pass batched rand+topk.

        Generates [num_layers, num_tokens, top_k] ids on the first routed
        layer of a forward pass and hands each subsequent layer a slice
        (a view: no extra kernels). Inside CUDA graph capture the batched
        rand+topk is captured once; graph-safe Philox makes replays draw
        fresh values, exactly as the per-call path did.

        Returns None whenever the per-call path must be used (fail-closed):
        no forward context, dynamo tracing, DBO microbatching, batch larger
        than _STEP_BATCH_MAX_TOKENS, unknown layer count, or any shape/dtype
        mismatch with the pass's batch.
        """
        if (
            not self.step_batched_rng_enabled
            or torch.compiler.is_compiling()
            or not is_forward_context_available()
        ):
            return None
        ctx = get_forward_context()
        step_key = (num_tokens, num_experts, top_k)
        if ctx is not self._step_ctx:
            self._begin_step(ctx, step_key)
        if step_key != self._step_key:
            # Heterogeneous routing shapes inside one pass: serve per-call
            # and never learn a layer count from this pass.
            self._step_clean = False
            self.step_batch_stats["fallback_calls"] += 1
            return None
        call_idx = self._step_calls
        self._step_calls += 1
        if dbo_enabled():
            # DBO microbatching interleaves layer calls across threads.
            # dbo_enabled() is True exactly inside ubatch threads; the
            # per-ubatch ForwardContexts are created without ubatch_slices,
            # so the context attribute cannot detect DBO here. Never learn
            # a layer count from such a pass.
            self._step_clean = False
            self.step_batch_stats["fallback_calls"] += 1
            return None
        if num_tokens > _STEP_BATCH_MAX_TOKENS:
            self.step_batch_stats["fallback_calls"] += 1
            return None
        if call_idx == 0:
            num_layers = self._learned_layers.get((num_experts, top_k), 0)
            if num_layers > 0:
                scores = torch.rand(num_layers * num_tokens, num_experts, device=device)
                ids = torch.topk(scores, top_k, dim=-1)[1].to(indices_type)
                self._step_ids = ids.view(num_layers, num_tokens, top_k)
        step_ids = self._step_ids
        if (
            step_ids is None
            or step_ids.dtype != indices_type
            or step_ids.shape[1] != num_tokens
        ):
            self.step_batch_stats["fallback_calls"] += 1
            return None
        if call_idx >= step_ids.shape[0]:
            # Learned layer count too small; relearn from this pass.
            self._learned_layers.pop((num_experts, top_k), None)
            self._step_ids = None
            self.step_batch_stats["fallback_calls"] += 1
            return None
        self._served_from_step_batch = True
        self.step_batch_stats["batched_calls"] += 1
        logger.info_once(
            "Routing simulator per-step batched RNG engaged "
            "(num_tokens=%d, layers=%d, top_k=%d).",
            num_tokens,
            step_ids.shape[0],
            top_k,
        )
        return step_ids[call_idx]

    def _get_cached_ones(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        key = (num_tokens, top_k, device)
        capturing = _is_current_stream_capturing()
        ones = self._ones_cache.get(key)
        if ones is None:
            ones = torch.ones((num_tokens, top_k), dtype=torch.float32, device=device)
            self._ones_cache[key] = ones
            if capturing:
                # Under capture the fill above was only RECORDED into the
                # capturing graph, not executed: the buffer stays
                # uninitialized graph-pool memory until that graph first
                # replays. Track it so every other reader fills it before
                # use (all fills write the identical all-ones payload, so
                # redundant fills are always safe).
                self._ones_pending_fill.add(key)
                self._step_ones_filled.add(key)
        elif key in self._ones_pending_fill:
            if not capturing:
                # Executed now: the buffer is initialized for good.
                ones.fill_(1.0)
                self._ones_pending_fill.discard(key)
            elif key not in self._step_ones_filled:
                # A different capture is baking this address without a
                # fill of its own: record one so its replays never depend
                # on the creating graph having replayed first.
                ones.fill_(1.0)
                self._step_ones_filled.add(key)
        return ones

    def _sample_continuous_distribution(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        """Sample from continuous distributions."""
        shape = (num_tokens, top_k)

        if self.distribution == "normal":
            mean = self.distribution_params["mean"]
            std = self.distribution_params["std"]
            return torch.normal(mean, std, size=shape, device=device)

        else:
            raise ValueError(
                f"Unsupported continuous distribution: {self.distribution}"
            )

    def _normalize_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Normalize samples to [0, 1] range."""
        if self.distribution == "normal":
            # Use sigmoid to map normal distribution to [0, 1]
            return torch.sigmoid(samples)

        else:
            raise ValueError(
                f"Unsupported distribution for normalization: {self.distribution}"
            )

    def _generate_weights(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        """Generate weights based on the distribution."""
        if self.distribution == "uniform":
            if self._served_from_step_batch:
                # One cached all-ones buffer shared across the step's
                # layers instead of a fresh fill kernel per layer.
                return self._get_cached_ones(num_tokens, top_k, device)
            # All-ones weights for uniform distribution
            return torch.ones(
                (num_tokens, top_k),
                dtype=torch.float32,
                device=device,
            )

        elif self.distribution == "normal":
            # For normal distribution, generate weights from the same
            # distribution
            continuous_weights = self._sample_continuous_distribution(
                num_tokens, top_k, device
            )
            # Normalize to positive values and sum to 1
            weights = torch.abs(continuous_weights)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            return weights

        else:
            raise ValueError(
                f"Unsupported distribution for weight generation: {self.distribution}"
            )

    def get_distribution_info(self) -> dict:
        """Get information about the current distribution configuration."""
        return {
            "distribution": self.distribution,
            "parameters": self.distribution_params.copy(),
        }


class RoutingSimulator:
    """
    Token-to-Expert Routing Simulator.

    This class provides a framework for testing and comparing different
    routing strategies for MoE models. It can simulate routing behavior
    and collect statistics for analysis.
    """

    # Class-level registry of routing strategies
    _routing_strategies: dict[str, RoutingStrategy] = {
        # Basic routing strategies
        "uniform_random": DistributionBasedRouting(
            distribution="uniform", mean=0.0, std=1.0
        ),
        "normal_routing": DistributionBasedRouting(
            distribution="normal", mean=0.0, std=1.0
        ),
    }

    @classmethod
    def register_strategy(cls, name: str, strategy: RoutingStrategy):
        """
        Register a custom routing strategy.

        Args:
            name: Name of the strategy
            strategy: RoutingStrategy instance
        """
        cls._routing_strategies[name] = strategy

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """
        Get list of available routing strategy names.

        Returns:
            List of available strategy names
        """
        return list(cls._routing_strategies.keys())

    @staticmethod
    def simulate_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        strategy_name: str,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate token-to-expert routing using the specified strategy.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            strategy_name: Name of the routing strategy to use
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        if strategy_name not in RoutingSimulator._routing_strategies:
            raise ValueError(
                f"Unknown routing strategy: {strategy_name}. "
                f"Available strategies: "
                f"{list(RoutingSimulator._routing_strategies.keys())}"
            )
        logger.warning_once(
            "Simulating MoE routing using a %s strategy. "
            "This should only be used for performance testing. "
            "Model outputs will not be valid.",
            strategy_name,
        )

        strategy = RoutingSimulator._routing_strategies[strategy_name]
        return strategy.route_tokens(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            indices_type=indices_type,
        )


class RoutingSimulatorRouter(BaseRouter):
    """Router that uses routing simulation strategies for testing/debugging."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Simulated

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use routing simulator to compute routing."""
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=routing_strategy,
            top_k=self.top_k,
            indices_type=indices_type,
        )
        return topk_weights, topk_ids
