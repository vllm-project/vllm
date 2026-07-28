# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RIY: Runtime expert masking and statistics for MoE models.

Provides two capabilities:
1. Per-(layer, expert) activation statistics (frequency + weight magnitude)
2. Expert masking with weight renormalization

Statistics are accumulated without filtering — the operator decides
what to do with them via the admin API or offline tooling.
"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Literal

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RiyLayerStats:
    """Per-layer expert statistics accumulator.

    Tensors live on GPU to avoid CPU transfers in the hot path
    (which fail silently during CUDA Graph replay).
    """

    num_experts: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    # Token count per expert (how often selected by router)
    freq: torch.Tensor = field(init=False)
    # Sum of routing weights per expert (contribution magnitude)
    weight_sum: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.freq = torch.zeros(self.num_experts, dtype=torch.int64, device=self.device)
        self.weight_sum = torch.zeros(
            self.num_experts, dtype=torch.float32, device=self.device
        )

    def reset(self):
        # Replace tensors instead of in-place zero — safe from HTTP thread
        # (in-place .zero_() on GPU tensors from a non-CUDA thread crashes)
        self.freq = torch.zeros(self.num_experts, dtype=torch.int64, device=self.device)
        self.weight_sum = torch.zeros(
            self.num_experts, dtype=torch.float32, device=self.device
        )

    def ensure_device(self, device: torch.device):
        """Move tensors to device on first call from GPU."""
        if self.freq.device != device:
            self.freq = self.freq.to(device)
            self.weight_sum = self.weight_sum.to(device)
            self.device = device


class RiyState:
    """Global RIY state: statistics + expert mask.

    Thread-safe for admin API access. The hot path (apply_mask,
    record_stats) uses pre-computed tensors without locks.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        self._collecting = False
        self._num_layers = 0
        self._num_experts = 0
        self._layer_stats: list[RiyLayerStats] = []
        # Expert dimensions for VRAM estimation
        self._hidden_size = 0
        self._intermediate_size = 0
        self._quantization = ""
        # Pre-allocated GPU tensors for compiled stats (R2)
        # Addresses must be stable — used by @torch.compile'd function
        self._freq_pass: torch.Tensor | None = None  # (num_layers, num_experts)
        self._weight_pass: torch.Tensor | None = None  # (num_layers, num_experts)
        self._collecting_flag: torch.Tensor | None = None  # scalar, 0 or 1
        self._tensors_initialized = False

    def initialize(self, num_layers: int, num_experts: int):
        """Called once during model init."""
        with self._lock:
            self._num_layers = num_layers
            self._num_experts = num_experts
            self._layer_stats = [RiyLayerStats(num_experts) for _ in range(num_layers)]
            self._enabled = True
            logger.info(
                "RIY initialized: %d layers, %d experts/layer", num_layers, num_experts
            )

    def register_layer(
        self,
        layer_idx: int,
        num_experts: int,
        hidden_size: int = 0,
        intermediate_size: int = 0,
        quantization: str = "",
    ):
        """Register a MoE layer. Called from FusedMoE.__init__."""
        with self._lock:
            if hidden_size and not self._hidden_size:
                self._hidden_size = hidden_size
                self._intermediate_size = intermediate_size
                self._quantization = quantization
            if num_experts > self._num_experts:
                self._num_experts = num_experts
            if layer_idx >= self._num_layers:
                # Grow stats list
                while len(self._layer_stats) <= layer_idx:
                    self._layer_stats.append(RiyLayerStats(num_experts))
                self._num_layers = len(self._layer_stats)
            self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def collecting(self) -> bool:
        return self._collecting

    def initialize_tensors(self, device: torch.device, num_layers: int = 0):
        """Allocate pre-sized GPU tensors for compiled stats.

        Called from FusedMoE.__init__. Uses max(num_layers, self._num_layers)
        to ensure the tensors are large enough for all layers.
        Tensor addresses must remain stable for the @torch.compile'd graph.
        """
        if self._tensors_initialized:
            return
        with self._lock:
            if self._tensors_initialized:
                return
            n_layers = max(num_layers, self._num_layers)
            if n_layers == 0:
                return  # Not ready yet
            self._freq_pass = torch.zeros(
                n_layers, self._num_experts, dtype=torch.int64, device=device
            )
            self._weight_pass = torch.zeros(
                n_layers, self._num_experts, dtype=torch.float32, device=device
            )
            self._collecting_flag = torch.zeros((), dtype=torch.int32, device=device)
            self._tensors_initialized = True
            # Update num_layers if we got a better count
            if n_layers > self._num_layers:
                self._num_layers = n_layers
            logger.info(
                "RIY tensors allocated on %s: %d layers x %d experts",
                device,
                n_layers,
                self._num_experts,
            )

    def get_freq_view(self, layer_idx: int) -> torch.Tensor | None:
        """Get 1D freq slice for a layer (stable address for compiled graph)."""
        if self._freq_pass is not None and layer_idx < self._freq_pass.shape[0]:
            return self._freq_pass[layer_idx]
        return None

    def get_weight_view(self, layer_idx: int) -> torch.Tensor | None:
        """Get 1D weight_sum slice for a layer."""
        if self._weight_pass is not None and layer_idx < self._weight_pass.shape[0]:
            return self._weight_pass[layer_idx]
        return None

    def start_collection(self):
        with self._lock:
            self._collecting = True
            if self._collecting_flag is not None:
                self._collecting_flag.fill_(1)
            logger.info("RIY stats collection started")

    def stop_collection(self):
        with self._lock:
            self._collecting = False
            if self._collecting_flag is not None:
                self._collecting_flag.fill_(0)
            logger.info("RIY stats collection stopped")

    def reset_stats(self):
        with self._lock:
            # In-place zero — addresses must stay stable for compiled graph
            if self._freq_pass is not None:
                self._freq_pass.zero_()
            if self._weight_pass is not None:
                self._weight_pass.zero_()
            # Also reset legacy per-layer stats
            for s in self._layer_stats:
                s.reset()
            logger.info("RIY stats reset")

    def get_stats(self) -> dict:
        """Export raw statistics as dict."""
        with self._lock:
            layers = []
            for i in range(self._num_layers):
                try:
                    if self._freq_pass is not None:
                        assert self._weight_pass is not None
                        freq = self._freq_pass[i].detach().cpu().tolist()
                        wsum = self._weight_pass[i].detach().cpu().tolist()
                    else:
                        s = self._layer_stats[i]
                        freq = s.freq.detach().cpu().tolist()
                        wsum = s.weight_sum.detach().cpu().tolist()
                except Exception:
                    freq = [0] * self._num_experts
                    wsum = [0.0] * self._num_experts
                layers.append(
                    {
                        "layer": i,
                        "freq": freq,
                        "weight_sum": wsum,
                    }
                )
            return {
                "num_layers": self._num_layers,
                "num_experts": self._num_experts,
                "collecting": self._collecting,
                "layers": layers,
            }

    def on_forward(self):
        """Called on every MoE forward pass (Python-level, not in graph).

        Starts the HTTP server lazily in the real EngineCore worker process.
        """
        ensure_riy_server()

    def record_stats(
        self, layer_idx: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ):
        """Record activation stats for a layer. Called from hot path.

        Skips during CUDA Graph capture/replay — scatter_add_ on
        non-graph tensors would invalidate the capture.
        """
        if not self._collecting or layer_idx >= len(self._layer_stats):
            return
        stats = self._layer_stats[layer_idx]
        stats.ensure_device(topk_ids.device)
        # Frequency: count per expert (on GPU)
        ids_flat = topk_ids.flatten().long()
        stats.freq.scatter_add_(
            0, ids_flat, torch.ones_like(ids_flat, dtype=torch.int64)
        )
        # Weight magnitude: sum of routing weights per expert (on GPU)
        stats.weight_sum.scatter_add_(0, ids_flat, topk_weights.flatten().float())


RiyRoutingMode = Literal["pre_topk_mask", "post_topk_drop"]
_SUPPORTED_RIY_ROUTING_MODES = {"pre_topk_mask", "post_topk_drop"}


@dataclass(frozen=True)
class RiyProfile:
    """Validated RIY profile contract."""

    version: int
    routing_mode: RiyRoutingMode
    pruned_experts: frozenset[tuple[int, int]]
    num_layers: int
    num_experts: int
    model: str | None = None
    model_revision: str | None = None


@dataclass(frozen=True)
class RiyLayerPrunePlan:
    """Allocation and routing plan for one MoE layer."""

    routing_mode: RiyRoutingMode
    num_kept: int
    expert_filter: torch.Tensor
    expert_map: torch.Tensor
    pre_topk_logit_mask: torch.Tensor | None
    post_topk_drop_mask: torch.Tensor | None


_parsed_riy_profile_cache: dict[tuple[str, int, int], RiyProfile] = {}


def load_riy_profile(
    profile_path: str, num_layers: int, num_experts: int
) -> RiyProfile:
    """Load and validate a RIY profile before expert allocation."""
    cache_key = (profile_path, num_layers, num_experts)
    if cache_key in _parsed_riy_profile_cache:
        return _parsed_riy_profile_cache[cache_key]

    with open(profile_path) as file:
        raw_profile = json.load(file)
    if not isinstance(raw_profile, dict):
        raise ValueError("RIY profile must be a JSON object")

    version = raw_profile.get("version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError(f"Unsupported RIY profile version: {version!r}")

    raw_routing_mode = raw_profile.get("routing_mode")
    if version == 1:
        if raw_routing_mode is not None:
            raise ValueError("RIY version-1 profiles must not define routing_mode")
        routing_mode: RiyRoutingMode = "pre_topk_mask"
    else:
        if raw_routing_mode is None:
            raise ValueError("RIY version-2 profiles require routing_mode")
        if raw_routing_mode not in _SUPPORTED_RIY_ROUTING_MODES:
            raise ValueError(f"Unsupported RIY routing_mode: {raw_routing_mode!r}")
        routing_mode = raw_routing_mode

    raw_pruned_experts = raw_profile.get("pruned_experts")
    if not isinstance(raw_pruned_experts, list):
        raise ValueError("RIY profile must contain a pruned_experts list")

    pruned_experts: set[tuple[int, int]] = set()
    for entry in raw_pruned_experts:
        if not (
            isinstance(entry, list)
            and len(entry) == 2
            and all(type(value) is int for value in entry)
        ):
            raise ValueError("Each pruned_experts entry must be a [layer, expert] pair")
        layer_idx, expert_idx = entry
        if not 0 <= layer_idx < num_layers:
            raise ValueError(f"Layer index {layer_idx} is out of range")
        if not 0 <= expert_idx < num_experts:
            raise ValueError(f"Expert index {expert_idx} is out of range")
        pair = (layer_idx, expert_idx)
        if pair in pruned_experts:
            raise ValueError(f"Duplicate pruned expert entry: {list(pair)}")
        pruned_experts.add(pair)

    model = raw_profile.get("model")
    model_revision = raw_profile.get("model_revision")
    if model is not None and not isinstance(model, str):
        raise ValueError("RIY profile model must be a string")
    if model_revision is not None and not isinstance(model_revision, str):
        raise ValueError("RIY profile model_revision must be a string")

    profile = RiyProfile(
        version=version,
        routing_mode=routing_mode,
        pruned_experts=frozenset(pruned_experts),
        num_layers=num_layers,
        num_experts=num_experts,
        model=model,
        model_revision=model_revision,
    )
    _parsed_riy_profile_cache[cache_key] = profile
    return profile


def build_riy_layer_prune_plan(
    profile: RiyProfile,
    layer_idx: int,
    top_k: int,
) -> RiyLayerPrunePlan:
    """Build a mode-explicit allocation and routing plan for one layer."""
    if not 0 <= layer_idx < profile.num_layers:
        raise ValueError(f"Layer index {layer_idx} is out of range")
    num_experts = profile.num_experts
    pruned_ids = {
        expert_idx
        for entry_layer, expert_idx in profile.pruned_experts
        if entry_layer == layer_idx
    }
    expert_filter = torch.ones(num_experts, dtype=torch.bool)
    if pruned_ids:
        expert_filter[list(pruned_ids)] = False
    num_kept = int(expert_filter.sum().item())
    if num_kept == 0:
        raise ValueError(f"RIY profile prunes every expert in layer {layer_idx}")
    if profile.routing_mode == "pre_topk_mask" and num_kept < top_k:
        raise ValueError(
            f"RIY profile keeps {num_kept} experts in layer {layer_idx}, "
            f"but top_k is {top_k}"
        )

    expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
    expert_map[expert_filter] = torch.arange(num_kept, dtype=torch.int32)
    pre_topk_logit_mask = None
    post_topk_drop_mask = None
    if profile.routing_mode == "pre_topk_mask":
        pre_topk_logit_mask = torch.zeros(num_experts, dtype=torch.float32)
        pre_topk_logit_mask[~expert_filter] = float("-inf")
    else:
        post_topk_drop_mask = ~expert_filter

    return RiyLayerPrunePlan(
        routing_mode=profile.routing_mode,
        num_kept=num_kept,
        expert_filter=expert_filter,
        expert_map=expert_map,
        pre_topk_logit_mask=pre_topk_logit_mask,
        post_topk_drop_mask=post_topk_drop_mask,
    )


_riy_profile_cache: dict[str, dict] = {}


def _load_riy_profile(profile_path: str) -> dict:
    """Load and cache RIY profile."""
    if profile_path not in _riy_profile_cache:
        with open(profile_path) as file:
            _riy_profile_cache[profile_path] = json.load(file)
    return _riy_profile_cache[profile_path]


def build_riy_prune_map(
    layer_idx: int,
    original_num_experts: int,
    profile_path: str,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Build per-layer expert map from RIY profile.

    Each MoE layer gets its own map — different experts can be pruned
    in different layers. This is the correct approach because each
    expert in each layer is a unique FFN with unique weights.

    Args:
        layer_idx: The layer index in the model
        original_num_experts: Total experts in the original model
        profile_path: Path to RIY profile JSON

    Returns:
        (num_kept, expert_map, logit_mask)
        - num_kept: number of kept experts for this layer
        - expert_map: (original_num_experts,) int32, -1 for pruned
        - logit_mask: (original_num_experts,) float32, 0.0 kept / -inf pruned
    """
    profile = _load_riy_profile(profile_path)

    pruned_experts = profile.get("pruned_experts")
    if not isinstance(pruned_experts, list):
        raise ValueError("RIY profile must contain a pruned_experts list")

    pruned_ids: set[int] = set()
    for entry in pruned_experts:
        if not (
            isinstance(entry, list)
            and len(entry) == 2
            and all(isinstance(value, int) for value in entry)
        ):
            raise ValueError("Each pruned_experts entry must be a [layer, expert] pair")
        entry_layer, expert_idx = entry
        if entry_layer < 0 or expert_idx < 0:
            raise ValueError("Layer and expert indices must be non-negative")
        if expert_idx >= original_num_experts:
            raise ValueError(
                f"Expert index {expert_idx} is out of range for "
                f"{original_num_experts} experts"
            )
        if entry_layer == layer_idx:
            pruned_ids.add(expert_idx)

    expert_map = torch.full((original_num_experts,), -1, dtype=torch.int32)
    logit_mask = torch.zeros(original_num_experts, dtype=torch.float32)
    compact_idx = 0
    for i in range(original_num_experts):
        if i not in pruned_ids:
            expert_map[i] = compact_idx
            compact_idx += 1
        else:
            logit_mask[i] = float("-inf")

    logger.info(
        "RIY layer %d: %d/%d experts kept (%d pruned)",
        layer_idx,
        compact_idx,
        original_num_experts,
        len(pruned_ids),
    )
    return compact_idx, expert_map, logit_mask


# Global singleton
_riy_state = RiyState()


def get_riy_state() -> RiyState:
    return _riy_state


# ── Standalone HTTP server (runs in EngineCore process) ───────────────────────


def _start_riy_server(host: str = "127.0.0.1", port: int = 8019):
    """Start a minimal HTTP server for RIY statistics.

    Runs in a daemon thread inside the EngineCore worker process, so it
    has direct access to the RiyState singleton (same process, same memory).
    The main vLLM API server runs on port 8011; this runs on a separate
    port to avoid any interference.

    Must be started from on_forward() (not register_layer), because
    register_layer runs in the parent process that forks and dies.
    """
    import json as _json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class RiyHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # silence per-request logs

        def _json_response(self, data, status=200):
            body = _json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            riy = get_riy_state()
            if self.path == "/riy/stats":
                if not riy.enabled:
                    self._json_response({"error": "not initialized"}, 503)
                else:
                    self._json_response(riy.get_stats())
            elif self.path == "/riy/health":
                self._json_response(
                    {
                        "enabled": riy.enabled,
                        "collecting": riy.collecting,
                        "num_layers": riy._num_layers,
                        "num_experts": riy._num_experts,
                        "hidden_size": riy._hidden_size,
                        "intermediate_size": riy._intermediate_size,
                        "quantization": riy._quantization,
                    }
                )
            else:
                self._json_response({"error": "not found"}, 404)

        def do_POST(self):
            riy = get_riy_state()
            if self.path == "/riy/stats/start":
                riy.start_collection()
                self._json_response({"status": "collecting"})
            elif self.path == "/riy/stats/stop":
                riy.stop_collection()
                self._json_response({"status": "stopped"})
            elif self.path == "/riy/stats/reset":
                riy.reset_stats()
                self._json_response({"status": "reset"})
            else:
                self._json_response({"error": "not found"}, 404)

    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        server = ReusableHTTPServer((host, port), RiyHandler)
        logger.info(
            "RIY HTTP server started on %s:%d (pid=%d)",
            host,
            port,
            os.getpid(),
        )
        server.serve_forever()
    except OSError as e:
        logger.warning("RIY HTTP server failed to start on port %d: %s", port, e)


_riy_server_started = False
_riy_server_lock = threading.Lock()


def ensure_riy_server(port: int = 8019):
    """Start RIY HTTP server once (idempotent).

    Also auto-starts stats collection if not already collecting.
    """
    global _riy_server_started
    if _riy_server_started:
        return
    with _riy_server_lock:
        if _riy_server_started:
            return
        _riy_server_started = True
        # Auto-start collection if monitor is active but not yet collecting
        riy = get_riy_state()
        if not riy._collecting:
            riy.start_collection()
        host = os.environ.get("VLLM_RIY_HOST", "127.0.0.1")
        t = threading.Thread(target=_start_riy_server, args=(host, port), daemon=True)
        t.start()
