# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HybridAdapterManager — dispatches adapter operations across a
Tokenformer sub-manager and a LoRA sub-manager.

Current state (phase 2 skeleton): the Tokenformer sub-manager is real;
the LoRA sub-manager is a placeholder. `add_adapter` classifies the
incoming `.pt` file and, if it's pure Tokenformer, delegates. Pure-LoRA
and hybrid adapters raise NotImplementedError until the LoRA-from-.pt
loader lands (option C in the rollout plan).

Once option C is in, this class will:
 1. Split the loaded state dict via split_adapter_state_dict.
 2. Register the Tokenformer tensors with TokenformerModelManager.
 3. Register the LoRA tensors with the LoRA worker manager.
 4. At `set_active_adapters` time, fan out to both sub-managers.

See `docs/design/hybrid_lora_tokenformer.md`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.tokenformer.adapter_format import (
    AdapterKind,
    load_adapter_from_pt,
    normalize_lora_state_dict,
)
from vllm.tokenformer.lora_from_pt import load_lora_model_from_pt
from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from vllm.config import VllmConfig
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


class PTWorkerLoRAManager(LRUCacheWorkerLoRAManager):
    """LoRA worker manager that prefers ScalarLM's `.pt` adapter format
    but falls back to upstream HF PEFT (`adapter_config.json +
    adapter_model.safetensors`) when no `.pt` is in the adapter
    directory.

    Overrides `_load_adapter` only. Slot management, kernel setup, and
    dummy-lora caching are inherited.
    """

    def _load_adapter(self, lora_request: "LoRARequest") -> "LoRAModel":
        # Prefer `.pt`. If there's no .pt in the dir, delegate to the
        # upstream PEFT path so users with standard HF LoRA checkpoints
        # keep working on the same server.
        from pathlib import Path
        from vllm.lora.utils import get_adapter_absolute_path

        lora_path = get_adapter_absolute_path(lora_request.lora_path)
        pt_files = list(Path(lora_path).glob("*.pt"))
        if not pt_files:
            logger.info(
                "No .pt file in %s; falling back to upstream PEFT loader.",
                lora_path,
            )
            return super()._load_adapter(lora_request)

        loaded = load_adapter_from_pt(lora_path)
        if not loaded.lora_sd:
            raise ValueError(
                f"Adapter at {loaded.source_path} has no LoRA tensors "
                f"(found only Tokenformer keys). Serve with "
                f"--enable-tokenformer instead, or as a hybrid adapter "
                f"with both --enable-lora and --enable-tokenformer."
            )
        # Re-normalize the lora_sd using the live model's prefix so the
        # same .pt file works across model families (e.g. Qwen3-4B uses
        # `model.layers.*` while Qwen3.5-4B uses
        # `language_model.model.layers.*`).
        lora_sd = self._renormalize_lora_sd_for_model(loaded.lora_sd)
        lora_model = load_lora_model_from_pt(
            lora_sd,
            lora_model_id=lora_request.adapter_id,
            device=self.device,
            dtype=(
                self.lora_config.lora_dtype
                if self.lora_config is not None
                else None
            ),
            model_vocab_size=self.vocab_size,
            metadata=loaded.metadata,
            max_lora_rank=(
                self.lora_config.max_lora_rank
                if self.lora_config is not None
                else None
            ),
        )
        self._warn_on_zero_base_match(lora_model, loaded.source_path)
        return lora_model

    def _detect_model_layers_prefix(self) -> str:
        """Probe the live base model to find which prefix its decoder
        layers use.  Returns the string that should replace the
        training-side ``model.layers.`` prefix in LoRA keys.

        Known variants:
          - ``model.layers.``            — standard decoder-only (Qwen3)
          - ``language_model.model.layers.`` — VL-wrapped decoder
                                               (Qwen3.5, Gemma4)

        Falls back to ``model.layers.`` (the vLLM default) if the
        model tree is not accessible.
        """
        try:
            model = self._adapter_manager.model
            for name, _ in model.named_modules():
                if name.endswith(".self_attn") or name.endswith(".mlp"):
                    # Strip the layer suffix to get the layers container
                    # e.g. "language_model.model.layers.0.self_attn"
                    #   -> "language_model.model.layers."
                    # or  "model.layers.0.self_attn"
                    #   -> "model.layers."
                    parts = name.split(".")
                    try:
                        layers_idx = next(
                            i for i, p in enumerate(parts) if p == "layers"
                        )
                        prefix = ".".join(parts[: layers_idx + 1]) + "."
                        logger.debug(
                            "Detected model layers prefix: %s", prefix
                        )
                        return prefix
                    except StopIteration:
                        continue
        except Exception:
            pass
        logger.debug(
            "Could not detect model layers prefix; defaulting to "
            "model.layers."
        )
        return "model.layers."

    def _renormalize_lora_sd_for_model(
        self, lora_sd: dict
    ) -> dict:
        """Re-run key normalization using the live model's prefix.

        ``adapter_format.normalize_lora_key`` uses a static rule that
        maps ``model.layers.*`` to ``language_model.model.layers.*``
        (needed for Qwen3.5/Gemma4).  For models whose vLLM tree really
        does use ``model.layers.*`` (e.g. Qwen3-4B-Instruct) that
        mapping is wrong.

        This method detects the correct target prefix from the live
        model, then re-applies *only* the structural part of the
        normalization (prefix swap + PEFT ``.default.`` strip +
        ClippableLinear ``.linear.`` strip) with the right target.
        """
        target_prefix = self._detect_model_layers_prefix()
        # target_prefix is e.g. "model.layers." or
        # "language_model.model.layers."

        out: dict = {}
        for k, v in lora_sd.items():
            # The key has already been through normalize_lora_key once
            # (inside load_adapter_from_pt -> split_adapter_state_dict).
            # That pass may have mapped it to the wrong prefix.  We
            # undo the structural prefix and re-apply with the correct
            # target.

            # Undo any previous prefix normalization: if the key starts
            # with a known vLLM prefix that isn't the target, strip it
            # back to bare "layers.*" then re-prefix.
            KNOWN_PREFIXES = (
                "language_model.model.layers.",
                "model.layers.",
            )
            bare = k
            for kp in KNOWN_PREFIXES:
                if k.startswith(kp):
                    bare = "layers." + k[len(kp):]
                    break

            # Re-prefix with the correct target
            if bare.startswith("layers."):
                nk = target_prefix + bare[len("layers."):]
            else:
                nk = k  # non-layers key (vision_tower, embed_vision, …)

            if nk in out:
                raise ValueError(
                    f"LoRA key re-normalization collision: {k!r} and an "
                    f"earlier key both map to {nk!r}."
                )
            out[nk] = v
        return out

    def _warn_on_zero_base_match(self, lora_model, source_path) -> None:
        """If every parsed LoRA module path is missing from the base
        model, the adapter will silently no-op at activation time (see
        `LoRAModelManager.activate_adapter` — `module_lora` lookup
        returns None and the slot is reset). That's the hardest
        failure to debug because the adapter still "loads". Turn it
        into a visible WARNING.

        We're permissive: a single match is enough to silence the
        warning. Partial matches (e.g. vision-tower keys landing on
        modules we don't support) are expected and benign.
        """
        try:
            base_modules = set(
                n for n, _ in self._adapter_manager.model.named_modules()
            )
        except Exception:
            # If we can't read the tree (e.g. _adapter_manager not yet
            # wired during some tests), skip the check silently.
            return

        lora_modules = set(lora_model.loras.keys()) if hasattr(
            lora_model, "loras") else set()
        if not lora_modules:
            return

        matches = lora_modules & base_modules
        if matches:
            return

        # Zero overlap — classic prefix/rename mismatch. Show the
        # caller a sample from each side so they can eyeball the
        # transform that's missing.
        sample_lora = sorted(lora_modules)[:3]
        sample_base = sorted(
            m for m in base_modules if "self_attn" in m or "mlp" in m
        )[:3]
        logger.warning(
            "LoRA adapter at %s loaded but NONE of its %d module paths "
            "match the base model. The adapter will silently have no "
            "effect at inference. Sample adapter keys: %s. Sample "
            "base-model modules: %s. Check your trainer-side key "
            "naming or add a rule to normalize_lora_key.",
            source_path, len(lora_modules), sample_lora, sample_base,
        )


class HybridAdapterManager:
    """Manager that composes a Tokenformer sub-manager and a LoRA
    sub-manager behind the same interface the runner mixin expects.

    Phase 2 skeleton: only the Tokenformer half is wired. LoRA/hybrid
    adapters raise NotImplementedError until the LoRA-from-.pt loader
    is in place.
    """

    def __init__(
        self,
        model: "nn.Module",
        device: "torch.device",
        vllm_config: "VllmConfig | None" = None,
    ):
        """Instantiate both sub-managers.

        Load order is LoRA layer replacement first, then the Tokenformer
        surgeon. This yields the composition:

            base(x) + lora_delta(x)  (inside the MLP block)
            + tokenformer_delta(x)   (added by the surgeon wrapper)

        When `vllm_config` is None (or `vllm_config.lora_config` is
        None), the LoRA sub-manager is not instantiated and the
        hybrid manager behaves like a pure Tokenformer manager — this
        keeps the skeleton path alive for callers that haven't flipped
        to hybrid yet.
        """
        lora_enabled = (
            vllm_config is not None and vllm_config.lora_config is not None
        )

        if lora_enabled:
            # LoRA sub-manager replaces targeted linears with *WithLoRA
            # wrappers, returning the transformed model. We then feed
            # that into the Tokenformer surgeon.
            self._lora: Any = PTWorkerLoRAManager(
                vllm_config,
                device,
                model.embedding_modules,
            )
            model = self._lora.create_lora_manager(model, vllm_config)
        else:
            self._lora = None

        self._tokenformer = TokenformerModelManager(model=model, device=device)
        # adapter_id -> AdapterKind, so remove/activate can route correctly.
        self._kinds: dict[int, AdapterKind] = {}

    # --- model handle exposed to the runner -----------------------------

    @property
    def model(self) -> "nn.Module":
        # Today this is the Tokenformer-wrapped model. When the LoRA
        # sub-manager arrives, order will be: apply LoRA layer
        # replacement first, then Tokenformer surgeon on top — so this
        # property will return the model after both passes. The
        # Tokenformer sub-manager already holds a reference to the
        # post-surgeon model, so reading from it Just Works.
        return self._tokenformer.model

    # --- adapter lifecycle ---------------------------------------------

    def add_adapter(self, lora_request) -> bool:
        """Classify the adapter, then route each half to its sub-manager.

        A hybrid adapter's Tokenformer tensors go to the Tokenformer
        manager; its LoRA tensors go to the LoRA manager. Both halves
        share the same adapter id. `activate`/`set_active_adapters` use
        `self._kinds` to fan out correctly.

        The sub-managers each re-load the `.pt` file from disk. This is
        redundant (we already classified once) but keeps their existing
        interfaces intact. Phase 2 isn't perf-sensitive; we'll tighten
        this in a later step by passing pre-split dicts through.
        """
        loaded = load_adapter_from_pt(lora_request.lora_path)
        kind = loaded.kind
        self._kinds[lora_request.adapter_id] = kind

        # Tokenformer half.
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.add_adapter(lora_request)

        # LoRA half.
        if kind in ("lora", "hybrid"):
            if self._lora is None:
                raise RuntimeError(
                    f"Adapter {lora_request.adapter_id} at "
                    f"{loaded.source_path} contains LoRA tensors, but the "
                    f"HybridAdapterManager was constructed without a "
                    f"LoRA-enabled vllm_config. Pass "
                    f"--enable-lora alongside --enable-tokenformer."
                )
            self._lora.add_adapter(lora_request)

        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        kind = self._kinds.pop(adapter_id, "tokenformer")
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.remove_adapter(adapter_id)
        if kind in ("lora", "hybrid") and self._lora is not None:
            self._lora.remove_adapter(adapter_id)
        return True

    def remove_all_adapters(self) -> None:
        self._kinds.clear()
        self._tokenformer.remove_all_adapters()
        if self._lora is not None:
            self._lora.remove_all_adapters()

    def pin_adapter(self, adapter_id: int) -> bool:
        kind = self._kinds.get(adapter_id)
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.pin_adapter(adapter_id)
        if kind in ("lora", "hybrid") and self._lora is not None:
            self._lora.pin_adapter(adapter_id)
        return True

    def list_adapters(self) -> set[int]:
        """Union of adapter ids across both sub-managers.

        TokenformerModelManager returns a `dict[int, Any]`; the LRU
        LoRA worker manager returns a `set[int]`. Normalize to a set
        so the union is sensible; hybrid adapters are naturally
        de-duplicated. Returning a set matches upstream behavior, which
        is what `LoRAModelRunnerMixin.list_adapters` forwards to API
        callers.
        """
        ids: set[int] = set()
        tk = self._tokenformer.list_adapters()
        if tk is not None:
            ids |= set(tk)  # works for both dict (keys) and set
        if self._lora is not None:
            lora = self._lora.list_adapters()
            if lora is not None:
                ids |= set(lora)
        return ids

    # --- per-step activation -------------------------------------------

    def set_active_adapters(self, lora_requests, lora_mapping) -> None:
        """Fan out to both sub-managers.

        Both managers see the full request set so hybrid adapters
        (whose id is in both managers' registries) are activated in both
        places. Each sub-manager is expected to skip ids it doesn't own
        — Tokenformer uses the skip-unregistered guard we added in
        6529423ba, and the LRU LoRA manager already no-ops on unknown
        ids via `list_adapters` membership checks.
        """
        self._tokenformer.set_active_adapters(lora_requests, lora_mapping)
        if self._lora is not None:
            self._lora.set_active_adapters(lora_requests, lora_mapping)

    def activate_adapter(self, adapter_id: int) -> bool:
        kind = self._kinds.get(adapter_id, "tokenformer")
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.activate_adapter(adapter_id)
        if kind in ("lora", "hybrid") and self._lora is not None:
            self._lora.activate_adapter(adapter_id)
        return True

    def deactivate_adapter(self, adapter_id: int) -> bool:
        kind = self._kinds.get(adapter_id, "tokenformer")
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.deactivate_adapter(adapter_id)
        if kind in ("lora", "hybrid") and self._lora is not None:
            self._lora.deactivate_adapter(adapter_id)
        return True

    # --- warmup plumbing ------------------------------------------------

    @contextmanager
    def dummy_lora_cache(self):
        with self._tokenformer.dummy_lora_cache():
            if self._lora is not None:
                with self._lora.dummy_lora_cache():
                    yield
            else:
                yield

    def add_dummy_lora(self, lora_request, rank: int = 8) -> bool:
        """Register the dummy with both sub-managers when both are present.

        Tokenformer's `add_dummy_lora` is a no-op that exists purely to
        satisfy the warmup path. The LRU LoRA sub-manager, however,
        actually registers a rank-`rank` zero adapter at a slot — that
        registration is load-bearing for cudagraph profiling of the
        LoRA kernels. If we only forward to Tokenformer, the LoRA-side
        cudagraph capture runs without any dummy in the slots and
        misses the LoRA path entirely.
        """
        self._tokenformer.add_dummy_lora(lora_request, rank=rank)
        if self._lora is not None:
            self._lora.add_dummy_lora(lora_request, rank=rank)
        return True

    # --- misc -----------------------------------------------------------

    def supports_tower_connector_lora(self) -> bool:
        return False