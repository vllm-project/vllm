# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Granite Switch: a Granite model with adapter switching.

A Granite Switch checkpoint carries its LoRA adapters baked in and selects one
per token from a control-token signal, so every field that describes the
adapter population is frozen into ``config.json`` when the checkpoint is built.
This config class is the on-disk contract: it is what makes a checkpoint
self-describing, and its validation is what turns a malformed checkpoint into a
startup error instead of silently wrong routing.
"""

from transformers import GraniteMoeHybridConfig

# Decoder-layer cache slots the switch reserves at the front of the model when
# adapters are present. The coded MultiSwitch engine owns two: a counting slot
# and a memory slot. Single source of truth: ``num_hidden_layers`` on disk is
# inflated by this count, and the model subtracts it to recover the physical
# decoder-layer count.
SWITCH_CACHE_LAYERS = 2


class GraniteSwitchConfig(GraniteMoeHybridConfig):
    """Configuration class for the GraniteSwitch model.

    Extends the Granite base config with parameters for adapter switching. The
    switch engine is the Kerdock/Delsarte-Goethals coded-memory MultiSwitch.
    Control tokens are handled exclusively via token exchange: the switch reads
    ``input_ids``, decides the active adapter, and rewrites each control token
    to its substitute id (from ``adapter_substitute_token_ids``) before the
    decoder embeds the sequence. The decoder is unaware of the substitution.

    Args:
        num_adapters: Number of LoRA adapters available. Default 0 (no
            adapters). Counts real LoRA adapters only, not the base; index 0
            always means "base / no adapter".
        adapter_token_ids: Token ids for adapter control. Length
            ``num_adapters`` (one token per real adapter), or
            ``num_adapters + 1`` with a leading base-reset token. Must be
            unique. ``adapter_token_ids[i]`` activates adapter ``i + 1``;
            output 0 is base, which needs no token.
        adapter_substitute_token_ids: Token ids whose embeddings replace the
            control-token embeddings before the decoder runs. Same length rule
            as ``adapter_token_ids``. Required when ``num_adapters > 0``.
        control_token_gain: Attention gain separating control from non-control
            tokens. Default 15.0.
        switch_head_dim: Dimension of the switch attention Q/K/V vectors.
            Default 32.
        ms_code_m: Code parameter of the coded memory head. The codebook holds
            ``2 ** (2 * m - 1)`` vectors of dimension ``2 ** m`` for kerdock.
            Default 6.
        ms_code_type: Codebook family, ``"kerdock"`` or ``"dg1"``.
            Default ``"kerdock"``.
        ms_memory_gain: Attention gain applied to the memory head's code
            vectors. Default 28.0.
        ms_counting_head_dim: Head dimension of the counting head, whose
            ``1 / (1 + n)`` signal recovers the write address. Default 32.
        adapter_names: Ordered adapter names, for name-to-index mapping.
        max_lora_rank: Maximum rank across all adapters. Every adapter's
            weights are padded to this rank on disk. Default 8.
        adapter_ranks: Per-adapter ranks; length must equal ``num_adapters``.
        lora_target_modules: Module *group* names to apply LoRA to, drawn from
            ``qkv_proj``, ``o_proj``, ``shared_input_linear`` and
            ``shared_output_linear``. Defaults to the groups the architecture
            actually has.
        cross_stream_rank: LoRA rank of the layer-level ``cross_stream``
            injection site. Required when ``dual_stream`` is True, and must be
            ``None`` otherwise, since the site is not allocated at all.
        dual_stream: Whole-checkpoint decoder mode. ``False`` (default) is
            ordinary LoRA/aLoRA, one stream. ``True`` is Shadow Residual, where
            every adapter runs a base stream plus an adapter stream and takes
            K/V from the base stream. A checkpoint holds either Shadow Residual
            adapters or LoRA/aLoRA adapters, never both, so this is a single
            flag rather than a per-adapter list.
        fused_add_norm: Whether the decoder applies the residual add and the
            following norm as one fused operation, which fixes the
            floating-point reduction order the checkpoint was built against.
        num_local_experts: Number of sparse MoE experts. 0 (default) selects
            the dense Granite 4 configuration.
        position_embedding_type: Positional encoding; the switch model is
            attention-only with RoPE.
        layer_types: Per-layer type list. Pinned to all-attention when unset;
            see the note in ``__init__``.
        kwargs: Additional arguments forwarded to the parent config.

    """

    model_type = "granite_switch"

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Reject SingleSwitch checkpoints; MultiSwitch is the only engine.

        A checkpoint built for the earlier SingleSwitch engine cannot run as
        MultiSwitch: MultiSwitch reserves ``SWITCH_CACHE_LAYERS == 2`` cache
        slots where SingleSwitch reserved 1, so a single-sized checkpoint has
        one decoder layer too few and its weights cannot map. There is no
        auto-migration; it must be rebuilt.

        This is the one seam that still sees the raw on-disk config, so the
        rejection lives here. Two shapes identify such a checkpoint (with
        adapters present):

        * an explicit ``switch_type`` that is not ``"multi"``; or
        * no ``switch_type`` key AND no ``ms_code_m`` key, which marks a
          checkpoint predating the coded engine. ``ms_code_m`` is serialized by
          every MultiSwitch checkpoint, so its absence is the reliable marker.

        A stale ``switch_type`` key on a real MultiSwitch checkpoint is
        stripped before delegating, so the removed parameter never reaches the
        parent.
        """
        if config_dict.get("num_adapters", 0) > 0:
            st = config_dict.get("switch_type")
            legacy_single = st is None and "ms_code_m" not in config_dict
            if (st is not None and st != "multi") or legacy_single:
                raise ValueError(
                    "This checkpoint was built for the SingleSwitch engine, "
                    "which has been removed. MultiSwitch is now the only "
                    "engine, and a SingleSwitch checkpoint cannot be loaded as "
                    "MultiSwitch because it was sized for a different "
                    "decoder-layer count. A loadable checkpoint must set "
                    "switch_type to 'multi' or omit it entirely, and must "
                    "carry ms_code_m; rebuild it from its source adapters."
                )
        if "switch_type" in config_dict:
            config_dict = {k: v for k, v in config_dict.items() if k != "switch_type"}
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        num_adapters: int = 0,
        adapter_token_ids: list[int] | None = None,
        adapter_substitute_token_ids: list[int] | None = None,
        # Switch attention parameters
        control_token_gain: float = 15.0,
        switch_head_dim: int = 32,
        # MultiSwitch (coded engine) parameters
        ms_code_m: int = 6,
        ms_code_type: str = "kerdock",
        ms_memory_gain: float = 28.0,
        ms_counting_head_dim: int = 32,
        # Adapter parameters
        adapter_names: list[str] | None = None,
        max_lora_rank: int = 8,
        adapter_ranks: list[int] | None = None,
        lora_target_modules: list[str] | None = None,
        # Shadow Residual (SR) parameters
        cross_stream_rank: int | None = None,
        dual_stream: bool = False,
        # Residual-norm convention
        fused_add_norm: bool = False,
        # Parent class defaults (Granite 4 dense configuration)
        num_local_experts: int = 0,
        position_embedding_type: str = "rope",
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        # This model is attention-only, but the hybrid parent fills an unset
        # ``layer_types`` with all-mamba, which would size the per-layer cache
        # wrongly. Pin it here. Length must equal ``num_hidden_layers`` (cache
        # slots included - they are attention too) for the parent's check.
        if layer_types is None:
            num_hidden_layers = kwargs.get("num_hidden_layers", 32)
            layer_types = ["full_attention"] * num_hidden_layers

        super().__init__(
            num_local_experts=num_local_experts,
            position_embedding_type=position_embedding_type,
            layer_types=layer_types,
            **kwargs,
        )

        # The parent's fixed 1024 default is the wrong width for dense bases and
        # cannot express the "no shared MLP" sentinel (0) that pure sparse-MoE
        # bases need, so resolve it here. An explicit value (including 0) is
        # honored; only an unset one is resolved.
        if kwargs.get("shared_intermediate_size") is None:
            self.shared_intermediate_size = (
                0 if num_local_experts > 0 else self.intermediate_size
            )

        if num_adapters < 0:
            raise ValueError(f"num_adapters must be >= 0, got {num_adapters}")
        self.num_adapters = num_adapters

        # MultiSwitch (Kerdock/DG coded-memory) params.
        self.ms_code_m = ms_code_m
        self.ms_code_type = ms_code_type
        self.ms_memory_gain = ms_memory_gain
        self.ms_counting_head_dim = ms_counting_head_dim

        # Allowed control-token-list lengths. MultiSwitch accepts num_adapters
        # (no base slot) OR num_adapters+1 (leading base-reset token that writes
        # expert_id 0, enabling return-to-base mid-request).
        _allowed_lens = (num_adapters, num_adapters + 1)

        if num_adapters > 0 and adapter_token_ids is not None:
            if len(adapter_token_ids) not in _allowed_lens:
                raise ValueError(
                    f"adapter_token_ids length ({len(adapter_token_ids)}) must "
                    f"be one of {_allowed_lens} (num_adapters={num_adapters})."
                )
            # Token exchange builds the control -> substitute lookup keyed by
            # adapter token id; duplicates would silently collapse to one slot.
            if len(set(adapter_token_ids)) != len(adapter_token_ids):
                raise ValueError(
                    f"adapter_token_ids must be unique; got {adapter_token_ids}"
                )
        self.adapter_token_ids = adapter_token_ids

        # adapter_substitute_token_ids is required when num_adapters > 0.
        if num_adapters > 0:
            if adapter_substitute_token_ids is None:
                raise ValueError(
                    "adapter_substitute_token_ids is required when "
                    "num_adapters > 0. Every adapter needs a substitute token "
                    "id whose embedding replaces the control-token embedding "
                    "before the decoder runs."
                )
            if len(adapter_substitute_token_ids) not in _allowed_lens:
                raise ValueError(
                    f"adapter_substitute_token_ids length "
                    f"({len(adapter_substitute_token_ids)}) must be one of "
                    f"{_allowed_lens} (num_adapters={num_adapters})."
                )
            if adapter_token_ids is not None and len(
                adapter_substitute_token_ids
            ) != len(adapter_token_ids):
                raise ValueError(
                    "adapter_token_ids and adapter_substitute_token_ids must "
                    f"have the same length; got {len(adapter_token_ids)} and "
                    f"{len(adapter_substitute_token_ids)}."
                )
            if any(sid < 0 for sid in adapter_substitute_token_ids):
                raise ValueError(
                    "adapter_substitute_token_ids must all be >= 0 (real token "
                    f"ids); got {adapter_substitute_token_ids}"
                )
            if adapter_token_ids is None:
                raise ValueError(
                    "adapter_token_ids is required when "
                    "adapter_substitute_token_ids is provided (token exchange "
                    "maps control ids to substitute ids)."
                )
        self.adapter_substitute_token_ids = adapter_substitute_token_ids

        # Switch attention parameters
        self.control_token_gain = control_token_gain
        self.switch_head_dim = switch_head_dim
        self.fused_add_norm = fused_add_norm

        # Shadow Residual (SR).
        # Pre-fusion SR checkpoints carried unfused_qkv; their weight layout is
        # incompatible with the fused one. transformers silently keeps unknown
        # config keys, so without this an old checkpoint would load into a
        # fused model and quietly mismatch keys.
        if kwargs.get("unfused_qkv"):
            raise ValueError(
                "This checkpoint sets unfused_qkv=True, so it carries the old "
                "unfused Shadow Residual weight layout. Only the fused layout "
                "is supported: qkv_proj and shared_mlp.input_linear must be "
                "single pre-fused tensors. Rebuild the checkpoint."
            )
        self.dual_stream = bool(dual_stream)
        self.cross_stream_rank = cross_stream_rank
        if self.dual_stream and cross_stream_rank is None:
            raise ValueError(
                "cross_stream_rank is required when dual_stream is True "
                "(the cross_stream site must be allocated)."
            )
        if not self.dual_stream and cross_stream_rank is not None:
            raise ValueError(
                f"cross_stream_rank must be None when dual_stream is False; "
                f"got {cross_stream_rank}. The cross_stream site only exists "
                "in a Shadow Residual checkpoint."
            )

        self.adapter_names = adapter_names

        # QKV outputs vectors of size projection_head_dim. Prefer the base
        # model's explicit head_dim, since some models have
        # head_dim != hidden_size // num_attention_heads. Kept separate from
        # head_dim, which RoPE also reads.
        explicit_head_dim = kwargs.get("head_dim")
        self.projection_head_dim = (
            explicit_head_dim
            if explicit_head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )

        if num_adapters > 0:
            if adapter_ranks is None:
                raise ValueError("adapter_ranks must be provided when num_adapters > 0")
            if len(adapter_ranks) != num_adapters:
                raise ValueError(
                    f"adapter_ranks length ({len(adapter_ranks)}) must equal "
                    f"num_adapters ({num_adapters})"
                )
            if max(adapter_ranks) != max_lora_rank:
                raise ValueError(
                    f"max(adapter_ranks)={max(adapter_ranks)} must equal "
                    f"max_lora_rank={max_lora_rank}"
                )

        self.max_lora_rank = max_lora_rank
        self.adapter_ranks = adapter_ranks

        # Default LoRA target module groups, determined by the architecture.
        # Empty when num_adapters == 0 (no LoRA to apply).
        if lora_target_modules is None:
            lora_target_modules = []

            if self.num_adapters > 0:
                # Attention modules: the switch model is attention-only, so
                # every layer has them.
                lora_target_modules.extend(["qkv_proj", "o_proj"])

                # MLP modules: only where a shared_mlp exists to hold them.
                # Pure sparse MoE bases have none, and asking for the groups
                # anyway would build zero-width LoRA projections.
                if self.shared_intermediate_size > 0:
                    lora_target_modules.extend(
                        ["shared_input_linear", "shared_output_linear"]
                    )

        self.lora_target_modules = lora_target_modules
