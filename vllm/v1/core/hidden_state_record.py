# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the P/D hidden-state handoff keeps its record (prototype).

The prefiller (P) hands the decoder (D) a per-request "record" of hidden states
from the last prompt positions (see ``RecordLayout`` and
``vllm.v1.worker.gpu.hidden_state_handoff``). It travels in the request's own
KV blocks: in the token slots just past the prompt, ``N .. N + tail - 1`` for an
``N``-token prompt, of one attention cache group (the "carrier"): a
full-attention group if there is one, else a sliding-window group (whose newest
blocks, covering the slots past the prompt, are kept and transferred).

Those slots are free on both sides when the record is written and read: P
writes the record once its last step is done (the request then finishes), and D
reads it in the step that samples token ``N``, before its first decode step
writes slot ``N``. The scheduler reserves the ``tail`` slots past the prompt on
both sides and P hands over the blocks covering them, so any connector that
transfers whole blocks carries the record with the KV.

Each slot holds the record bytes in every KV head (one copy per head), so a
decoder at a different TP size, which receives a different head slice of each
block, still finds a whole record in each of its heads.

The slots past the prompt are unused KV as far as attention is concerned, but
kernels may still load them (masked), and blocks are reused with their contents.
So each record byte is stored as two bytes holding a nibble each (0x0 - 0xF),
which read as small finite values in any KV cache format (including FP8 and
scale bytes), never as NaN or Inf.
"""

import os
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)

logger = init_logger(__name__)

# Auxiliary hidden-state layers the target captures for an EAGLE3-style
# drafter when the drafter config names none (models' default).
_DEFAULT_NUM_AUX_LAYERS = 3

# Stored bytes per record byte (one nibble each; see the module docstring).
RECORD_ENCODING_EXPANSION = 2


@dataclass(frozen=True)
class RecordLayout:
    """A record is ``[num_slots, hidden_size]``:

    * slot 0: the model output at the last prompt position (sampling).
    * With a hidden-state drafter (EAGLE, EAGLE3, MTP, DFlash, DSpark), the
      drafter's inputs for the last ``window`` prompt positions: the target
      hidden states handed to the drafter, then each auxiliary layer's.
      ``window`` is the drafter's prefill lookahead: 1, or k for k-module MTP.
      Token-only drafters (ngram, draft model) need no drafter inputs.
    """

    hidden_size: int
    # Prompt positions of drafter inputs stored (0: no hidden-state drafter).
    window: int = 0
    # EAGLE3 auxiliary hidden states per position.
    num_aux: int = 0

    @property
    def num_slots(self) -> int:
        return 1 + self.window * (1 + self.num_aux)

    def drafter_slots(self, kind: int) -> slice:
        """Slots of the drafter input ``kind`` (0: target hidden states,
        1 + i: auxiliary layer i), ordered by position, last position last."""
        start = 1 + kind * self.window
        return slice(start, start + self.window)


def get_record_layout(vllm_config: VllmConfig) -> RecordLayout:
    hidden_size = vllm_config.model_config.get_hidden_size()
    spec_config = vllm_config.speculative_config
    if spec_config is None or spec_config.use_ngram() or spec_config.uses_draft_model():
        # The drafter reads only token ids (its own KV comes with the transfer).
        return RecordLayout(hidden_size)
    # Hidden-state drafters: EAGLE, EAGLE3, MTP (incl. multi-module and Gemma4),
    # DFlash (incl. DFlash2/LiLiCorr), DSpark, and the hidden-state extractor
    # (which reads only the auxiliary ones).
    if not (spec_config.use_eagle() or spec_config.method == "extract_hidden_states"):
        raise NotImplementedError(
            "P/D hidden-state handoff does not support "
            f"{spec_config.method!r} speculative decoding."
        )
    num_aux = 0
    if spec_config.method in ("eagle3", "dflash", "dspark", "extract_hidden_states"):
        # The target model outputs auxiliary hidden states for these drafters
        # (see GPUModelRunner.use_aux_hidden_state_outputs). The worker checks
        # the count against the model's.
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        aux_layers = get_eagle3_aux_layers_from_config(spec_config)
        num_aux = len(aux_layers) if aux_layers else _DEFAULT_NUM_AUX_LAYERS
    window = max(1, vllm_config.num_prefill_lookahead_tokens)
    return RecordLayout(hidden_size, window, num_aux)


def get_record_carrier_group(kv_cache_config: KVCacheConfig) -> int:
    """The transferred attention group whose blocks carry the record: the first
    full-attention one (incl. MLA and R-SWA), else the first sliding-window
    one. Its layers must keep one token per cache slot. Ring buffers and
    compressed caches cannot carry: their slots past the prompt are in use."""
    candidates: dict[type, int] = {}
    for group_id in kv_cache_config.transfer_group_ids:
        specs = iter_layer_specs(
            kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        )
        for kind in (FullAttentionSpec, SlidingWindowSpec):
            if all(
                isinstance(spec, kind) and spec.tokens_per_state == 1 for spec in specs
            ):
                candidates.setdefault(kind, group_id)
    preference: tuple[type, ...] = (FullAttentionSpec, SlidingWindowSpec)
    if os.environ.get("VLLM_HIDDEN_STATE_RECORD_CARRIER") == "sliding_window":
        # Testing only: carry the record in a sliding-window group even when a
        # full-attention group exists.
        preference = preference[::-1]
    for preferred in preference:
        if preferred in candidates:
            return candidates[preferred]
    raise NotImplementedError(
        "P/D hidden-state handoff needs a transferred full-attention or "
        "sliding-window KV cache group."
    )


def _group_record_layers(kv_cache_config: KVCacheConfig) -> dict[str, int]:
    """The carrier group's layers in ``kv_cache_config`` with the bytes one KV
    head of one token slot holds."""
    group = kv_cache_config.kv_cache_groups[get_record_carrier_group(kv_cache_config)]
    spec = group.kv_cache_spec
    return {
        name: (
            spec.kv_cache_specs[name]
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else spec
        ).state_content_size_bytes
        for name in group.layer_names
    }


def _layer_index(layer_name: str) -> int | None:
    """A layer's index in the model, or None if its name has no single one."""
    indices = [int(part) for part in layer_name.split(".") if part.isdigit()]
    return indices[0] if len(indices) == 1 else None


def get_record_producer_pp_size(vllm_config: VllmConfig) -> int:
    """The prefiller's pipeline-parallel size: this instance's own on P; on D,
    ``kv_connector_extra_config["hidden_state_handoff_producer_pp_size"]``
    (default 1). The prefiller keeps the records on its last stage, which has
    the hidden states."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    if kv_transfer_config.is_kv_producer:
        return vllm_config.parallel_config.pipeline_parallel_size
    return int(
        kv_transfer_config.get_from_extra_config(
            "hidden_state_handoff_producer_pp_size", 1
        )
    )


def resolve_record_layers(
    vllm_config: VllmConfig, kv_cache_configs: list[KVCacheConfig]
) -> tuple[tuple[str, int], ...]:
    """The layers holding the record, resolved from the per-layer specs of every
    worker (pipeline stage): the carrier group's layers on the prefiller's last
    stage (all of them without PP), in model order, each with the bytes one KV
    head of one token slot holds."""
    layers: dict[str, int] = {}
    for kv_cache_config in kv_cache_configs:
        layers.update(_group_record_layers(kv_cache_config))
    pp_size = get_record_producer_pp_size(vllm_config)
    if pp_size > 1:
        from vllm.distributed.utils import get_pp_indices

        num_layers = vllm_config.model_config.get_total_num_hidden_layers()
        start, end = get_pp_indices(num_layers, pp_size - 1, pp_size)
        layers = {
            name: size
            for name, size in layers.items()
            if (index := _layer_index(name)) is not None and start <= index < end
        }
        if not layers:
            raise NotImplementedError(
                "P/D hidden-state handoff needs a carrier-group layer on the "
                "prefiller's last pipeline stage."
            )
    return tuple(sorted(layers.items(), key=lambda item: _layer_index(item[0]) or 0))


def get_record_layers(kv_cache_config: KVCacheConfig) -> tuple[tuple[str, int], ...]:
    """The layers holding the record, each with the bytes one KV head of one
    token slot holds (see ``resolve_record_layers``, which the engine resolves
    into the config; the scheduler's copy keeps one spec per group)."""
    if kv_cache_config.hidden_state_record_layers:
        return kv_cache_config.hidden_state_record_layers
    # A config not built by the engine (e.g. in tests): no PP.
    return tuple(_group_record_layers(kv_cache_config).items())


def get_record_tail_tokens(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> int:
    """Token slots past the prompt that hold the record."""
    layout = get_record_layout(vllm_config)
    record_bytes = (
        layout.num_slots
        * layout.hidden_size
        * get_dtype_size(vllm_config.model_config.dtype)
        * RECORD_ENCODING_EXPANSION
    )
    head_bytes = sum(size for _, size in get_record_layers(kv_cache_config))
    return cdiv(record_bytes, head_bytes)


# NIXL connectors the handoff is on by default for.
_DEFAULT_CONNECTORS = ("NixlConnector", "NixlPullConnector", "NixlPushConnector")


def _handoff_default_blockers(vllm_config: VllmConfig) -> list[str]:
    """Conditions, the same on the prefiller and the decoder, under which the
    handoff is off by default (an explicit opt-in is checked later)."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    model_config = vllm_config.model_config
    blockers = {
        f"the {kv_transfer_config.kv_connector} KV connector": (
            kv_transfer_config.kv_connector not in _DEFAULT_CONNECTORS
        ),
        f"kv_role {kv_transfer_config.kv_role}": (
            kv_transfer_config.kv_role not in ("kv_producer", "kv_consumer")
        ),
        "pooling models": model_config is not None
        and model_config.runner_type == "pooling",
        # No attention KV to carry the record.
        "attention-free models": model_config is not None
        and model_config.is_attention_free,
        "multimodal multi-module MTP": model_config is not None
        and model_config.is_multimodal_model
        and vllm_config.num_prefill_lookahead_tokens > 1,
        # Unsupported, but configured alike on both sides in practice (a
        # mismatch fails the NIXL handshake with a clear error).
        "NIXL host buffers": kv_transfer_config.kv_buffer_device == "cpu",
        "model runner V1": not vllm_config.use_v2_model_runner,
    }
    return [name for name, on in blockers.items() if on]


def _handoff_unsupported(vllm_config: VllmConfig) -> list[str]:
    """Features the handoff does not support, errors with it on. The parallel
    and HiSparse ones are configured on one side only, so they cannot turn the
    default off without the other side disagreeing."""
    parallel_config = vllm_config.parallel_config
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    unsupported = {
        "decode context parallelism": (
            parallel_config.decode_context_parallel_size > 1
        ),
        "prefill context parallelism": (
            parallel_config.prefill_context_parallel_size > 1
        ),
        # A prefiller keeps the records on its last stage (NIXL supports PP
        # only there).
        "pipeline parallelism on the decoder": (
            parallel_config.pipeline_parallel_size > 1
            and not kv_transfer_config.is_kv_producer
        ),
        "HiSparse": vllm_config.attention_config.hisparse_config is not None,
        "NIXL host buffers": kv_transfer_config.kv_buffer_device == "cpu",
        "model runner V1": not vllm_config.use_v2_model_runner,
    }
    return [name for name, on in unsupported.items() if on]


def resolve_hidden_state_handoff(vllm_config: VllmConfig) -> None:
    """Resolve whether the P/D hidden-state handoff is on: as configured
    (``kv_connector_extra_config["hidden_state_handoff"]``), else by default
    wherever both the prefiller and the decoder can tell it is supported.
    Raise for features it does not support while on."""
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or kv_transfer_config.kv_connector is None:
        return
    extra_config = kv_transfer_config.kv_connector_extra_config
    explicit = "hidden_state_handoff" in extra_config
    if not explicit:
        extra_config["hidden_state_handoff"] = not _handoff_default_blockers(
            vllm_config
        )
    if not kv_transfer_config.hidden_state_handoff:
        return
    if unsupported := _handoff_unsupported(vllm_config):
        how = "is enabled" if explicit else "is on by default with NIXL"
        raise ValueError(
            f"The P/D hidden-state handoff {how} but does not support "
            f"{', '.join(unsupported)}. Disable it on both the prefiller and "
            "the decoder with "
            '--kv-transfer-config \'{"kv_connector_extra_config": '
            '{"hidden_state_handoff": false}}\'.'
        )
    logger.info_once(
        "P/D hidden-state handoff enabled%s: the prefiller computes the whole "
        "prompt and hands the decoder its last hidden states.",
        "" if explicit else " (default)",
    )


def get_hidden_state_handoff_compat(vllm_config: VllmConfig) -> int:
    """What a prefiller and decoder must agree on: 0 with the handoff off,
    else the prefiller's pipeline-parallel size (where the records live)."""
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or not kv_transfer_config.hidden_state_handoff:
        return 0
    return get_record_producer_pp_size(vllm_config)
