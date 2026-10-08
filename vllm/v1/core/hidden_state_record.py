# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the P/D hidden-state handoff keeps its record (prototype).

The prefiller (P) hands the decoder (D) a per-request "record" of hidden states
from the last prompt positions (see ``RecordLayout`` and
``vllm.v1.worker.gpu.hidden_state_handoff``). It travels in the request's own
KV blocks: in the token slots just past the prompt, ``N .. N + tail - 1`` for an
``N``-token prompt, of one full-attention cache group (the "carrier").

Those slots are free on both sides when the record is written and read: P
writes the record once its last step is done (the request then finishes), and D
reads it in the step that samples token ``N``, before its first decode step
writes slot ``N``. The scheduler reserves the ``tail`` slots past the prompt on
both sides and P hands over the blocks covering them, so any connector that
transfers whole blocks carries the record with the KV.

Each slot holds the record bytes in every KV head (one copy per head), so a
decoder at a different TP size, which receives a different head slice of each
block, still finds a whole record in each of its heads.
"""

from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    RSWASpec,
    iter_layer_specs,
)

# Auxiliary hidden-state layers the target captures for an EAGLE3-style
# drafter when the drafter config names none (models' default).
_DEFAULT_NUM_AUX_LAYERS = 3


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
    # DFlash (incl. DFlash2/LiLiCorr) and DSpark.
    if not spec_config.use_eagle():
        raise NotImplementedError(
            "P/D hidden-state handoff does not support "
            f"{spec_config.method!r} speculative decoding."
        )
    num_aux = 0
    if spec_config.method in ("eagle3", "dflash", "dspark"):
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
    """The transferred full-attention group whose blocks carry the record."""
    for group_id in kv_cache_config.transfer_group_ids:
        specs = iter_layer_specs(
            kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        )
        if all(
            isinstance(spec, FullAttentionSpec)
            and not isinstance(spec, RSWASpec)
            and spec.tokens_per_state == 1
            for spec in specs
        ):
            return group_id
    raise NotImplementedError(
        "P/D hidden-state handoff needs a transferred full-attention KV cache group."
    )


def get_record_head_bytes(kv_cache_config: KVCacheConfig, group_id: int) -> int:
    """Bytes of one KV head of one token slot, summed over the carrier group's
    layers: the record bytes one token slot holds."""
    group = kv_cache_config.kv_cache_groups[group_id]
    specs = iter_layer_specs(group.kv_cache_spec)
    if len(specs) == len(group.layer_names):
        return sum(spec.state_content_size_bytes for spec in specs)
    # A uniform group spec stands for each of its layers.
    (spec,) = specs
    return spec.state_content_size_bytes * len(group.layer_names)


def get_record_tail_tokens(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> int:
    """Token slots past the prompt that hold the record."""
    layout = get_record_layout(vllm_config)
    record_bytes = (
        layout.num_slots
        * layout.hidden_size
        * get_dtype_size(vllm_config.model_config.dtype)
    )
    group_id = get_record_carrier_group(kv_cache_config)
    return cdiv(record_bytes, get_record_head_bytes(kv_cache_config, group_id))
