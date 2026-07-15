# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongCat-Flash config for checkpoints without usable remote code.

``meituan-longcat/LongCat-2.0`` ships ``model_type: null`` with
``architectures: ["LongcatCausalLM"]`` and no ``auto_map``, so it cannot be
resolved by ``AutoConfig``. This subclass of the upstream transformers
``LongcatFlashConfig`` fills that gap and aliases the LongCat-2.0
``oe_{vocab_size_ratio,neighbor_num,split_num}`` field names to the n-gram
embedding fields (``ngram_vocab_size_ratio``/``emb_neighbor_num``/
``emb_split_num``) the vLLM model code uses.
"""

import dataclasses

from transformers.models.longcat_flash import (
    LongcatFlashConfig as _HfLongcatFlashConfig,
)


def _coerce_float_fields(kwargs: dict) -> dict:
    """The upstream config is a strict dataclass; promote ints in the
    checkpoint json (e.g. ``routed_scaling_factor: 9``) to float fields."""
    if not dataclasses.is_dataclass(_HfLongcatFlashConfig):
        return kwargs
    for field in dataclasses.fields(_HfLongcatFlashConfig):
        if field.type in (float, "float") and isinstance(kwargs.get(field.name), int):
            kwargs[field.name] = float(kwargs[field.name])
    return kwargs


class LongcatFlashNgramConfig(_HfLongcatFlashConfig):
    model_type = "longcat_flash_ngram"

    def __init__(
        self,
        oe_vocab_size_ratio=None,
        oe_neighbor_num=None,
        oe_split_num=None,
        ngram_vocab_size_ratio=None,
        emb_neighbor_num=None,
        emb_split_num=None,
        **kwargs,
    ):
        self.ngram_vocab_size_ratio = (
            ngram_vocab_size_ratio
            if ngram_vocab_size_ratio is not None
            else oe_vocab_size_ratio
        )
        self.emb_neighbor_num = (
            emb_neighbor_num if emb_neighbor_num is not None else oe_neighbor_num
        )
        self.emb_split_num = (
            emb_split_num if emb_split_num is not None else oe_split_num
        )

        super().__init__(**_coerce_float_fields(kwargs))

        # ``num_hidden_layers`` counts attention sublayers (two per decoder
        # layer); the model builds ``num_layers`` decoder layers
        # (FlashNgramModel re-syncs it at build).
        if kwargs.get("num_hidden_layers") is None:
            self.num_hidden_layers = self.num_layers * 2
