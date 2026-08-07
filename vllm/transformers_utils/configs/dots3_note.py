# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import DeepseekV3Config


class Dots3NoteConfig(DeepseekV3Config):
    """Configuration for the Dots3 Note hybrid DSA/SWA model."""

    model_type = "dots3_note"

    def __init__(self, **kwargs):
        # These fields are absent from existing Note config.json files.  Do
        # not inherit DeepSeek-V3's 8-group/4-group router defaults: Note was
        # trained with an ungrouped (1/1) noaux_tc router, as defined by the
        # reference Dots3Config in SGLang haimain.  A different grouping
        # changes the selected experts at every MoE layer.
        kwargs.setdefault("n_group", 1)
        kwargs.setdefault("topk_group", 1)
        # Dots3 projects the indexer RoPE coordinates in adjacent (GPT-J)
        # pairs.  DeepSeek-V3.2 defaults to the split-half NeoX layout when
        # this flag is absent, which rotates different learned coordinates.
        kwargs.setdefault("indexer_rope_interleave", True)
        kwargs.setdefault("num_nextn_predict_layers", 1)
        super().__init__(**kwargs)
