# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .hf import HfRenderer


class Nemotron3_5AsrRenderer(HfRenderer):
    def get_dec_start_token_id(self) -> int:
        return self.model_config.hf_config.blank_token_id

    def get_eos_token_id(self) -> int:
        # Acoustic blanks are consumed inside the RNNT loop. The model only
        # exposes blank to the engine once all encoder frames are exhausted.
        return self.model_config.hf_config.blank_token_id
