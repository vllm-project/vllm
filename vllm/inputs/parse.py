# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .data import ProcessorInputs, SingletonInputs


def split_enc_dec_inputs(
    inputs: ProcessorInputs,
) -> tuple[SingletonInputs | None, SingletonInputs]:
    if inputs["type"] == "enc_dec":
        return inputs["encoder_prompt"], inputs["decoder_prompt"]

    return None, inputs
