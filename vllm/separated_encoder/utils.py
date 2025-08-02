# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.multimodal.inputs import PlaceholderRange


def mm_pos_info_to_dict(pos_info: PlaceholderRange):
    return {
        "offset": pos_info.offset,
        "length": pos_info.length,
        "is_embed": pos_info.is_embed if
        (pos_info.is_embed is not None) else None,
    }


def dict_to_pos_info(pos_info_dict: dict):
    return PlaceholderRange(offset=pos_info_dict["offset"],
                            length=pos_info_dict["length"],
                            is_embed=pos_info_dict["is_embed"])
