# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def get_layer_name_after_index(layer_name: str) -> str:
    """Return the suffix following the final numeric component of a layer name."""
    parts = layer_name.split(".")
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].isdigit():
            return ".".join(parts[index + 1 :])
    return layer_name