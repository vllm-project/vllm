# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

WEIGHT_SLEEP_TAGS = ("weights", "shared_weights", "expert_weights")


def expand_weight_sleep_tags(tags: list[str] | None) -> list[str] | None:
    if tags is None:
        return None

    expanded: list[str] = []
    for tag in tags:
        if tag == "weights":
            expanded.extend(WEIGHT_SLEEP_TAGS)
        else:
            expanded.append(tag)
    return list(dict.fromkeys(expanded))
