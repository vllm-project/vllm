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


def retag_moe_sleep_weights(model) -> None:
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.model_executor.models.interfaces import get_mixture_of_experts_model

    moe = get_mixture_of_experts_model(model)
    if moe is None:
        return
    allocator = CuMemAllocator.get_instance()
    allocator.rename_tag("weights", "shared_weights")
    expert_ptrs = {
        weight.untyped_storage().data_ptr()
        for group in moe.expert_weights
        for weight in group
    }
    allocator.retag_allocations_by_ptrs(expert_ptrs, "expert_weights")
