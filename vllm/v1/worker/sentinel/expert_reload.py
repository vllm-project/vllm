# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import wraps
from inspect import signature


def _placement_weight_loader(
    loader: Callable,
    local_physical_ids: dict[int, list[int]],
    num_logical: int,
    num_physical: int,
    use_global_sf: bool = False,
) -> Callable:
    """Translate initial checkpoint IDs before buffering layerwise reload calls."""
    loader_signature = signature(loader)

    @wraps(loader)
    def online_process_loader(*args, **kwargs):
        bound = loader_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        source = bound.arguments["expert_id"]
        # Preserve global activation scales: NVFP4 reduces all experts' input
        # scales, including experts that are not hosted on this rank.
        if use_global_sf and "input_scale" in bound.arguments.get("weight_name", ""):
            return loader(*args, **kwargs)
        if source >= num_physical:
            return loader(*args, **kwargs)
        # Reload every local replica: checkpoint mappings enumerate the initial
        # placement, so dispatch its first copy to every recovered physical slot.
        if source >= num_logical:
            return False if bound.arguments.get("return_success") else None
        success = False
        for physical_id in local_physical_ids.get(source, []):
            bound.arguments["expert_id"] = physical_id
            success = bool(loader(*bound.args, **bound.kwargs)) or success
        return success if bound.arguments.get("return_success") else None

    return online_process_loader
