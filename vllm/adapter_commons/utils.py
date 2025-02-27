# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional


## model functions
def deactivate_adapter(adapter_id: int, active_adapters: dict[int, None],
                       deactivate_func: Callable) -> bool:
    if adapter_id in active_adapters:
        deactivate_func(adapter_id)
        active_adapters.pop(adapter_id)
        return True
    return False


def add_adapter(adapter: Any, registered_adapters: dict[int, Any],
                capacity: int, add_func: Callable) -> bool:
    if adapter.id not in registered_adapters:
        if len(registered_adapters) >= capacity:
            raise RuntimeError('No free adapter slots.')
        add_func(adapter)
        registered_adapters[adapter.id] = adapter
        return True
    return False


def set_adapter_mapping(mapping: Any, last_mapping: Any,
                        set_mapping_func: Callable) -> Any:
    if last_mapping != mapping:
        set_mapping_func(mapping)
        return mapping
    return last_mapping


def remove_adapter(adapter_id: int, registered_adapters: dict[int, Any],
                   deactivate_func: Callable) -> bool:
    deactivate_func(adapter_id)
    return bool(registered_adapters.pop(adapter_id, None))


def list_adapters(registered_adapters: dict[int, Any]) -> dict[int, Any]:
    return dict(registered_adapters)


def get_adapter(adapter_id: int,
                registered_adapters: dict[int, Any]) -> Optional[Any]:
    return registered_adapters.get(adapter_id)


## worker functions
def set_active_adapters_worker(requests: set[Any], mapping: Optional[Any],
                               apply_adapters_func,
                               set_adapter_mapping_func) -> None:
    apply_adapters_func(requests)
    set_adapter_mapping_func(mapping)


def add_adapter_worker(adapter_request: Any, list_adapters_func,
                       load_adapter_func, add_adapter_func,
                       activate_adapter_func) -> bool:
    if adapter_request.adapter_id in list_adapters_func():
        return False
    loaded_adapter = load_adapter_func(adapter_request)
    loaded = add_adapter_func(loaded_adapter)
    activate_adapter_func(loaded_adapter.id)
    return loaded


def apply_adapters_worker(adapter_requests: set[Any], list_adapters_func,
                          adapter_slots: int, remove_adapter_func,
                          add_adapter_func) -> None:
    models_that_exist = list_adapters_func()
    models_map = {
        adapter_request.adapter_id: adapter_request
        for adapter_request in adapter_requests if adapter_request
    }
    if len(models_map) > adapter_slots:
        raise RuntimeError(
            f"Number of requested models ({len(models_map)}) is greater "
            f"than the number of GPU model slots "
            f"({adapter_slots}).")
    new_models = set(models_map)
    models_to_add = new_models - models_that_exist
    models_to_remove = models_that_exist - new_models
    for adapter_id in models_to_remove:
        remove_adapter_func(adapter_id)
    for adapter_id in models_to_add:
        add_adapter_func(models_map[adapter_id])


def list_adapters_worker(adapter_manager_list_adapters_func) -> set[int]:
    return set(adapter_manager_list_adapters_func())
