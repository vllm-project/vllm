# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension for inspecting production reload tracing in day0 tests."""

import gc
import hashlib
import threading
import time
from functools import wraps
from itertools import chain
from typing import Any

import torch
from vllm.model_executor.model_loader.reload.integration import get_model_reload_tracer

from vllm.distributed import get_ep_group, get_world_group
from vllm.distributed.eplb.eplb_state import _commit_eplb_maps
from vllm.distributed.eplb.rebalance_execute import (
    rearrange_expert_weights_inplace,
)


def gather_rank_evidence(result):
    # DP broadcasts the RPC to all engines but returns only the first reply.
    group = get_world_group()
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, result, group=group.cpu_group)
    return {str(rank): value for rank, value in enumerate(gathered)}


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), 8 * 1024 * 1024):
        digest.update(raw[start : start + 8 * 1024 * 1024].cpu().numpy())
    return {
        "id": id(value),
        "ptr": value.data_ptr(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "hash": digest.hexdigest(),
    }


def _object_summary(value: Any, depth: int = 0) -> Any:
    """Serialize runtime tensor references without serializing kernel objects."""
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if depth >= 3 or value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_object_summary(item, depth + 1) for item in value[:16]]
    if isinstance(value, dict):
        return {
            str(key): _object_summary(item, depth + 1)
            for key, item in list(value.items())[:32]
        }
    if hasattr(value, "__dict__"):
        result = {"class": type(value).__name__}
        for name, item in vars(value).items():
            if name.startswith("__"):
                continue
            if isinstance(
                item, (torch.Tensor, list, tuple, dict, bool, int, float, str)
            ) or name in {
                "workspace",
                "config",
                "layer_config",
                "compute_config",
                "moe_quant_config",
                "precision_config",
                "impl",
                "fused_experts",
                "prepare_finalize",
                "locks",
            }:
                result[name] = _object_summary(item, depth + 1)
        return result
    return {"class": type(value).__name__}


class ReloadTraceEvidence:
    @staticmethod
    def _physical_to_logical(module):
        state = getattr(module, "eplb_state", None)
        manager = getattr(module, "expert_map_manager", None)
        if state is None or state.logical_to_physical_map is None:
            return None
        num_physical = manager.local_num_experts * manager.ep_size
        result = [-1] * num_physical
        for logical, replicas in enumerate(
            state.logical_to_physical_map.detach().cpu().tolist()
        ):
            for physical in replicas:
                if physical < 0:
                    continue
                if physical >= num_physical or result[physical] != -1:
                    raise RuntimeError("Invalid EPLB placement")
                result[physical] = logical
        if -1 in result:
            raise RuntimeError("Incomplete EPLB placement")
        return result

    def inspect_eplb(self):
        """Capture the current EPLB placement and local expert ownership."""
        result = []
        model = self.model_runner.model
        for name, module in model.named_modules():
            manager = getattr(module, "expert_map_manager", None)
            if manager is None:
                continue
            if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
                continue
            mapping = self._physical_to_logical(module)
            result.append(
                {
                    "module": name,
                    "physical_to_logical": mapping,
                    "logical_to_local": [
                        manager.map_global_to_local(expert)
                        for expert in range(manager.global_num_experts)
                    ],
                    "local_num_experts": manager.local_num_experts,
                    "global_num_experts": manager.global_num_experts,
                }
            )
        return gather_rank_evidence(result)

    def inspect_expert_slots(self):
        """Hash each routed physical slot together with its logical owner."""
        result = []
        model = self.model_runner.model
        for name, module in model.named_modules():
            manager = getattr(module, "expert_map_manager", None)
            if manager is None:
                continue
            placement = self._physical_to_logical(module)
            tensors = {}
            direct_tensors = dict(module.named_parameters(recurse=False))
            direct_tensors.update(module.named_buffers(recurse=False))
            for role, tensor in direct_tensors.items():
                if (
                    tensor is None
                    or tensor.ndim == 0
                    or tensor.shape[0] != manager.local_num_experts
                ):
                    continue
                tensors[role] = {
                    "slots": {
                        str(index): _tensor_summary(tensor[index])["hash"]
                        for index in range(tensor.shape[0])
                    },
                }
            result.append(
                {
                    "module": name,
                    "physical_to_logical": placement,
                    "local_num_experts": manager.local_num_experts,
                    "logical_to_local": [
                        manager.map_global_to_local(expert)
                        for expert in range(manager.global_num_experts)
                    ],
                    "slot_hashes": tensors,
                }
            )
            # The regular parameter snapshot already hashes every expert
            # tensor.  This targeted snapshot only proves the changed slots;
            # one representative MoE layer is sufficient and avoids hashing
            # hundreds of gigabytes during an RPC.
        return gather_rank_evidence(result)

    def set_eplb_mapping(self, physical_to_logical):
        """Install one mapping on every worker for a controlled reload round."""
        runner = self.model_runner
        if not getattr(runner.parallel_config, "enable_eplb", False):
            raise RuntimeError("EPLB is not enabled")
        mapping = torch.tensor(
            physical_to_logical,
            dtype=torch.int64,
            device=runner.device,
        )
        runner.setup_eplb_from_mapping(mapping)
        torch.accelerator.synchronize()
        return self.inspect_eplb()

    def rearrange_eplb_mapping(self, physical_to_logical):
        """Move existing expert tensors with the native EPLB transfer path."""
        runner = self.model_runner
        eplb_state = runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("EPLB state is not initialized")
        model_state = eplb_state.model_states[runner.model_config.compute_hash()]
        old_mapping = model_state.physical_to_logical_map.detach().clone()
        new_mapping = torch.tensor(
            physical_to_logical,
            dtype=old_mapping.dtype,
            device=old_mapping.device,
        )
        if new_mapping.shape != old_mapping.shape:
            raise ValueError(
                f"Expected mapping shape {tuple(old_mapping.shape)}, "
                f"got {tuple(new_mapping.shape)}"
            )
        rearrange_expert_weights_inplace(
            old_mapping,
            new_mapping,
            model_state.model.expert_weights,
            model_state.expert_buffer,
            get_ep_group().device_group,
            model_state.communicator,
        )
        _commit_eplb_maps(model_state, new_mapping.cpu())
        torch.accelerator.synchronize()
        return self.inspect_eplb()

    def swap_eplb_first_two(self, first=0, second=1):
        """Swap two physical slots in the current EPLB mapping."""
        return self.swap_eplb_pairs([first, second])

    def swap_eplb_pairs(self, pairs):
        """Apply several physical-slot swaps in one mapping update."""
        eplb_state = self.model_runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("EPLB state is not initialized")
        model_state = eplb_state.model_states[
            self.model_runner.model_config.compute_hash()
        ]
        mapping = model_state.physical_to_logical_map.detach().clone()
        if len(pairs) == 0 or len(pairs) % 2:
            raise ValueError("Expected a non-empty even list of swap slots")
        slots = list(pairs)
        if len(set(slots)) != len(slots):
            raise ValueError("Swap slots must be unique")
        if not all(0 <= slot < mapping.shape[-1] for slot in slots):
            raise ValueError("Swap slot is outside the physical mapping")
        for first, second in zip(slots[::2], slots[1::2]):
            if first == second:
                raise ValueError("Swap slots must be distinct")
            mapping[:, [first, second]] = mapping[:, [second, first]]
        return self.set_eplb_mapping(mapping.tolist())

    def probe_eplb_reload_gate(self, hold_seconds=0.2):
        """Exercise the reload/mapping critical section on every worker."""
        runner = self.model_runner
        state = runner.eplb_state
        if state is None:
            raise RuntimeError("EPLB is not enabled")

        mapping_started = threading.Event()
        release_mapping = threading.Event()
        reload_started = threading.Event()
        mapping_thread_error = []

        def hold_mapping():
            try:
                with state._mapping_guard():
                    mapping_started.set()
                    release_mapping.wait(timeout=hold_seconds)
            except BaseException as error:
                mapping_thread_error.append(error)

        mapping_thread = threading.Thread(target=hold_mapping)
        mapping_thread.start()
        if not mapping_started.wait(timeout=5):
            raise RuntimeError("EPLB mapping probe did not start")

        def begin_reload():
            runner.begin_weight_update()
            reload_started.set()

        reload_thread = threading.Thread(target=begin_reload)
        reload_thread.start()
        time.sleep(0.05)
        blocked = not reload_started.is_set()

        release_mapping.set()
        mapping_thread.join(timeout=5)
        reload_thread.join(timeout=5)
        if mapping_thread_error:
            raise mapping_thread_error[0]
        if not reload_started.is_set():
            raise RuntimeError("Reload gate did not open after mapping finished")

        mapping_rejected = False
        current_mapping = (
            state.model_states[runner.model_config.compute_hash()]
            .physical_to_logical_map.detach()
            .clone()
        )
        try:
            runner.setup_eplb_from_mapping(current_mapping)
        except RuntimeError as error:
            mapping_rejected = "during weight update" in str(error)
            if not mapping_rejected:
                raise
        finally:
            runner.finish_weight_update()

        return gather_rank_evidence(
            {
                "mapping_started": True,
                "reload_blocked_until_mapping_finished": blocked,
                "reload_started": True,
                "mapping_rejected_during_reload": mapping_rejected,
                "reload_finished": True,
            }
        )

    def inspect_memory(self):
        """Capture allocator state after a layerwise reload checkpoint."""
        torch.accelerator.synchronize()
        result = {
            "allocated": torch.accelerator.memory_allocated(),
            "reserved": torch.accelerator.memory_reserved(),
            "max_allocated": torch.accelerator.max_memory_allocated(),
            "max_reserved": torch.accelerator.max_memory_reserved(),
        }
        gc.collect()
        return gather_rank_evidence(result)

    def inspect_workspace(self):
        """Inspect per-rank vLLM scratch and persistent workspace state."""
        from vllm.v1.worker.workspace import current_workspace_manager

        manager = current_workspace_manager()
        workspaces = []
        for index, workspace in enumerate(manager._current_workspaces):
            if workspace is None:
                workspaces.append(None)
                continue
            workspaces.append(
                {
                    "index": index,
                    "id": id(workspace),
                    "ptr": workspace.data_ptr(),
                    "numel": workspace.numel(),
                    "nbytes": workspace.numel() * workspace.element_size(),
                    "dtype": str(workspace.dtype),
                }
            )
        persistent = []
        for index, resources in enumerate(manager._persistent_resources):
            entries = []
            for key, value in resources.items():
                item = {"key": repr(key), "class": type(value).__name__}
                if isinstance(value, torch.Tensor):
                    item.update(
                        {
                            "id": id(value),
                            "ptr": value.data_ptr(),
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "nbytes": value.numel() * value.element_size(),
                        }
                    )
                entries.append(item)
            persistent.append({"index": index, "entries": entries})
        return gather_rank_evidence(
            {
                "device": str(manager._device),
                "num_ubatches": manager._num_ubatches,
                "num_lanes": manager._num_lanes,
                "locked": manager.is_locked(),
                "workspaces": workspaces,
                "persistent": persistent,
            }
        )

    def inspect_model_parameters(self):
        """Inspect parameters and buffers, excluding frozen lookup tensors."""
        result = {}
        model = self.model_runner.model
        for name, parameter in chain(model.named_parameters(), model.named_buffers()):
            if getattr(parameter, "reload_frozen", False):
                continue
            value = parameter.detach()
            raw = value.contiguous().reshape(-1).view(torch.uint8)
            digest = hashlib.sha256()
            for start in range(0, raw.numel(), 8 * 1024 * 1024):
                digest.update(raw[start : start + 8 * 1024 * 1024].cpu().numpy())
            result[name] = {
                "id": id(parameter),
                "ptr": value.data_ptr(),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
                "hash": digest.hexdigest(),
            }
        assert result
        return gather_rank_evidence(result)

    def inspect_reload_trace(self, arm=False):
        trace = get_model_reload_tracer(self.model_runner.model)
        result = {}
        for key, state in trace.states.items():
            method = getattr(state.module, "quant_method", None)
            if arm and method is not None:

                def forbidden(*args, **kwargs):
                    raise AssertionError("Post-load processing called during reload")

                method.process_weights_after_loading = forbidden
            tensors = {}
            for role, target in state.targets.items():
                value = target.tensor
                raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
                tensors[role] = {
                    "id": id(value),
                    "ptr": value.data_ptr(),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "hash": hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest(),
                }
            result[key] = {
                "policy": type(state.policy).__name__,
                "method": id(method),
                "kernel": id(getattr(method, "moe_kernel", None)),
                "config": id(getattr(method, "moe_quant_config", None)),
                "runtime": {
                    "module_quant_method": type(method).__name__
                    if method is not None
                    else None,
                    "method_kernel": _object_summary(getattr(method, "kernel", None)),
                    "processing_plan": _object_summary(
                        getattr(method, "processing_plan", None)
                    ),
                    "moe_kernel": _object_summary(getattr(method, "moe_kernel", None)),
                    "moe_quant_config": _object_summary(
                        getattr(method, "moe_quant_config", None)
                    ),
                },
                "complete": state.complete,
                "staging": bool(state.checkpoint),
                "tensors": tensors,
            }
        assert result
        return gather_rank_evidence(result)

    def arm_forward_capture(self):
        """Record each invocation without overwriting earlier decode steps."""
        for hook in getattr(self, "_forward_hooks", []):
            hook.remove()
        self._forward_capture = {}
        self._forward_hooks = []
        self._forward_wrappers = []

        def summarize(value):
            if isinstance(value, torch.Tensor):
                return _tensor_summary(value)
            if isinstance(value, (tuple, list)):
                return [summarize(item) for item in value]
            if isinstance(value, dict):
                return {str(key): summarize(item) for key, item in value.items()}
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return {"class": type(value).__name__}

        model = self.model_runner.model

        def record(name, value):
            self._forward_capture.setdefault(name, []).append(
                {
                    "invocation": len(self._forward_capture.get(name, [])),
                    "value": summarize(value),
                }
            )

        # Capture raw and final routing. ``_select_experts`` is before EPLB
        # remapping; ``select_experts`` is the result consumed by the runner.
        for name, module in model.named_modules():
            router = getattr(module, "router", None)
            if router is None:
                continue

            raw_select = getattr(router, "_select_experts", None)
            if raw_select is not None:

                @wraps(raw_select)
                def raw_select_wrapper(*args, _select=raw_select, _name=name, **kwargs):
                    record(f"{_name}.router_raw_input", (args, kwargs))
                    result = _select(*args, **kwargs)
                    record(f"{_name}.router_raw_output", result)
                    return result

                router._reload_trace_original_select_experts = raw_select
                router._select_experts = raw_select_wrapper
                self._forward_wrappers.append(router)

            select = getattr(router, "select_experts", None)
            if select is None:
                continue

            @wraps(select)
            def select_wrapper(*args, _select=select, _name=name, **kwargs):
                record(f"{_name}.router_input", (args, kwargs))
                result = _select(*args, **kwargs)
                record(f"{_name}.router_output", result)
                return result

            router._reload_trace_original_select_experts_public = select
            router.select_experts = select_wrapper
            self._forward_wrappers.append(router)

        # RoutedExperts.forward_modular is the boundary immediately before
        # the quantization method invokes shared/routed expert computation.
        for name, module in model.named_modules():
            forward_modular = getattr(module, "forward_modular", None)
            if forward_modular is None or not name.endswith("routed_experts"):
                continue

            @wraps(forward_modular)
            def routed_wrapper(*args, _forward=forward_modular, _name=name, **kwargs):
                record(f"{_name}.input", (args, kwargs))
                output = _forward(*args, **kwargs)
                record(f"{_name}.output", output)
                return output

            module._reload_trace_original_forward_modular = forward_modular
            module.forward_modular = routed_wrapper
            self._forward_wrappers.append(module)

        # SharedExperts is held by MoERunner under a private attribute and
        # therefore is not reliably discoverable from a name-based filter.
        # Register hooks on both the wrapper and its actual MLP layer.
        for name, module in model.named_modules():
            shared = getattr(module, "_shared_experts", None)
            if shared is None:
                continue
            for shared_name, shared_module in (
                (f"{name}._shared_experts", shared),
                (f"{name}._shared_experts._layer", getattr(shared, "_layer", None)),
            ):
                if not isinstance(shared_module, torch.nn.Module):
                    continue

                def shared_input(_module, inputs, kwargs, name=shared_name):
                    records = self._forward_capture.setdefault(name, [])
                    records.append(
                        {
                            "invocation": len(records),
                            "inputs": summarize(inputs),
                            "kwargs": summarize(kwargs),
                        }
                    )

                def shared_output(_module, _inputs, output, name=shared_name):
                    self._forward_capture[name][-1]["output"] = summarize(output)

                self._forward_hooks.extend(
                    [
                        shared_module.register_forward_pre_hook(
                            shared_input, with_kwargs=True
                        ),
                        shared_module.register_forward_hook(shared_output),
                    ]
                )
        for name, module in model.named_modules():
            prefix = "language_model.model.layers."
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            parts = suffix.split(".")
            if not parts[0].isdigit():
                continue
            if len(parts) > 2 and parts[1] != "ffn":
                continue

            def capture_input(_module, inputs, kwargs, name=name):
                records = self._forward_capture.setdefault(name, [])
                # Hash before forward: kernels may overwrite input storage.
                records.append(
                    {
                        "invocation": len(records),
                        "inputs": summarize(inputs),
                        "kwargs": summarize(kwargs),
                    }
                )

            def capture(_module, _inputs, output, name=name):
                self._forward_capture[name][-1]["output"] = summarize(output)

            self._forward_hooks.extend(
                [
                    module.register_forward_pre_hook(capture_input, with_kwargs=True),
                    module.register_forward_hook(capture),
                ]
            )
        return len(self._forward_hooks)

    def read_forward_capture(self):
        for hook in getattr(self, "_forward_hooks", []):
            hook.remove()
        self._forward_hooks = []
        for module in getattr(self, "_forward_wrappers", []):
            original_select = getattr(
                module, "_reload_trace_original_select_experts", None
            )
            if original_select is not None:
                module._select_experts = original_select
                del module._reload_trace_original_select_experts
            original_public_select = getattr(
                module, "_reload_trace_original_select_experts_public", None
            )
            if original_public_select is not None:
                module.select_experts = original_public_select
                del module._reload_trace_original_select_experts_public
            original_forward = getattr(
                module, "_reload_trace_original_forward_modular", None
            )
            if original_forward is not None:
                module.forward_modular = original_forward
                del module._reload_trace_original_forward_modular
        self._forward_wrappers = []
        return gather_rank_evidence(getattr(self, "_forward_capture", {}))
