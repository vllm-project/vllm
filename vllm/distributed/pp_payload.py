# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional tensors carried alongside pipeline-parallel model activations."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar, overload

import torch

from vllm.sequence import IntermediateTensors

PP_SIDECAR_PREFIX = "__vllm_pp_sidecar__/"
AUX_HIDDEN_STATES_NAMESPACE = "aux_hidden_states"
TOPK_INDICES_NAMESPACE = "topk_indices"
TOPK_INDICES_NAME = "buffer"

_T = TypeVar("_T")


class _SidecarAllGatherPolicy(dict[str, bool]):
    """Keep sidecars on their matching TP lane during PP communication."""

    @overload
    def get(self, key: str, default: None = None, /) -> bool | None: ...

    @overload
    def get(self, key: str, default: bool, /) -> bool: ...

    @overload
    def get(self, key: str, default: _T, /) -> bool | _T: ...

    def get(self, key: str, default: _T | None = None, /) -> bool | _T | None:
        if key.startswith(PP_SIDECAR_PREFIX) and key not in self:
            return False
        return super().get(key, default)


@dataclass
class PPForwardPayload:
    """Model activations and optional sidecars for one PP microbatch.

    Sidecars use the existing PP tensor-dict transport, but are kept outside
    the next stage's model inputs. This lets features carry tensors across
    several PP stages without changing each model's intermediate tensor schema.
    """

    intermediate_tensors: IntermediateTensors
    sidecars: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)

    @staticmethod
    def _wire_key(namespace: str, name: str) -> str:
        if not namespace or not name or "/" in namespace or "/" in name:
            raise ValueError(
                "PP sidecar namespace and name must be non-empty and cannot "
                f"contain '/': {(namespace, name)!r}"
            )
        return f"{PP_SIDECAR_PREFIX}{namespace}/{name}"

    @classmethod
    def from_intermediate_tensors(
        cls, intermediate_tensors: IntermediateTensors
    ) -> "PPForwardPayload":
        model_tensors: dict[str, torch.Tensor] = {}
        sidecars: dict[tuple[str, str], torch.Tensor] = {}
        for key, tensor in intermediate_tensors.tensors.items():
            if not key.startswith(PP_SIDECAR_PREFIX):
                model_tensors[key] = tensor
                continue

            parts = key.removeprefix(PP_SIDECAR_PREFIX).split("/")
            if len(parts) != 2 or not all(parts):
                raise ValueError(f"Malformed PP sidecar key: {key!r}")
            sidecar_key = (parts[0], parts[1])
            if sidecar_key in sidecars:
                raise ValueError(f"Duplicate PP sidecar: {sidecar_key!r}")
            sidecars[sidecar_key] = tensor
        return cls(IntermediateTensors(model_tensors), sidecars)

    def to_intermediate_tensors(self) -> IntermediateTensors:
        tensors = dict(self.intermediate_tensors.tensors)
        for (namespace, name), tensor in self.sidecars.items():
            key = self._wire_key(namespace, name)
            if key in tensors:
                raise ValueError(f"Duplicate PP tensor key: {key!r}")
            tensors[key] = tensor
        return IntermediateTensors(tensors)

    def add_tensor(
        self,
        namespace: str,
        name: str,
        tensor: torch.Tensor,
        *,
        replace: bool = False,
    ) -> None:
        self._wire_key(namespace, name)
        key = (namespace, name)
        if key in self.sidecars and not replace:
            raise ValueError(f"Duplicate PP sidecar: {key!r}")
        self.sidecars[key] = tensor

    def pop_tensor(
        self, namespace: str, name: str, default: _T | None = None
    ) -> torch.Tensor | _T | None:
        return self.sidecars.pop((namespace, name), default)

    def carry(self, next_stage_tensors: IntermediateTensors) -> "PPForwardPayload":
        output = self.from_intermediate_tensors(next_stage_tensors)
        for (namespace, name), tensor in self.sidecars.items():
            output.add_tensor(namespace, name, tensor)
        return output

    def discard_intermediate_tensors(self) -> None:
        self.intermediate_tensors = IntermediateTensors({})

    def add_aux_hidden_states(
        self, layer_ids: Sequence[int], hidden_states: Sequence[torch.Tensor]
    ) -> None:
        if len(layer_ids) != len(hidden_states):
            raise ValueError(
                "Aux hidden-state layer and tensor counts differ: "
                f"{len(layer_ids)} != {len(hidden_states)}"
            )
        for layer_id, hidden_state in zip(layer_ids, hidden_states):
            if layer_id < 0:
                raise ValueError(
                    f"Aux hidden-state layer id must be non-negative: {layer_id}"
                )
            self.add_tensor(AUX_HIDDEN_STATES_NAMESPACE, str(layer_id), hidden_state)

    def pop_aux_hidden_states(self) -> dict[int, torch.Tensor]:
        result: dict[int, torch.Tensor] = {}
        for namespace, name in list(self.sidecars):
            if namespace != AUX_HIDDEN_STATES_NAMESPACE:
                continue
            try:
                layer_id = int(name)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid aux hidden-state layer id: {name!r}"
                ) from exc
            if str(layer_id) != name or layer_id < 0:
                raise ValueError(f"Invalid aux hidden-state layer id: {name!r}")
            result[layer_id] = self.sidecars.pop((namespace, name))
        return result

    def set_topk_indices(self, tensor: torch.Tensor) -> None:
        self.add_tensor(
            TOPK_INDICES_NAMESPACE,
            TOPK_INDICES_NAME,
            tensor,
            replace=True,
        )

    def pop_topk_indices(self) -> torch.Tensor | None:
        tensor = self.pop_tensor(TOPK_INDICES_NAMESPACE, TOPK_INDICES_NAME)
        assert tensor is None or isinstance(tensor, torch.Tensor)
        return tensor

    def copy_topk_indices_to(
        self, topk_indices_buffer: torch.Tensor, num_tokens: int
    ) -> None:
        topk_indices = self.pop_topk_indices()
        if topk_indices is None:
            return
        expected_shape = topk_indices_buffer[:num_tokens].shape
        if topk_indices.shape != expected_shape:
            raise ValueError(
                "PP top-k indices shape does not match the local buffer: "
                f"{tuple(topk_indices.shape)} != {tuple(expected_shape)}"
            )
        topk_indices_buffer[:num_tokens].copy_(topk_indices)

    @staticmethod
    def make_all_gather_policy(
        overrides: dict[str, bool] | None = None,
    ) -> dict[str, bool]:
        return _SidecarAllGatherPolicy(overrides or {})


def add_pp_aux_hidden_states(
    intermediate_tensors: IntermediateTensors,
    layer_ids: Sequence[int],
    hidden_states: Sequence[torch.Tensor],
) -> IntermediateTensors:
    """Attach locally produced aux states to a model's PP output."""

    payload = PPForwardPayload.from_intermediate_tensors(intermediate_tensors)
    payload.add_aux_hidden_states(layer_ids, hidden_states)
    return payload.to_intermediate_tensors()


def merge_pp_aux_hidden_states(
    payload: PPForwardPayload | None,
    layer_ids: Sequence[int],
    local_hidden_states: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Merge received and local aux states in configured global-layer order."""

    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError(f"Duplicate aux hidden-state layer ids: {layer_ids}")

    hidden_states_by_layer = {} if payload is None else payload.pop_aux_hidden_states()
    unexpected = set(hidden_states_by_layer).difference(layer_ids)
    if unexpected:
        raise ValueError(
            f"Received aux hidden states for unconfigured layers: {sorted(unexpected)}"
        )

    local_layer_ids = [
        layer_id for layer_id in layer_ids if layer_id not in hidden_states_by_layer
    ]
    if len(local_layer_ids) != len(local_hidden_states):
        raise ValueError(
            "Local aux hidden-state count does not match layers owned by the "
            f"last PP stage: {len(local_hidden_states)} != {len(local_layer_ids)}"
        )
    hidden_states_by_layer.update(zip(local_layer_ids, local_hidden_states))

    missing = [
        layer_id for layer_id in layer_ids if layer_id not in hidden_states_by_layer
    ]
    if missing:
        raise ValueError(f"Missing aux hidden states for layers: {missing}")
    return [hidden_states_by_layer[layer_id] for layer_id in layer_ids]
