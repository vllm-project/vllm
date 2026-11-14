# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import msgspec
import torch

from vllm.config import VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.exceptions import LoRAAdapterNotFoundError
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.model_manager import (
    LoRAModelManager,
    LRUCacheLoRAModelManager,
    create_lora_manager,
)
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

logger = init_logger(__name__)


def _checkpoint_stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class WorkerLoRAManager:
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Every request, the requested LoRAs will be loaded (unless they are already
    loaded), and every other LoRA will be unloaded."""

    _manager_cls: type[LoRAModelManager] = LoRAModelManager

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        embedding_modules: dict[str, str],
        lora_model_cls: type[LoRAModel] = LoRAModel,
    ):
        self._lora_model_cls = lora_model_cls
        self.embedding_modules = embedding_modules
        self._cached_dummy_lora: None | Literal[False] | LoRAModel = False
        self._group_snapshot: LRUCacheLoRAModelManager.Snapshot | None = None
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        lora_config = vllm_config.lora_config
        if lora_config is None:
            raise ValueError("LoRA config must be set for WorkerLoRAManager.")
        self.lora_config: LoRAConfig = lora_config

        # Use get_text_config() in case of multimodal models
        text_config = vllm_config.model_config.hf_config.get_text_config()

        # For encoder-decoder models (e.g., Whisper), use max_target_positions
        # instead of max_position_embeddings
        # TODO: Generalize max_position_embeddings handling for
        # out-of-tree (OOT) encoder-decoder models
        if vllm_config.model_config.is_encoder_decoder:
            self.max_position_embeddings = getattr(
                text_config, "max_target_positions", None
            )
        else:
            self.max_position_embeddings = getattr(
                text_config, "max_position_embeddings", None
            )
        self.device = device
        # Lazily initialized by create_lora_manager.
        self._adapter_manager: LoRAModelManager

    @contextmanager
    def dummy_lora_cache(self):
        """Use this context manager to reuse the dummy lora model
        to avoid creating it repeatedly."""
        self._cached_dummy_lora = None
        yield
        self._cached_dummy_lora = False

    @property
    def is_enabled(self) -> bool:
        return True

    def create_lora_manager(
        self,
        model: torch.nn.Module,
        vllm_config: VllmConfig,
    ) -> Any:
        if vllm_config is None:
            raise ValueError("vllm_config must be provided to create a LoRA manager.")
        lora_manager = create_lora_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            device=self.device,
            lora_manager_cls=self._manager_cls,
            vllm_config=vllm_config,
        )
        self._adapter_manager = lora_manager
        return lora_manager.model

    def _load_adapter(self, lora_request: LoRARequest) -> LoRAModel:
        try:
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_lst: list[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_lst.extend(packed_modules_mapping[module])
                else:
                    expected_lora_lst.append(module)
                if module == "experts":
                    expected_lora_lst.append(module)
            expected_lora_modules = set(expected_lora_lst)
            lora_path = get_adapter_absolute_path(lora_request.lora_path)
            files = []
            stamps = []
            fingerprint = None
            if self.lora_config.enable_group_admission:
                directory = Path(lora_path)
                weights = next(
                    (
                        directory / name
                        for name in (
                            "adapter_model.safetensors",
                            "adapter_model.bin",
                            "adapter_model.pt",
                        )
                        if (directory / name).is_file()
                    ),
                    None,
                )
                if weights is None:
                    raise FileNotFoundError(f"No adapter weights in {directory}")
                files = [directory / "adapter_config.json", weights]
                stamps = [_checkpoint_stamp(path) for path in files]
                digest = hashlib.sha256()
                for path in files:
                    with path.open("rb") as stream:
                        digest.update(hashlib.file_digest(stream, "sha256").digest())
                fingerprint = digest.hexdigest()

            peft_helper = PEFTHelper.from_local_dir(
                lora_path, self.max_position_embeddings
            )

            # Validates the LoRA configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.lora_config)

            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper to ensure
            # correct loading of lora weights. We only need to know about renames for
            # this, so we use get_rename_mapper() to ignore stacking and deletions.
            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
            if hf_to_vllm_mapper is not None:
                hf_to_vllm_mapper = hf_to_vllm_mapper.get_rename_mapper()

            # Get model-defined prefixes to skip during LoRA loading.
            lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

            lora = self._lora_model_cls.from_local_checkpoint(
                lora_path,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                weights_mapper=hf_to_vllm_mapper,
                skip_prefixes=lora_skip_prefixes,
                moe_ep_spec=self._adapter_manager.moe_ep_load_spec,
                unsupported_lora_modules=self._adapter_manager.unsupported_lora_modules,
            )
            # Stamp the on-disk MoE layout onto the loaded model so the
            # adapter manager can route 3D-format checkpoints through the
            # 3D->2D conversion when running under the universal 2D wrapper.
            lora.is_3d_lora_weight = lora_request.is_3d_lora_weight
            if fingerprint is not None:
                if any(
                    _checkpoint_stamp(path) != stamp
                    for path, stamp in zip(files, stamps)
                ):
                    raise RuntimeError("Adapter checkpoint changed while loading")
                lora.source_fingerprint = fingerprint
            self._adapter_manager._validate_moe_lora_format(lora)

            # Validate classification-head weights.
            self._adapter_manager._validate_classification_head_lora(lora)

        except FileNotFoundError as e:
            # FileNotFoundError should be raised if both
            # - No adapter found to download from huggingface (or in
            #       offline mode)
            # - No local adapter files found at `lora_request.lora_path`
            # For NotFoundError
            raise LoRAAdapterNotFoundError(
                lora_request.lora_name, lora_request.lora_path
            ) from e
        except Exception as e:
            raise e

        return lora

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        if lora_request.lora_int_id in self.list_adapters():
            return False
        if isinstance(self._cached_dummy_lora, LoRAModel):
            dummy_lora = self._cached_dummy_lora.clone(lora_request.lora_int_id)
        else:
            dummy_lora = self._adapter_manager.create_dummy_lora(
                lora_request.lora_int_id, rank, self.embedding_modules
            )
            if self._cached_dummy_lora is None:
                self._cached_dummy_lora = dummy_lora
        return self._adapter_manager.add_adapter(dummy_lora)

    def get_dummy_lora_warmup_rank(self, default_rank: int) -> int:
        return self._adapter_manager.get_dummy_lora_warmup_rank(default_rank)

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: set[Any], mapping: Any | None) -> None:
        self._apply_adapters(requests)
        if mapping is not None:
            self._adapter_manager.set_adapter_mapping(mapping)

    def supports_tower_connector_lora(self) -> bool:
        return (
            self._adapter_manager.supports_mm
            and self._adapter_manager.supports_tower_connector_lora
        )

    def _apply_adapters(self, adapter_requests: set[Any]) -> None:
        existing_adapters = self.list_adapters()
        models_map = {
            adapter_request.adapter_id: adapter_request
            for adapter_request in adapter_requests
            if adapter_request
        }
        if len(models_map) > self._adapter_manager.adapter_slots:
            raise RuntimeError(
                f"Number of requested models ({len(models_map)}) is greater "
                "than the number of GPU model slots "
                f"({self._adapter_manager.adapter_slots})."
            )
        requested_ids = set(models_map)
        for adapter_id in existing_adapters - requested_ids:
            self.remove_adapter(adapter_id)
        for adapter_id in requested_ids - existing_adapters:
            self.add_adapter(models_map[adapter_id])

    def add_adapter(self, adapter_request: Any) -> bool:
        if adapter_request.adapter_id in self.list_adapters():
            return False
        # One-time per adapter
        with gpu_sync_allowed():
            loaded_adapter = self._load_adapter(adapter_request)
            loaded = self._adapter_manager.add_adapter(loaded_adapter)
            self._adapter_manager.activate_adapter(loaded_adapter.id)
        return loaded

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> set[int]:
        return set(self._adapter_manager.list_adapters())

    def group_update(self, method: str, argument: Any) -> dict:
        """Apply a retired-epoch mutation and acknowledge actual resident weights."""
        manager = self._adapter_manager
        assert isinstance(manager, LRUCacheLoRAModelManager)
        previous = None
        try:
            if method == "commit":
                self._group_snapshot = None
                return {"error": None, "fingerprints": {}, "result": True}
            if method == "rollback":
                assert self._group_snapshot is not None
                manager.restore_adapters(self._group_snapshot)
                return {
                    "error": None,
                    "fingerprints": {},
                    "result": True,
                    "state": self.group_adapter_state(),
                }
            if self._group_snapshot is not None:
                raise RuntimeError("Previous group adapter transaction is unfinished")
            self._group_snapshot = manager.snapshot_adapters()
            previous = self.group_adapter_state()
            if method == "prepare":
                requests = [
                    msgspec.msgpack.decode(item, type=LoRARequest) for item in argument
                ]
                desired = {request.lora_int_id for request in requests}
                if not manager._active_adapters.pinned_items <= desired:
                    raise RuntimeError(
                        "A pinned adapter was omitted from the group epoch"
                    )
                self.set_active_adapters(set(requests), None)
                result = True
            elif method == "add_lora":
                request = msgspec.msgpack.decode(argument, type=LoRARequest)
                result = self.add_adapter(request)
                desired = {request.lora_int_id}
            elif method == "remove_lora":
                result = self.remove_adapter(argument)
                desired = set()
            elif method == "pin_lora":
                result = self.pin_adapter(argument)
                desired = {argument}
            else:
                raise ValueError(f"Unknown group LoRA operation: {method}")
            fingerprints = {}
            for adapter in desired:
                model = self._adapter_manager.get_adapter(adapter)
                if (
                    model is None
                    or model.source_fingerprint is None
                    or adapter not in self._adapter_manager.lora_index_to_id
                ):
                    raise RuntimeError(f"Adapter {adapter} is not ready on this worker")
                fingerprints[adapter] = model.source_fingerprint
            return {
                "error": None,
                "fingerprints": fingerprints,
                "result": result,
                "previous": previous,
            }
        except Exception as error:
            return {
                "error": str(error),
                "fingerprints": {},
                "result": False,
                "previous": previous,
            }

    def group_adapter_state(self) -> dict:
        manager = self._adapter_manager
        assert isinstance(manager, LRUCacheLoRAModelManager)
        state = manager.snapshot_adapters()
        return {
            "models": [
                (key, model.source_fingerprint) for key, model in state.models.items()
            ],
            "active": list(state.active),
            "cpu_pinned": sorted(state.cpu_pinned),
            "gpu_pinned": sorted(state.gpu_pinned),
        }


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""

    _manager_cls: type[LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    def create_lora_manager(
        self,
        model: torch.nn.Module,
        vllm_config: VllmConfig,
    ) -> Any:
        if vllm_config is None:
            raise ValueError("vllm_config must be provided to create a LoRA manager.")
        lora_manager = create_lora_manager(
            model,
            lora_manager_cls=self._manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            device=self.device,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vllm_config=vllm_config,
        )
        self._adapter_manager = lora_manager
        return lora_manager.model

    def _apply_adapters(self, lora_requests: set[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests
            if lora_request
        }
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots})."
            )
        for lora in loras_map.values():
            self.add_adapter(lora)

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        # Note that this method is not thread-safe. It may be invoked multiple
        # times for the same adapter when using multiple API servers.
        # This is ok because it's currently only called from
        # the single-threaded core engine loop.

        with gpu_sync_allowed():
            if (
                lora_request.lora_int_id not in self.list_adapters()
                or lora_request.load_inplace
            ):
                # Load the new adapter first to ensure it is actually valid, before
                # evicting any existing adapters.
                # This may cause the # of loaded lora adapters to very temporarily
                # exceed `--max-cpu-loras`.
                lora = self._load_adapter(lora_request)

                # Remove the existing adapter if it exists
                # Use case for LoRA inplace
                self._adapter_manager.remove_adapter(lora.id)

                # Loading succeeded, now check if we will exceed cache capacity and
                # evict if the oldest adapter if so
                if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                    assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                    self._adapter_manager.remove_oldest_adapter()
                # Then add the new adapter to the cache
                loaded = self._adapter_manager.add_adapter(lora)
            else:
                # If the lora is already loaded, just touch it to
                # update its position in the caches
                loaded = (
                    self._adapter_manager.get_adapter(lora_request.lora_int_id)
                    is not None
                )
            self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded
