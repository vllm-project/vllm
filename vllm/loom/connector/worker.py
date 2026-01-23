# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from collections import defaultdict
from pathlib import Path

import torch

from vllm.attention.layer import Attention
from vllm.config import get_layers_from_vllm_config
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec

from .metadata import ReqId, LoomConnectorMetadata
from ..logger import get_loom_logger

logger = get_loom_logger(__name__)


def _get_forward_context_arg(_forward_context: object) -> None:
    # Avoid importing ForwardContext at runtime.
    return None


class LoomConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()

        loom_cfg = getattr(spec, "loom_config", None)
        self._loom_load_only: bool = bool(getattr(loom_cfg, "loom_load_only", False))

        self._shared_prefix_kvcache_path: str | None = None
        self._shared_prefix_layout_version: int = 1
        if loom_cfg is not None:
            path_raw = getattr(loom_cfg, "shared_prefix_kvcache_path", None)
            if path_raw is not None:
                self._shared_prefix_kvcache_path = str(path_raw)
            layout_raw = getattr(loom_cfg, "shared_prefix_layout_version", None)
            if layout_raw is not None:
                self._shared_prefix_layout_version = int(layout_raw)

        if self._shared_prefix_kvcache_path is None:
            extra = getattr(spec, "extra_config", None)
            if isinstance(extra, Mapping):
                path_raw = extra.get("shared_prefix_kvcache_path")
                if path_raw is not None:
                    self._shared_prefix_kvcache_path = str(path_raw)
                layout_raw = extra.get("shared_prefix_layout_version")
                if layout_raw is not None:
                    self._shared_prefix_layout_version = int(layout_raw)

        self._shared_prefix_ingested: bool = False

        self._job_counter = 0

        # req_id -> (job_id, store)
        self._jobs: dict[int, tuple[ReqId, bool]] = {}
        # req_id -> active job IDs
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active job IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)
        # list of store jobs pending submission (job_id, transfer_spec)
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

        self._shared_prefix_manager = self.spec.get_manager()

    def _normalize_kv_for_seq_len(
        self,
        tensor: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(tensor).__name__}")
        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")

        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D KV tensor, got shape={tuple(tensor.shape)}")
        if tensor.shape[0] != 1:
            raise ValueError(
                "Only batch_size=1 is supported for shared prefix ingest, got "
                f"shape={tuple(tensor.shape)}"
            )

        if tensor.shape[1] == seq_len:
            return tensor[0].contiguous()
        if tensor.shape[2] == seq_len:
            return tensor[0].permute(1, 0, 2).contiguous()

        raise ValueError(
            "Unsupported KV tensor layout for shared prefix ingest: "
            f"shape={tuple(tensor.shape)} seq_len={seq_len}"
        )

    def _maybe_ingest_shared_prefixes(self) -> None:
        if self._shared_prefix_ingested:
            return
        if self._shared_prefix_kvcache_path is None:
            return

        cxl_handlers = getattr(self.spec, "_cxl_handlers", None)
        cxl_tensors = getattr(cxl_handlers, "cxl_tensors", None)
        if not isinstance(cxl_tensors, list) or not cxl_tensors:
            return
        if len(cxl_tensors) != 1:
            raise RuntimeError(
                "Shared prefix ingest currently requires a single cross-layer KV cache tensor; "
                f"got {len(cxl_tensors)} CXL tensors"
            )
        cxl_tensor = cxl_tensors[0]
        if cxl_tensor.device.type != "cpu":
            raise RuntimeError(
                "Expected CPU CXL tensor for shared prefix ingest, got device="
                f"{cxl_tensor.device}"
            )

        manager = self._shared_prefix_manager
        allocate_extent = getattr(manager, "allocate_shared_prefix_extent", None)
        if allocate_extent is None:
            raise RuntimeError(
                "Offloading manager does not support shared prefix extent allocation"
            )

        path = Path(self._shared_prefix_kvcache_path)
        if not path.exists():
            raise FileNotFoundError(f"shared_prefix_kvcache_path not found: {path}")

        kvache_obj = torch.load(path, map_location="cpu")
        if not isinstance(kvache_obj, dict) or kvache_obj.get("format") != "kvcache_prefill_v1":
            raise ValueError(
                "Unsupported shared prefix artifact format (expect kvcache_prefill_v1)"
            )

        prefix_len = int(kvache_obj["prefix_len"])
        block_size_in_artifact = kvache_obj.get("kv_cache_block_size")
        if block_size_in_artifact is not None:
            block_size_in_artifact = int(block_size_in_artifact)
            if block_size_in_artifact != int(self.spec.offloaded_block_size):
                raise ValueError(
                    "shared prefix artifact kv_cache_block_size must match offloaded_block_size: "
                    f"artifact={block_size_in_artifact} offloaded_block_size={self.spec.offloaded_block_size}"
                )

        if prefix_len % int(self.spec.offloaded_block_size) != 0:
            raise ValueError(
                "shared prefix artifact prefix_len must be block-aligned to offloaded_block_size: "
                f"prefix_len={prefix_len} offloaded_block_size={self.spec.offloaded_block_size}"
            )

        loom_cfg = getattr(self.spec, "loom_config", None)
        if loom_cfg is None:
            raise RuntimeError("Missing spec.loom_config for Loom shared prefix ingest")
        layer_group_size = int(getattr(loom_cfg, "layer_group_size", 0) or 0)
        if layer_group_size <= 0:
            raise ValueError(f"Invalid layer_group_size={layer_group_size} for shared prefix ingest")

        num_layers = int(cxl_tensor.shape[1])
        if num_layers % layer_group_size != 0:
            raise ValueError(
                "num_layers must be divisible by layer_group_size for shared prefix ingest: "
                f"num_layers={num_layers} layer_group_size={layer_group_size}"
            )

        gpu_block_size = int(cxl_tensor.shape[3])
        offloaded_block_size = int(self.spec.offloaded_block_size)
        if offloaded_block_size % gpu_block_size != 0:
            raise ValueError(
                "offloaded_block_size must be divisible by gpu_block_size for shared prefix ingest: "
                f"offloaded_block_size={offloaded_block_size} gpu_block_size={gpu_block_size}"
            )
        f_per_offloaded = offloaded_block_size // gpu_block_size
        chunk_num = prefix_len // offloaded_block_size

        prefixes = kvache_obj.get("prefixes")
        if not isinstance(prefixes, list):
            raise ValueError("shared prefix artifact is missing 'prefixes' list")

        dtype = cxl_tensor.dtype
        num_groups = num_layers // layer_group_size
        for prefix in prefixes:
            if not isinstance(prefix, dict):
                raise TypeError(f"Invalid prefix entry type: {type(prefix).__name__}")
            prefix_id = int(prefix["prefix_id"])
            past_key_values = prefix.get("past_key_values")
            if past_key_values is None:
                raise ValueError(f"prefix_id={prefix_id} missing past_key_values")

            for layer_group_id in range(num_groups):
                extent = allocate_extent(
                    prefix_id=prefix_id,
                    layer_group_id=layer_group_id,
                    num_blocks=chunk_num,
                    layout_version=self._shared_prefix_layout_version,
                )
                base_block_id = int(getattr(extent, "base_block_id"))

                for lg in range(layer_group_size):
                    layer_idx = layer_group_id * layer_group_size + lg
                    layer_past = past_key_values[layer_idx]
                    if not (
                        isinstance(layer_past, (tuple, list)) and len(layer_past) == 2
                    ):
                        raise TypeError(
                            "Unsupported past_key_values layer type for shared prefix ingest: "
                            f"{type(layer_past).__name__}"
                        )
                    k_raw, v_raw = layer_past
                    k = self._normalize_kv_for_seq_len(k_raw, prefix_len).to(dtype=dtype)
                    v = self._normalize_kv_for_seq_len(v_raw, prefix_len).to(dtype=dtype)

                    for c in range(chunk_num):
                        for f in range(f_per_offloaded):
                            gpu_block_idx = (base_block_id + c) * f_per_offloaded + f
                            token_start = c * offloaded_block_size + f * gpu_block_size
                            token_end = token_start + gpu_block_size
                            cxl_tensor[gpu_block_idx, layer_idx, 0].copy_(
                                k[token_start:token_end]
                            )
                            cxl_tensor[gpu_block_idx, layer_idx, 1].copy_(
                                v[token_start:token_end]
                            )

        logger.debug(
            "Ingested %d shared prefixes into CXL/DRAM pinned buffers from %s",
            len(prefixes),
            str(path),
        )
        self._shared_prefix_ingested = True

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def _register_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        for src_cls, dst_cls, handler in self.spec.get_handlers(
            kv_caches, attn_backends
        ):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert False, "Never called"
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config, Attention, layer_names
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }
        self._register_handlers(kv_caches, attn_backends)
        self._maybe_ingest_shared_prefixes()

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        cross_layer_name = "ALL_LAYERS"
        kv_caches = {cross_layer_name: kv_cache}
        attn_backends = {cross_layer_name: attn_backend}
        self._register_handlers(kv_caches, attn_backends)
        self._maybe_ingest_shared_prefixes()

    def handle_preemptions(self, preempted_req_ids: set[str]):
        if self._loom_load_only:
            return
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for req_id in preempted_req_ids:
            job_ids = self._store_jobs.get(req_id)
            if job_ids:
                self.worker.wait(job_ids)

    def start_kv_transfers(self, metadata: LoomConnectorMetadata, forward_context: object):
        _get_forward_context_arg(forward_context)
        if not self._loom_load_only:
            for job_id, transfer_spec in self._unsubmitted_store_jobs:
                success = self.worker.transfer_async(job_id, transfer_spec)
                assert success
            self._unsubmitted_store_jobs.clear()

        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, False)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success

    def prepare_store_kv(self, metadata: LoomConnectorMetadata):
        if self._loom_load_only:
            return
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            # NOTE(orozery): defer the store to the beginning of the next engine step,
            # so that offloading starts AFTER transfers related to token sampling,
            # thereby avoiding delays to token generation due to offloading.
            self._unsubmitted_store_jobs.append((job_id, transfer_spec))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_sending = set()
        finished_recving = set()
        if self._loom_load_only:
            finished_req_ids = set()
        for job_id, success in self.worker.get_finished():
            # we currently do not support job failures
            assert success
            req_id, store = self._jobs.pop(job_id)
            if store:
                if self._loom_load_only:
                    continue
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if req_jobs:
                    continue

                if req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                req_job = self._load_job[req_id]
                assert job_id == req_job
                del self._load_job[req_id]
                finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_jobs = self._store_jobs.get(req_id)
            if pending_req_jobs:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_jobs is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving
