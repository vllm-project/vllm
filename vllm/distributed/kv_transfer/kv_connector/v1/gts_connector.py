# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import os
import time
import json
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp, KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

try:
    import gts
    GTS_AVAILABLE = True
except ImportError:
    GTS_AVAILABLE = False
    gts = None

logger = init_logger(__name__)


# Data Model following the proposed structure
# Using torch.dtype directly instead of custom enum


@dataclass
class Range:
    start: int
    end: int


# Type aliases matching the proposed data model
TensorName = str  # e.g., "prefill:0/layer:0/kv"
DimensionMap = Dict[str, int]  # {dimension_name → size}
OwnershipMap = Dict[str, Range]  # {dim_name → [start, end]}
IndexSet = Set[int]
SliceSpec = Dict[str, Union[Range, IndexSet]]


@dataclass
class TensorMetadata:
    """Metadata for a tensor following the proposed data model"""
    name: TensorName
    dtype: torch.dtype  # Use torch.dtype directly
    dims: DimensionMap
    layout: Optional[str] = None  # e.g., "HND", "NHD"


@dataclass
class CopyIntent:
    """Intent to copy tensor data with slice specifications"""
    src: TensorName
    src_slice: SliceSpec
    dst: TensorName
    dst_slice: SliceSpec


@dataclass 
class GTSShardInfo:
    """Information about a tensor shard owned by this worker"""
    tensor_name: TensorName
    metadata: TensorMetadata
    ownership: OwnershipMap  # Which dimensions/ranges this worker owns
    local_tensor: Optional[torch.Tensor] = None
    gts_handle: Optional[Any] = None


@dataclass
class GTSPutTask:
    """Represents a put operation task"""
    copy_intent: CopyIntent
    future: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class GTSConnectorMetadata(KVConnectorMetadata):
    """Metadata for GTS operations following collective design"""
    
    def __init__(self):
        # Requests that need to save KV cache
        self.reqs_to_save: Dict[str, Dict[str, Any]] = {}
        # Engine identification
        self.engine_id: str = ""
        self.tp_rank: int = 0
        self.tp_size: int = 0
        self.num_layers: int = 0
        self.kv_cache_layout: str = ""


class GTSShardManager:
    """
    Manages tensor shards for this worker in the collective GTS design.
    Each worker owns a shard of the global tensor and registers it with
    a common name for collective operations.
    """
    
    def __init__(self, client: 'gts.Client', tp_rank: int, tp_size: int):
        self.client = client
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.registered_shards: Dict[str, GTSShardInfo] = {}
        self.active_puts: Dict[str, GTSPutTask] = {}
        
    def register_layer_shard(self, layer_name: str, kv_cache: torch.Tensor,
                             kv_cache_layout: str, block_size: int, 
                             engine_id: str) -> None:
        """
        Register this worker's shard of a layer's KV cache.
        Uses a common tensor name across all workers for collective operations.
        """
        # Hierarchical tensor name following the proposed format
        common_tensor_name = f"{engine_id}/layer:{layer_name}/kv"
        
        # Parse tensor dimensions
        if kv_cache_layout == "HND":
            num_blocks, num_pairs, block_tokens, num_heads, head_dim = kv_cache.shape
        else:  # NHD
            num_blocks, num_pairs, num_heads, block_tokens, head_dim = kv_cache.shape
        
        # Build dimension map
        dims = {
            "blocks": num_blocks,
            "kv": num_pairs, 
            "tokens": block_tokens,
            "heads": num_heads,
            "head_dim": head_dim
        }
        
        # Use vLLM's proper ownership calculation
        # The kv_cache tensor already represents this worker's shard
        # num_heads is already the per-worker count from get_num_kv_heads()
        ownership = {
            "blocks": Range(0, num_blocks),   # Own all blocks
            "kv": Range(0, num_pairs),        # Own all kv pairs  
            "tokens": Range(0, block_tokens), # Own all tokens
            "heads": Range(0, num_heads),     # Own our assigned heads (already sharded)
            "head_dim": Range(0, head_dim)    # Own all head dimensions
        }
        
        # Create tensor metadata using torch.dtype directly
        tensor_metadata = TensorMetadata(
            name=common_tensor_name,
            dtype=kv_cache.dtype,  # Use torch.dtype directly
            dims=dims,
            layout=kv_cache_layout
        )
        
        # Register the shard with GTS using common name
        # We need to calculate global dimensions for GTS coordination
        global_num_heads = num_heads * self.tp_size  # Reconstruct global head count
        local_head_start = self.tp_rank * num_heads  # Our head range start
        local_head_end = (self.tp_rank + 1) * num_heads  # Our head range end
        
        gts_handle = self.client.create_and_register(
            name=common_tensor_name,
            dims=[
                ("blocks", num_blocks, (0, num_blocks)),
                ("kv", num_pairs, (0, num_pairs)),
                ("tokens", block_tokens, (0, block_tokens)),
                ("heads", global_num_heads, (local_head_start, local_head_end)),  # Global dims, local range
                ("head_dim", head_dim, (0, head_dim))
            ],
            dtype=kv_cache.dtype,
            device=str(kv_cache.device),
            data=kv_cache  # Local shard data
        )
        
        # Store shard info using new data model
        shard_info = GTSShardInfo(
            tensor_name=common_tensor_name,
            metadata=tensor_metadata,
            ownership=ownership,
            local_tensor=kv_cache,
            gts_handle=gts_handle
        )
        
        self.registered_shards[layer_name] = shard_info
        
        logger.info(f"Worker {self.tp_rank} registered shard {ownership['heads']} "
                   f"of {common_tensor_name}")
    
    def put_request_shard(self, request_id: str, layer_name: str, 
                         blocks: List[int], kv_cache: torch.Tensor,
                         metadata: Dict[str, Any]) -> 'gts.Future':
        """
        Put this worker's shard of a request's KV cache.
        Uses collective tensor naming for cross-worker coordination.
        """
        # Build tensor name using real vLLM metadata
        engine_id = metadata.get('engine_id', 'unknown')
        kv_role = metadata.get('kv_role', 'unknown')
        kv_rank = metadata.get('kv_rank', 0)
        
        # Role-first naming for GTS routing between prefill/decode
        common_tensor_name = (f"role:{kv_role}/"
                             f"engine:{engine_id}/"
                             f"rank:{kv_rank}/"
                             f"request:{request_id}/"
                             f"layer:{layer_name}")
        
        # Parse dimensions
        if metadata['kv_cache_layout'] == "HND":
            _, num_pairs, block_tokens, num_heads, head_dim = kv_cache.shape
        else:  # NHD
            _, num_pairs, num_heads, block_tokens, head_dim = kv_cache.shape
        
        # Calculate this worker's shard
        heads_per_rank = num_heads // self.tp_size
        shard_start = self.tp_rank * heads_per_rank
        shard_end = (self.tp_rank + 1) * heads_per_rank
        
        # Build selectors for put operation
        # Source: select specific blocks from local KV cache
        src_selector = self.client.tensor(kv_cache) \
            .where(blocks=blocks) \
            .where(kv=gts.ALL) \
            .where(tokens=gts.ALL) \
            .where(heads=gts.slice(0, shard_end - shard_start)) \
            .where(head_dim=gts.ALL) \
            .build()
        
        # Destination: just specify the target role
        # GTS infers request/layer from source automatically
        target_role = "kv_consumer" if kv_role == "kv_producer" else "kv_producer" 
        dst = self.client.tensor(f"role:{target_role}")
        
        # Perform collective put (GTS handles coordination)
        future = self.client.copy(src_selector, dst,
                                options={"async": True, "collective": True})
        
        # Track the put operation
        copy_intent = CopyIntent(
            src=common_tensor_name,
            src_slice=src_selector,
            dst=f"role:{target_role}",
            dst_slice={}
        )
        
        task = GTSPutTask(
            copy_intent=copy_intent,
            future=future,
            metadata=metadata
        )
        
        task_key = f"{request_id}_{layer_name}"
        self.active_puts[task_key] = task
        
        return future
    
    def wait_for_puts(self, layer_name: Optional[str] = None) -> None:
        """Wait for put operations to complete"""
        if layer_name:
            # Wait for specific layer
            tasks_to_wait = [
                (key, task) for key, task in self.active_puts.items()
                if task.layer_name == layer_name
            ]
        else:
            # Wait for all
            tasks_to_wait = list(self.active_puts.items())
        
        for key, task in tasks_to_wait:
            if task.future:
                task.future.wait()
                del self.active_puts[key]
    
    def get_finished_puts(self, request_ids: Set[str]) -> Set[str]:
        """Check which put operations have completed"""
        finished = set()
        
        for key, task in list(self.active_puts.items()):
            if task.request_id in request_ids and task.future:
                if task.future.is_ready():
                    finished.add(task.request_id)
                    del self.active_puts[key]
        
        return finished


class GTSConnectorImpl:
    """
    Implementation for GTS connector with collective tensor design.
    Focuses on put-only operations with worker-specific shards.
    """
    
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole,
                 connector: 'GTSConnector'):
        self.vllm_config = vllm_config
        self.role = role
        self.connector = connector
        
        # Initialize GTS client
        # Get server address from environment variable or use default
        gts_server = os.environ.get('GTS_SERVER_ADDRESS', 'localhost:6174')
        # Also check if it's in kv_transfer_config (for future use)
        if hasattr(vllm_config.kv_transfer_config, 'gts_server_address'):
            gts_server = vllm_config.kv_transfer_config.gts_server_address or gts_server
        
        self.client = gts.Client(gts_server)
        logger.info(f"Connected to GTS server at {gts_server}")
        
        # Get topology information
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        
        # Get KV transfer configuration from vLLM
        kv_transfer_config = vllm_config.kv_transfer_config
        self.engine_id = kv_transfer_config.engine_id if kv_transfer_config else f"engine_{id(self)}"
        self.kv_role = kv_transfer_config.kv_role if kv_transfer_config else None
        self.kv_rank = kv_transfer_config.kv_rank if kv_transfer_config else None
        
        # Initialize shard manager
        self.shard_manager = GTSShardManager(self.client, self.tp_rank, self.tp_size)
        
        # Initialize metadata capture file
        self.metadata_file = f"gts_metadata_{self.engine_id}_{self.tp_rank}.jsonl"
        logger.info(f"[GTS] Writing metadata to {self.metadata_file}")
        
        # Model configuration using proper parallel-aware methods
        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        self.block_size = vllm_config.cache_config.block_size
        # This already accounts for tensor parallelism
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        self.head_dim = vllm_config.model_config.get_head_size()
        
        # Pipeline parallel info for layer ownership
        self.layer_start, self.layer_end = vllm_config.model_config.get_layers_start_end_indices(
            vllm_config.parallel_config)
        
        # Track pending operations
        self.pending_saves: Set[str] = set()
        
        # Layer KV caches reference
        self.layer_kv_caches: Dict[str, torch.Tensor] = {}
        
        # Request tracking (scheduler-side)
        self.request_states: Dict[str, Dict[str, Any]] = {}
        
    def _write_metadata_to_file(self, metadata: Dict[str, Any]) -> None:
        """Write metadata to JSONL file"""
        try:
            with open(self.metadata_file, 'a') as f:
                f.write(json.dumps(metadata) + '\n')
        except Exception as e:
            logger.error(f"Failed to write metadata to file: {e}")
        
    def register_kv_caches(self, kv_caches: Dict[str, torch.Tensor],
                          kv_cache_layout: str) -> None:
        """Register this worker's shard of each layer's KV cache"""
        self.layer_kv_caches = kv_caches
        
        for layer_name, kv_cache in kv_caches.items():
            self.shard_manager.register_layer_shard(
                layer_name, kv_cache, kv_cache_layout, self.block_size, 
                self.engine_id
            )
    
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                     attn_metadata: 'AttentionMetadata',
                     metadata: GTSConnectorMetadata) -> None:
        """
        Save this worker's shard of a layer's KV cache.
        This is a collective operation across all workers.
        """
        for req_id, req_info in metadata.reqs_to_save.items():
            local_blocks = req_info['local_blocks']
            layer_names = req_info['layer_names']
            
            if layer_name not in layer_names:
                continue
            
            # Metadata for the put operation
            put_metadata = {
                'num_blocks': len(local_blocks),
                'block_size': self.block_size,
                'num_heads': self.num_kv_heads // self.tp_size,  # Sharded heads
                'head_dim': self.head_dim,
                'dtype': kv_layer.dtype,
                'device': str(kv_layer.device),
                'kv_cache_layout': metadata.kv_cache_layout,
                'num_tokens': req_info.get('num_tokens', 0),
                'tp_rank': self.tp_rank,
                'tp_size': self.tp_size
            }
            
            # Add connector context to metadata
            put_metadata.update({
                'engine_id': self.engine_id,
                'kv_role': self.kv_role,
                'kv_rank': self.kv_rank
            })
            
            # Put this worker's shard
            future = self.shard_manager.put_request_shard(
                req_id, layer_name, local_blocks, kv_layer, put_metadata
            )
            
            if future:
                self.pending_saves.add(f"{req_id}_{layer_name}")
    
    def wait_for_save(self) -> None:
        """Wait for all pending save operations"""
        self.shard_manager.wait_for_puts()
        self.pending_saves.clear()
    
    def wait_for_layer_save(self, layer_name: str) -> None:
        """Wait for a specific layer's save operations"""
        self.shard_manager.wait_for_puts(layer_name)
        
        # Clear pending saves for this layer
        self.pending_saves = {
            s for s in self.pending_saves 
            if not s.endswith(f"_{layer_name}")
        }
    
    def get_finished(self, finished_req_ids: Set[str]
                    ) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """Check which requests have finished saving"""
        finished_saves = self.shard_manager.get_finished_puts(finished_req_ids)
        
        # Clear from pending
        for req_id in finished_saves:
            self.pending_saves = {
                s for s in self.pending_saves
                if not s.startswith(f"{req_id}_")
            }
        
        # No loads in put-only design
        return (finished_saves if finished_saves else None, None)
    
    def build_connector_meta(self, scheduler_output: SchedulerOutput
                           ) -> GTSConnectorMetadata:
        """Build metadata for worker-side operations"""
        # Capture metadata to file
        metadata_log = {
            "timestamp": datetime.now().isoformat(),
            "method": "build_connector_meta",
            "engine_id": self.engine_id,
            "scheduler_output": {
                "finished_req_ids": list(scheduler_output.finished_req_ids),
                "num_scheduled_new_reqs": len(scheduler_output.scheduled_new_reqs),
                "scheduled_new_reqs": [
                    {
                        "request_id": req.req_id,
                        "prompt_token_ids_len": len(req.prompt_token_ids),
                        "num_computed_tokens": req.num_computed_tokens,
                        "num_input_tokens": getattr(req, 'num_input_tokens', None),
                        "block_ids": [list(block_list) for block_list in req.block_ids] if req.block_ids else None,
                    } for req in scheduler_output.scheduled_new_reqs
                ]
            }
        }
        
        metadata = GTSConnectorMetadata()
        metadata.engine_id = self.engine_id
        metadata.tp_rank = self.tp_rank
        metadata.tp_size = self.tp_size
        metadata.num_layers = self.num_layers
        metadata.kv_cache_layout = get_kv_cache_layout()
        
        metadata_log["built_metadata"] = {
            "engine_id": metadata.engine_id,
            "tp_rank": metadata.tp_rank,
            "tp_size": metadata.tp_size,
            "num_layers": metadata.num_layers,
            "kv_cache_layout": metadata.kv_cache_layout
        }
        
        # Write to metadata capture file
        self._write_metadata_to_file(metadata_log)
        
        # Finished requests are handled separately via request_finished()
        # The scheduler_output only contains finished_req_ids (set of strings)
        
        return metadata
    
    def request_finished(self, request: 'Request', block_ids: List[int]
                       ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle finished request, potentially saving KV cache"""
        save_kv = getattr(request, 'save_kv_cache', False)
        
        metadata_log = {
            "timestamp": datetime.now().isoformat(),
            "method": "request_finished",
            "engine_id": self.engine_id,
            "request_id": request.request_id,
            "block_ids": block_ids,
            "save_kv_cache": save_kv,
            "request_details": {
                "num_computed_tokens": getattr(request, 'num_computed_tokens', None),
                "allocated_blocks": getattr(request, 'allocated_blocks', None),
                "num_tokens": getattr(request, 'num_tokens', None),
                "prompt_len": getattr(request, 'prompt_len', None)
            }
        }
        
        # Write to metadata capture file
        self._write_metadata_to_file(metadata_log)
        
        if save_kv:
            # Generate metadata for saved cache
            kv_transfer_params = {
                'gts_request_id': request.request_id,
                'num_cached_tokens': request.num_computed_tokens,
                'engine_id': self.engine_id,
                'tp_size': self.tp_size,
                'timestamp': time.time(),
                'is_collective': True
            }
            
            # Return True to indicate async save
            return True, kv_transfer_params
        
        return False, None


class GTSConnector(KVConnectorBase_V1):
    """
    GTS-based KV cache connector for distributed inference.
    Implements collective tensor operations with put-only semantics.
    """
    
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config, role)
        
        if not GTS_AVAILABLE:
            raise ImportError("GTS is not installed. Please install GTS first.")
        
        self.kv_cache_layout = get_kv_cache_layout()
        
        # Create implementation instance
        self._impl = GTSConnectorImpl(vllm_config, role, self)
        
    def register_kv_caches(self, kv_caches: Dict[str, torch.Tensor]) -> None:
        """Register this worker's shards of KV caches"""
        self._impl.register_kv_caches(kv_caches, self.kv_cache_layout)
    
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp) -> None:
        """Set operations for host<->device transfer if needed"""
        # GTS handles transfers internally
        pass
    
    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        """Not implemented in put-only design"""
        # No loading in put-only design
        pass
    
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Not implemented in put-only design"""
        # No loading in put-only design
        pass
    
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                     attn_metadata: 'AttentionMetadata', **kwargs) -> None:
        """Save this worker's shard of a KV cache layer"""
        if self.role != KVConnectorRole.WORKER:
            return
        
        metadata = self._get_connector_metadata()
        if isinstance(metadata, GTSConnectorMetadata):
            self._impl.save_kv_layer(layer_name, kv_layer, attn_metadata, metadata)
    
    def wait_for_save(self) -> None:
        """Wait for all pending save operations"""
        self._impl.wait_for_save()
    
    def get_finished(self, finished_req_ids: Set[str]
                    ) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """Check which requests have finished saving"""
        return self._impl.get_finished(finished_req_ids)
    
    def get_num_new_matched_tokens(self, request: 'Request',
                                  num_computed_tokens: int) -> Tuple[int, bool]:
        """No external KV cache loading in put-only design"""
        return 0, False
    
    def update_state_after_alloc(self, request: 'Request',
                                blocks: 'KVCacheBlocks',
                                num_external_tokens: int) -> None:
        """Update state after block allocation"""
        metadata_log = {
            "timestamp": datetime.now().isoformat(),
            "method": "update_state_after_alloc",
            "engine_id": self._vllm_config.kv_transfer_config.engine_id,
            "request_id": request.request_id,
            "num_external_tokens": num_external_tokens,
            "blocks": {
                "allocated": [str(block) for block in blocks.blocks] if blocks and blocks.blocks else [],
                "num_blocks": len(blocks.blocks) if blocks and blocks.blocks else 0
            },
            "request_details": {
                "num_computed_tokens": getattr(request, 'num_computed_tokens', None),
                "num_tokens": getattr(request, 'num_tokens', None),
                "prompt_len": getattr(request, 'prompt_len', None)
            }
        }
        
        # Write to metadata capture file
        self._impl._write_metadata_to_file(metadata_log)
        
        # Store allocation info for later save operations
        request.allocated_blocks = blocks.blocks if blocks else []
    
    def build_connector_meta(self, scheduler_output: SchedulerOutput
                           ) -> KVConnectorMetadata:
        """Build metadata for worker-side operations"""
        return self._impl.build_connector_meta(scheduler_output)
    
    def request_finished(self, request: 'Request', block_ids: List[int]
                       ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle finished request"""
        return self._impl.request_finished(request, block_ids)
    
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig
                                  ) -> Optional[str]:
        """GTS connector can work with any layout"""
        return None