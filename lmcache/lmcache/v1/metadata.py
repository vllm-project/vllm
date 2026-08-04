# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

logger = init_logger(__name__)


@dataclass
class LMCacheMetadata:
    """
    LMCacheMetadata should be extracted from the northbound
    serving engine configuration and wrap the extraction of
    attributes (e.g. model name, tp rank, etc.)
    """

    """name of the LLM model"""
    model_name: str
    """ global world size when running under a distributed setting
    (total number of workers)"""
    world_size: int
    """ host world size (workers on active localhost)
    This information can be useful for multi-node
    deployment. Will be the same as world_size
    in single-node deployments.
    """
    local_world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ host worker id (a gpu bound worker id on active localhost)
    This information can be useful for multi-node deployment.
    Will be the same as worker_id in single-node deployments.
    """
    local_worker_id: int
    """ the data type of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    """ whether use MLA"""
    use_mla: bool = False
    """ the role of the current instance (e.g., 'scheduler', 'worker') """
    role: Optional[str] = None
    """ the first rank of the distributed setting """
    # TODO(baoloongmao): first_rank should be configurable
    first_rank = 0
    served_model_name: Optional[str] = None
    """chunk size"""
    chunk_size: int = 256
    """Manager for groups of layers with identical KV cache structure.

    ``None`` until the serving-engine adapter (e.g. the vLLM connector)
    registers KV caches and constructs the manager via
    :class:`KVLayerGroupsManager`. Consumers must guard against ``None``
    when accessed before registration.
    """
    kv_layer_groups_manager: Optional["KVLayerGroupsManager"] = None
    """ engine_id for RPC path (used by lookup client/server) """
    engine_id: Optional[str] = None
    """ extra config from kv_connector (e.g., lmcache_rpc_port) """
    kv_connector_extra_config: Optional[dict] = None

    def is_first_rank(self) -> bool:
        """Check if the current worker is the first rank"""
        return self.worker_id == self.first_rank

    def get_dtypes(self) -> list[torch.dtype]:
        """Return per-group dtypes, or the legacy single dtype if the
        manager has not been registered yet (e.g. some unit tests)."""
        klg_manager = self.kv_layer_groups_manager
        if klg_manager is not None and klg_manager.kernel_groups:
            return [group.dtype for group in klg_manager.kernel_groups]
        return [self.kv_dtype]

    def get_shapes(self, num_tokens: Optional[int] = None) -> list[torch.Size]:
        """Get the shapes of the KV cache in LMCache"""
        if num_tokens is None:
            num_tokens = self.chunk_size
        klg_manager = self.kv_layer_groups_manager
        if klg_manager is not None and klg_manager.kernel_groups:
            # Read kv_size from each group's shape_desc rather than self.use_mla
            # so heterogeneous groups (should any ever co-exist) are handled.
            return [
                torch.Size(
                    [
                        group.shape_desc.kv_size,
                        group.num_layers,
                        num_tokens,
                        group.hidden_dim_size,
                    ]
                )
                for group in klg_manager.kernel_groups
            ]
        return [
            torch.Size(
                [
                    self.kv_shape[1],
                    self.kv_shape[0],
                    num_tokens,
                    self.kv_shape[3] * self.kv_shape[4],
                ]
            )
        ]

    def get_num_groups(self) -> int:
        klg_manager = self.kv_layer_groups_manager
        if klg_manager is not None and klg_manager.kernel_groups:
            return klg_manager.num_kernel_groups
        return 1
