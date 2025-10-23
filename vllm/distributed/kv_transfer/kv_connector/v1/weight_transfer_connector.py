# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta
from typing import Any

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger

logger = init_logger(__name__)


class WeightTransferConnector:
    """weight transfer connectors for RemoteInstanceLoader."""

    def __init__(self, url: str):
        self.url = url
        self.closed = False
        self._model_update_group = None

    def build_group(
        self,
        gpu_id: int = -1,
        client_rank: int = -1,
        client_id: str = "",
        group_rank: int = 1,
        world_size: int = 2,
    ):
        assert gpu_id != -1 and client_rank != -1, (
            "gpu_id and tp_rank must be specified for RemoteInstanceConnector. "
        )

        self.device_id = torch.device("cuda", gpu_id)
        master_address, master_port = self.url.split(":")
        group_name = f"send_weights_{client_id}_{client_rank}"
        backend = "nccl"

        logger.info(
            "init custom process group: master_address=%s, master_port=%s, "
            "rank_offset=%s, world_size=%s, group_name=%s, backend=%s, gpu_id=%s",
            master_address,
            master_port,
            group_rank,
            world_size,
            group_name,
            backend,
            gpu_id,
        )

        try:
            self._model_update_group = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                timeout=timedelta(seconds=60),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=self.device_id,
            )

            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._model_update_group is not None:
            torch.distributed.distributed_c10d.destroy_process_group(
                self._model_update_group
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self):
        return


# Copy from pytorch and OpenRLHF to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend: str | None = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = "",
    pg_options: Any | None = None,
    device_id: torch.device | int | None = None,
):
    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    backend = Backend(backend) if backend else Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
        device_id=device_id,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg
