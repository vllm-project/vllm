# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import functools
import os
import pickle
import socket
import sys
import time
import uuid
from collections import deque
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import torch
from torch.distributed import ProcessGroup, Store, TCPStore
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _get_default_timeout,
    _unregister_process_group,
)
from torch.distributed.rendezvous import rendezvous

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.network_utils import get_tcp_uri
from vllm.utils.system_utils import suppress_stdout

logger = init_logger(__name__)

# We prefer to use os.sched_yield as it results in tighter polling loops,
# measured to be around 3e-7 seconds. However on earlier versions of Python
# os.sched_yield() does not release the GIL, so we fall back to time.sleep(0)
USE_SCHED_YIELD = (sys.version_info[:3] >= (3, 11, 1)) or (
    sys.version_info[:2] == (3, 10) and sys.version_info[2] >= 8
)


def sched_yield():
    if USE_SCHED_YIELD:
        os.sched_yield()
    else:
        time.sleep(0)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def verify_group_size_divides_partition(
    input_size_per_partition: int,
    group_size: int,
    layer_name: str | None = None,
    extra_suggestion: str = "",
) -> None:
    """Validate that a TP-sharded layer holds a whole number of quant groups."""
    if input_size_per_partition % group_size == 0:
        return
    location = f" for layer '{layer_name}'" if layer_name else ""
    raise ValueError(
        f"Weight {input_size_per_partition=}{location} is not divisible by "
        f"{group_size=}. This happens when tensor_parallel_size splits the layer input "
        "into shards that are not a whole number of quant groups. Consider reducing "
        f"tensor_parallel_size{extra_suggestion}."
    )


# --------------------------------------------------------------------------
# Uneven tensor-parallel partitioning (--rank-tp-ratio).
#
# When a ratio vector like (2, 1, 1) is active, TP rank r owns
# total * ratio[r] / sum(ratio) of every sharded dimension instead of
# total / tp_size. Offsets become prefix sums (same pattern as
# VLLM_PP_LAYER_PARTITION for pipeline parallel). The ratio vector is
# process-global state set once per worker before the model is built;
# when unset, all helpers reproduce the classic even split exactly.
# --------------------------------------------------------------------------

_TP_PARTITION_RATIOS: list[int] | None = None


def set_tp_partition_ratios(ratios: list[int] | None) -> None:
    """Install the uneven-TP ratio vector for this process (or None)."""
    global _TP_PARTITION_RATIOS
    _TP_PARTITION_RATIOS = list(ratios) if ratios else None


def get_tp_partition_ratios() -> list[int] | None:
    return _TP_PARTITION_RATIOS


def partition_units(units: int, weights: list[int]) -> list[int]:
    """Split `units` indivisible units over ranks proportionally to
    `weights` (largest-remainder rounding, every rank gets >= 1 unit).

    Deterministic pure function of (units, weights) so every process
    computes the identical partition. Ties in the fractional parts are
    broken toward the lower rank index.
    """
    n = len(weights)
    if units < n:
        raise ValueError(
            f"Cannot give each of {n} ranks at least one of {units} units."
        )
    total_w = sum(weights)
    quotas = [units * w / total_w for w in weights]
    sizes = [int(q) for q in quotas]
    # Reserve a minimum of one unit per rank before distributing the rest.
    sizes = [max(s, 1) for s in sizes]
    remaining = units - sum(sizes)
    if remaining < 0:
        # Minimum-1 bumping overshot: take back from the largest shares.
        for _ in range(-remaining):
            i = max(range(n), key=lambda r: (sizes[r], -r))
            sizes[i] -= 1
        remaining = 0
    order = sorted(
        range(n), key=lambda r: (quotas[r] - int(quotas[r]), -r), reverse=True
    )
    for k in range(remaining):
        sizes[order[k % n]] += 1
    assert sum(sizes) == units and all(s >= 1 for s in sizes)
    return sizes


def tp_partition_sizes(total: int, tp_size: int, units: int | None = None) -> list[int]:
    """Per-rank sizes of a dimension of `total` elements.

    `units` is the master unit count of the dimension FAMILY (v4 free
    partitioning): e.g. attention q heads (24), GDN k heads (16),
    intermediate/quant_group. All dimensions derived from the same family
    pass the same `units`, so they round identically and stay mutually
    consistent (qkv columns = 2 x q heads, GDN v dim = 3 x k heads, ...).
    total must be a multiple of units.

    Without `units` (un-migrated call sites), the v1 rule applies: every
    sharded dimension must be divisible by sum(ratios) so per-rank sizes
    are exact; otherwise this raises with the offending dimension size.
    """
    ratios = _TP_PARTITION_RATIOS
    if not ratios or len(ratios) != tp_size:
        # No ratios installed, or this layer runs with its own tp_size
        # (disable_tp layers use tp_size=1): classic even split.
        ensure_divisibility(total, tp_size)
        return [total // tp_size] * tp_size
    if units is not None:
        if total % units != 0:
            raise ValueError(
                f"Dimension of size {total} is not a multiple of its "
                f"family unit count {units}."
            )
        scale = total // units
        return [s * scale for s in partition_units(units, ratios)]
    denom = sum(ratios)
    if total % denom != 0:
        raise ValueError(
            f"Cannot partition dimension of size {total} with "
            f"--rank-tp-ratio {ratios}: {total} is not divisible by "
            f"sum(ratios)={denom}. Choose ratios whose sum divides every "
            "sharded dimension (attention heads, kv heads, GDN heads, "
            "hidden/intermediate sizes), or migrate this call site to "
            "unit-based partitioning (v4)."
        )
    unit = total // denom
    return [unit * r for r in ratios]


def tp_partition_size(
    total: int, tp_size: int, rank: int, units: int | None = None
) -> int:
    """This rank's size of a sharded dimension."""
    return tp_partition_sizes(total, tp_size, units)[rank]


def tp_partition_offset(
    total: int, tp_size: int, rank: int, units: int | None = None
) -> int:
    """This rank's start offset (prefix sum) in a sharded dimension."""
    return sum(tp_partition_sizes(total, tp_size, units)[:rank])


_CP_TOKEN_RATIOS: list[int] | None = None


def set_cp_token_ratios(ratios: list[int] | None) -> None:
    """Install the token-axis split vector for uneven DCP (v4: derived
    from the GDN k-head partition so the align invariant holds; v3:
    identical to the weight vector)."""
    global _CP_TOKEN_RATIOS
    _CP_TOKEN_RATIOS = list(ratios) if ratios else None


def get_cp_token_ratios() -> list[int] | None:
    return _CP_TOKEN_RATIOS


def gdn_family_units(vllm_config) -> int | None:
    """Master unit count of the GDN head family, respecting the
    quantization block granularity: shard cuts of the GDN out_proj INPUT
    (value dim) must land on quant-block boundaries. E.g. Qwen3.6 Q6_K:
    one k-head spans 3*128=384 value elements but K-quant superblocks are
    256 -> units of 2 k-heads (768 = 3*256), i.e. 8 family units instead
    of 16. None for non-hybrid models."""
    import math

    hf_config = vllm_config.model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config) or hf_config
    num_k_heads = getattr(text_config, "linear_num_key_heads", None)
    if not num_k_heads:
        return None
    num_v_heads = getattr(text_config, "linear_num_value_heads", num_k_heads)
    head_v_dim = getattr(text_config, "linear_value_head_dim", 128)
    unit_v_elems = head_v_dim * (num_v_heads // num_k_heads)
    group = getattr(vllm_config.quant_config, "group_size", None) or 1
    if group > 1 and unit_v_elems % group:
        k_per_unit = math.lcm(unit_v_elems, group) // unit_v_elems
        if num_k_heads % k_per_unit:
            raise ValueError(
                f"GDN k heads ({num_k_heads}) cannot be partitioned in "
                f"steps of {k_per_unit} (quant group {group} vs "
                f"{unit_v_elems} value elements per k head)."
            )
        return num_k_heads // k_per_unit
    return num_k_heads


def gdn_head_partition(vllm_config, weights: list[int]) -> list[int]:
    """Per-rank GDN k-head counts under the quant-aware family units
    (whole units are partitioned, then scaled back to heads)."""
    units = gdn_family_units(vllm_config)
    assert units is not None
    hf_config = vllm_config.model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config) or hf_config
    num_k_heads = text_config.linear_num_key_heads
    scale = num_k_heads // units
    return [u * scale for u in partition_units(units, list(weights))]


def resolve_cp_token_ratios(vllm_config) -> list[int] | None:
    """Token-axis split vector for uneven DCP, derived from the config.

    Hybrid models (GDN/linear attention): the GDN k-head partition - the
    mamba state on rank r scales with its k-head share, and the align
    invariant (mamba page == attention bytes per virtual block, per rank)
    requires the token split to be proportional to exactly that. Dense
    models: the raw --rank-tp-ratio weights.
    """
    parallel_config = vllm_config.parallel_config
    weights = parallel_config.rank_tp_ratio
    if (
        not weights
        or parallel_config.decode_context_parallel_size <= 1
        or len(set(weights)) == 1
    ):
        return None
    import math

    import vllm.envs as envs

    override = envs.VLLM_UNEVEN_TOKEN_VECTOR
    if override:
        # Manual override (see the imbalance warning in
        # kv_cache_utils.get_kv_cache_configs): a token vector measured
        # from a previous run's per-worker free memory. Must be set for
        # ALL processes (inherited env), one entry per rank.
        vector = [int(x) for x in override.split(",")]
        if len(vector) != len(weights) or any(v <= 0 for v in vector):
            raise ValueError(
                f"VLLM_UNEVEN_TOKEN_VECTOR={override!r} must name "
                f"{len(weights)} positive integers (one per rank)."
            )
        g = math.gcd(*vector)
        return [v // g for v in vector]

    budgets = parallel_config.rank_gpu_memory_mib
    if isinstance(budgets, list) and len(budgets) == len(weights):
        # Split the KV cache proportionally to each rank's FREE
        # memory (budget minus its weight share minus a fixed overhead),
        # so every card fills up. The mamba pages are padded per rank to
        # the attention bytes-per-virtual-block (align solver), so the
        # token vector no longer needs to be proportional to the GDN
        # head partition. Integerized to 32 units (>= 1 per rank).
        w_mib = _checkpoint_size_mib(vllm_config)
        if w_mib > 0:
            total_w = sum(weights)
            avail = [
                max(
                    b - w_mib * w / total_w - _AUTO_TOKEN_OVERHEAD_MIB,
                    1.0,
                )
                for b, w in zip(budgets, weights)
            ]
            token_vector = partition_units(
                _TOKEN_VECTOR_UNITS, [max(int(a), 1) for a in avail]
            )
            g = math.gcd(*token_vector)
            return [t // g for t in token_vector]

    hf_config = vllm_config.model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config) or hf_config
    gdn_k_heads = getattr(text_config, "linear_num_key_heads", None)
    if gdn_k_heads:
        part = gdn_head_partition(vllm_config, list(weights))
        # Reduce by the gcd: smaller entries keep the virtual scheduler
        # blocks (block_size * sum(token ratios)) as fine as possible.
        # For divisible weights like (2,1,1) this reproduces the v3
        # vector exactly ([8,4,4] -> [2,1,1]).
        g = math.gcd(*part)
        return [p // g for p in part]
    return list(weights)


#: Token-vector resolution (units) and assumed weight-independent
#: per-rank overhead for the free-memory split.
_TOKEN_VECTOR_UNITS = 64
_AUTO_TOKEN_OVERHEAD_MIB = 1536


def _checkpoint_size_mib(vllm_config) -> int:
    """Total checkpoint size of the model on disk (MiB), 0 if unknown.
    Deterministic in every process - the token vector derived from it
    must be identical everywhere."""
    import glob
    import os

    path = vllm_config.model_config.model
    if os.path.isfile(path):
        # Single-file checkpoints (e.g. GGUF).
        return os.path.getsize(path) // 2**20
    if not os.path.isdir(path):
        return 0
    total = sum(
        os.path.getsize(f) for f in glob.glob(os.path.join(path, "*.safetensors"))
    )
    if total == 0:
        total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "*.gguf")))
    return total // 2**20


def uneven_cp_ratios(cp_world_size: int) -> list[int] | None:
    """Ratio vector for uneven (token-axis) context parallelism.

    Uneven DCP splits the KV cache of the full-attention layers along
    the TOKEN axis (dcp group == tp group, so len(ratios) ==
    cp_world_size). The token vector is the installed CP token ratio
    vector when set (v4: the GDN k-head partition, which keeps the
    mamba/attention align invariant), else the --rank-tp-ratio weights
    (v3). Returns None when the classic even split applies.
    """
    ratios = _CP_TOKEN_RATIOS or _TP_PARTITION_RATIOS
    if not ratios or len(ratios) != cp_world_size or cp_world_size <= 1:
        return None
    if all(r == ratios[0] for r in ratios):
        # Uniform ratios degenerate to the even split; keep default path.
        return None
    return ratios


def cp_q_head_counts(total_q_heads: int, cp_world_size: int) -> list[int] | None:
    """Per-rank q-head counts across the DCP group under uneven ratios
    (v4: units-based partition, which need not be proportional to the
    token vector). None when no ratios are installed for this group
    size. Deterministic pure function - identical in every process."""
    ratios = _TP_PARTITION_RATIOS
    if not ratios or len(ratios) != cp_world_size or cp_world_size <= 1:
        return None
    return tp_partition_sizes(total_q_heads, cp_world_size, units=total_q_heads)


def uneven_dcp_active() -> bool:
    """True when uneven DCP runs in this process: non-uniform
    --rank-tp-ratio weights are installed and the DCP group spans the
    whole TP group. Attention KV heads are then fully replicated per
    rank and the KV cache is split along the token axis (whose vector
    may itself be uniform after budget balancing)."""
    try:
        from vllm.distributed.parallel_state import get_dcp_group

        dcp_world_size = get_dcp_group().world_size
    except (AssertionError, ImportError):
        return False
    weights = _TP_PARTITION_RATIOS
    return (
        dcp_world_size > 1
        and bool(weights)
        and len(weights) == dcp_world_size
        and len(set(weights)) > 1
    )


def cp_token_split_factor(dcp_world_size: int, pcp_world_size: int = 1) -> int:
    """Number of block_size-units one "virtual" scheduler block spans.

    Even CP: dcp * pcp (each rank owns exactly one physical block per
    virtual block). Uneven DCP: sum(ratios) - rank r owns a contiguous
    "superblock" of ratios[r] physical blocks per virtual block, so the
    scheduler stays CP-agnostic while ranks store token counts
    proportional to their ratio.
    """
    ratios = uneven_cp_ratios(dcp_world_size)
    if ratios is not None:
        return sum(ratios) * pcp_world_size
    return dcp_world_size * pcp_world_size


def cp_rank_ratio_prefix(cp_world_size: int, cp_rank: int) -> tuple[int, int, int]:
    """(ratio, prefix, sum) of this rank's token-axis share under uneven
    DCP; (1, cp_rank, cp_world_size) for the classic even split (the
    generalized slot-mapping formulas reduce exactly to the even ones
    with these values)."""
    ratios = uneven_cp_ratios(cp_world_size)
    if ratios is None:
        return 1, cp_rank, cp_world_size
    return ratios[cp_rank], sum(ratios[:cp_rank]), sum(ratios)


def is_weak_contiguous(inp: torch.Tensor) -> bool:
    """Check that *inp* occupies a single contiguous block of memory.

    Unlike ``torch.Tensor.is_contiguous()``, this also accepts tensors
    whose strides are not strictly C-contiguous (e.g. column-major) as
    long as the underlying storage from the tensor's offset onward is
    exactly ``numel * element_size`` bytes.
    """
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """Try to evenly distribute layers across partitions.

    If the number of layers is not divisible by the number of partitions,
    the remaining layers are evenly distributed across all but the last
    partition. The last partition is excluded because it often contains an
    additional norm layer and we are attempting to balance compute.

    If `pp_size > 2` and the number of remaining layers is
    `0 < x <= pp_size - 2` then the remaining layers are evenly distributed
    across the middle partitions. The first and last partitions are excluded
    because they contain the input and output embeddings respectively and we
    are attempting to reduce maximum memory consumption across partitions.
    """
    partition_list_str = envs.VLLM_PP_LAYER_PARTITION
    if partition_list_str is not None:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError(
                "Invalid partition string: {}".format(partition_list_str)
            ) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
    else:
        layers_per_partition = num_hidden_layers // pp_size
        partitions = [layers_per_partition for _ in range(pp_size)]

        if remaining_layers := num_hidden_layers % pp_size:
            for i in range(2, remaining_layers + 2):
                partitions[-i] += 1
            logger.info(
                "Hidden layers were unevenly partitioned: [%s]. "
                "This can be manually overridden using the "
                "VLLM_PP_LAYER_PARTITION environment variable",
                ",".join(str(p) for p in partitions),
            )

    start_layer = sum(partitions[:pp_rank])
    end_layer = start_layer + partitions[pp_rank]

    return (start_layer, end_layer)


def create_tcp_store(
    host: str,
    port: int,
    listen_socket: socket.socket | None = None,
    **kwargs: Any,
) -> TCPStore:
    """Create a TCPStore, optionally taking ownership of ``listen_socket``."""
    if listen_socket is None:
        return TCPStore(host_name=host, port=port, **kwargs)

    listen_fd = listen_socket.detach()
    try:
        return TCPStore(
            host_name=host,
            port=port,
            master_listen_fd=listen_fd,
            **kwargs,
        )
    except Exception:
        socket.close(listen_fd)
        raise


@dataclasses.dataclass
class StatelessProcessGroup:
    """A dataclass to hold a metadata store, and the rank, world_size of the
    group. Only use it to communicate metadata between processes.
    For data-plane communication, create NCCL-related objects.
    """

    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store

    data_expiration_seconds: int = 3600  # 1 hour

    # dst rank -> counter
    send_dst_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    # src rank -> counter
    recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)

    # A deque to store the data entries, with key and timestamp.
    entries: deque[tuple[str, float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {i: 0 for i in range(self.world_size)}

    def send_obj(self, obj: Any, dst: int):
        """Send an object to a destination rank."""
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.time()))

    def expire_data(self):
        """Expire data that is older than `data_expiration_seconds` seconds."""
        while self.entries:
            # check the oldest entry
            key, timestamp = self.entries[0]
            if time.time() - timestamp > self.data_expiration_seconds:
                self.store.delete_key(key)
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        obj = pickle.loads(
            self.store.get(f"send_to/{self.rank}/{self.recv_src_counter[src]}")
        )
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Any | None, src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks.
        It does not clean up after all ranks have received the object.
        Use it for limited times, e.g., for initialization.
        """
        if self.rank == src:
            self.expire_data()
            key = f"broadcast_from/{src}/{self.broadcast_send_counter}"
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.time()))
            return obj
        else:
            key = f"broadcast_from/{src}/{self.broadcast_recv_src_counter[src]}"
            recv_obj = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """All gather an object from all ranks."""
        gathered_objs = []
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs.append(obj)
                self.broadcast_obj(obj, src=self.rank)
            else:
                recv_obj = self.broadcast_obj(None, src=i)
                gathered_objs.append(recv_obj)
        return gathered_objs

    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """Broadcast a tensor from source rank to all other ranks."""
        if self.rank == src:
            tensor_bytes = pickle.dumps(tensor)
            self.expire_data()
            key = f"broadcast_tensor/{src}/{self.broadcast_send_counter}"
            self.store.set(key, tensor_bytes)
            self.broadcast_send_counter += 1
            self.entries.append((key, time.time()))
            return tensor
        else:
            key = f"broadcast_tensor/{src}/{self.broadcast_recv_src_counter[src]}"
            tensor = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return tensor

    def send(self, tensor: torch.Tensor, dst: int):
        """Send a tensor to a destination rank."""
        self.expire_data()
        key = f"send_tensor/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(tensor))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.time()))

    def recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """Receive a tensor from a source rank."""
        key = f"send_tensor/{self.rank}/{self.recv_src_counter[src]}"
        received = pickle.loads(self.store.get(key))
        self.recv_src_counter[src] += 1
        tensor.copy_(received)
        return tensor

    def all_reduce(
        self, tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM
    ) -> torch.Tensor:
        """All-reduce a tensor across all ranks."""
        tensors = self.all_gather_obj(tensor)
        result = tensors[0].clone()
        for t in tensors[1:]:
            if op == torch.distributed.ReduceOp.SUM:
                result.add_(t)
            elif op == torch.distributed.ReduceOp.PRODUCT:
                result.mul_(t)
            elif op == torch.distributed.ReduceOp.MAX:
                result = torch.maximum(result, t)
            elif op == torch.distributed.ReduceOp.MIN:
                result = torch.minimum(result, t)
        return result

    def barrier(self, timeout: float = 30.0):
        """A robust barrier to synchronize all ranks.


        Uses a multi-phase approach to ensure all processes reach the barrier
        before proceeding:

        1. Each process signals it has reached the barrier

        2. Each process signals that it has confirmed the arrival of all other
        ranks.

        3. Rank 0 waits for all other ranks to signal their departure to ensure
        that all ranks have departed the barrier first.

        Args:
            timeout: Maximum time in seconds to wait for each phase (in seconds)


        Raises:
            RuntimeError: If coordination fails or times out
        """
        # Generate a barrier ID that is globally unique
        try:
            if self.rank == 0:
                barrier_id = f"barrier_{uuid.uuid4()}"
                self.broadcast_obj(barrier_id, src=0)
            else:
                barrier_id = self.broadcast_obj(None, src=0)
        except Exception as e:
            raise RuntimeError("Failed to broadcast barrier_id") from e

        # Phase 1: Signal arrival at barrier
        # Wait for all processes to arrive
        # We need all ranks to confirm the arrival of all other ranks.
        # This is the key synchronization point.
        arrival_key = f"arrival_{barrier_id}_{self.rank}"
        try:
            self.store.set(arrival_key, b"1")
        except Exception as e:
            raise RuntimeError("Failed to signal barrier arrival") from e

        start_time = time.time()
        processes_arrived: set[int] = set()

        while len(processes_arrived) < self.world_size:
            # Check for timeout
            cur_time = time.time()
            if cur_time - start_time > timeout:
                raise RuntimeError(f"Barrier timed out after {timeout:.2f} seconds")

            # Check for each process
            for i in range(self.world_size):
                if i in processes_arrived:
                    continue

                key = f"arrival_{barrier_id}_{i}"
                try:
                    # Try to get the key - if it exists, we'll get a value
                    # If it doesn't exist, it will throw an exception
                    self.store.get(key)
                    processes_arrived.add(i)
                except KeyError:
                    # Key doesn't exist yet
                    pass
                except Exception as check_e:
                    logger.debug("Error checking key existence: %s", check_e)
                    sched_yield()

            # Short sleep to avoid tight polling
            if len(processes_arrived) < self.world_size:
                sched_yield()

        # Phase 2: Signal departure from barrier
        # We only care to block at this stage in rank 0, which runs the
        # server side of the TCPStore. We want to make sure that all
        # clients have departed the barrier before rank 0 in case the
        # next thing after the barrier is a shutdown, including tearing
        # down the TCPStore. Other ranks can exit the barrier immediately
        # after signaling their departure.
        departure_key = f"departure_{barrier_id}_{self.rank}"
        try:
            self.store.set(departure_key, b"1")
        except Exception as e:
            raise RuntimeError("Failed to signal barrier departure") from e

        if self.rank != 0:
            return

        # Make rank 0 wait for all processes to signal departure
        start_time = time.time()
        processes_departed: set[int] = set()

        while len(processes_departed) < self.world_size:
            # Check for timeout
            if time.time() - start_time > timeout:
                raise RuntimeError(
                    f"Barrier departure timed out after {timeout:.2f} seconds"
                )

            # Check for each process
            for i in range(self.world_size):
                if i in processes_departed:
                    continue

                key = f"departure_{barrier_id}_{i}"
                try:
                    # Try to get the key - if it exists, we'll get a value
                    # If it doesn't exist, it will throw an exception
                    self.store.get(key)
                    processes_departed.add(i)
                except KeyError:
                    # Key doesn't exist yet
                    pass
                except Exception as check_e:
                    logger.debug("Error checking key existence: %s", check_e)
                    sched_yield()

            # Short sleep to avoid tight polling
            if len(processes_departed) < self.world_size:
                sched_yield()

        # Clean up keys to avoid leaking memory in the store
        for i in range(self.world_size):
            try:
                self.store.delete_key(f"arrival_{barrier_id}_{i}")
            except Exception:
                logger.debug("Error deleting key: %s", f"arrival_{barrier_id}_{i}")

            try:
                self.store.delete_key(f"departure_{barrier_id}_{i}")
            except Exception:
                logger.debug("Error deleting key: %s", f"departure_{barrier_id}_{i}")

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
        store_timeout: int = 300,
        listen_socket: socket.socket | None = None,
    ) -> "StatelessProcessGroup":
        """A replacement for `torch.distributed.init_process_group` that does not
        pollute the global state.

        If we have process A and process B called `torch.distributed.init_process_group`
        to form a group, and then we want to form another group with process A, B, C,
        D, it is not possible in PyTorch, because process A and process B have already
        formed a group, and process C and process D cannot join that group. This
        function is a workaround for this issue.

        `torch.distributed.init_process_group` is a global call, while this function
        is a stateless call. It will return a `StatelessProcessGroup` object that can be
        used for exchanging metadata. With this function, process A and process B
        can call `StatelessProcessGroup.create` to form a group, and then process A, B,
        C, and D can call `StatelessProcessGroup.create` to form another group.
        """  # noqa
        launch_server = rank == 0
        if launch_server and listen_socket is None:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind((host, port))
            listen_socket.listen()
        store = create_tcp_store(
            host,
            port,
            listen_socket=listen_socket,
            world_size=world_size,
            is_master=launch_server,
            timeout=timedelta(seconds=store_timeout),
            use_libuv=False,  # for now: github.com/pytorch/pytorch/pull/150215
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds,
        )


@functools.lru_cache(maxsize=1)
def get_cached_tcp_store_client(host: str, port: int) -> TCPStore:
    """Return a cached TCPStore client.

    Cached so that every call with the same ``(host, port)`` reuses the
    same connection.  A new ``(host, port)`` evicts the old entry.
    """
    return TCPStore(host, port, is_master=False, wait_for_workers=False)


def get_cpu_distributed_timeout_or_none() -> timedelta | None:
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None
    timeout_seconds = vllm_config.parallel_config.cpu_distributed_timeout_seconds
    return timedelta(seconds=timeout_seconds) if timeout_seconds is not None else None


def get_distributed_timeout_or_none() -> timedelta | None:
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None
    timeout_seconds = vllm_config.parallel_config.distributed_timeout_seconds
    return timedelta(seconds=timeout_seconds) if timeout_seconds is not None else None


def init_gloo_process_group(
    prefix_store: PrefixStore,
    group_rank: int,
    group_size: int,
    timeout: timedelta,
) -> ProcessGroup:
    """
    Stateless init ProcessGroup with gloo backend compatible with
    different torch versions.
    """
    with suppress_stdout():
        pg = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
        from torch.distributed.distributed_c10d import ProcessGroupGloo

        backend_class = ProcessGroupGloo(
            prefix_store, group_rank, group_size, timeout=timeout
        )
        backend_type = ProcessGroup.BackendType.GLOO
        device = torch.device("cpu")
        pg._set_default_backend(backend_type)
        backend_class._set_sequence_number_for_group()

        pg._register_backend(device, backend_type, backend_class)
    return pg


def stateless_init_torch_distributed_process_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    backend: str,
    group_name: str | None = None,
    return_store: bool = False,
    listen_socket: socket.socket | None = None,
) -> ProcessGroup | tuple[ProcessGroup, Store]:
    """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.

    # TODO: ask for help from PyTorch team if we need the `broadcast` operation.

    This function is useful when we are not sure about the total number of
    processes in the process group. For example, we may have process
    1, 2, ..., 8 who want to communicate, and process 9 might be the same
    process as process 1, or it might be a different process; process 10
    might be the same process as process 5, or it might be a different process.
    In this case, how can we reliably form a communication channel within
    process 9 and 10, without affecting the communication channel within
    process 1, 2, ..., 8?

    One possible solution is to figure out if process 9 and 10 are the same
    as process 1 and 5 beforehand, and then form a communication channel
    based on the information, adjusting the ranks and world_size etc. However,
    figuring out the information is not always easy, and it will interfere
    with the main communication channel.

    Our solution is to always form a communication channel with process 1, 2,
    ..., 8, and then use this function to form another communication channel
    with process 9 and 10. This way, regardless of whether process 9 and 10
    are the same as process 1 and 5, the main communication channel is
    always formed with process 1, 2, ..., 8, and the additional communication
    channel is formed with process 9 and 10.

    When *listen_socket* is provided, the rendezvous step
    is skipped and a ``TCPStore`` server is created directly using the
    pre-bound socket.  This is useful for eliminating TOCTOU races
    between port allocation and binding.
    """
    init_method = get_tcp_uri(host, port)
    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)
    if backend == "gloo":
        gloo_timeout = get_cpu_distributed_timeout_or_none()
        if gloo_timeout is not None:
            timeout = gloo_timeout
    else:
        device_timeout = get_distributed_timeout_or_none()
        if device_timeout is not None:
            timeout = device_timeout

    if listen_socket is not None:
        store = create_tcp_store(
            host,
            port,
            listen_socket=listen_socket,
            world_size=world_size,
            is_master=True,
            timeout=timeout,
            multi_tenant=True,
        )
    else:
        store, rank, world_size = next(
            rendezvous(init_method, rank, world_size, timeout=timeout)
        )
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    if backend == "gloo":
        pg = init_gloo_process_group(
            prefix_store=prefix_store,
            group_rank=group_rank,
            group_size=group_size,
            timeout=timeout,
        )
    else:
        from vllm.platforms import current_platform

        pg = current_platform.stateless_init_device_torch_dist_pg(
            backend=backend,
            prefix_store=prefix_store,
            group_rank=group_rank,
            group_size=group_size,
            timeout=timeout,
        )

    if group_name is not None:
        from torch._C._distributed_c10d import _register_process_group

        pg._set_group_name(group_name)
        _register_process_group(group_name, pg)

    if return_store:
        return pg, store
    else:
        return pg


def stateless_destroy_torch_distributed_process_group(pg: ProcessGroup) -> None:
    """
    Destroy ProcessGroup returned by
        stateless_init_torch_distributed_process_group().
    """
    pg.shutdown()
    _unregister_process_group(pg.group_name)


def get_worker_rank_suffix(global_rank: int | None = None) -> str:
    """Generate a descriptive rank suffix for worker identification.

    Returns a string like 'dp0_pp0_tp0_dcp0_ep0_rank0' including all
    parallel dimensions: DP, PP, TP, DCP, EP.

    Args:
        global_rank: Optional global rank to append. If not provided,
                     only parallel dimension ranks are included.

    Returns:
        A string suffix identifying the worker's position in the
        distributed topology.
    """
    from vllm.distributed.parallel_state import (
        get_dcp_group,
        get_dp_group,
        get_ep_group,
        get_pp_group,
        get_tp_group,
    )

    try:
        dp_rank = get_dp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        dcp_rank = get_dcp_group().rank_in_group
        ep_rank = get_ep_group().rank_in_group

        suffix = f"dp{dp_rank}_pp{pp_rank}_tp{tp_rank}_dcp{dcp_rank}_ep{ep_rank}"
        if global_rank is not None:
            suffix = f"{suffix}_rank{global_rank}"
        return suffix
    except Exception:
        # Fallback if parallel state not initialized
        if global_rank is not None:
            return f"rank{global_rank}"
        return ""
