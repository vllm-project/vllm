# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NVLink multicast budget of the Kimi K3 symmetric-memory mailboxes
(`specialized_decode/symm.py`).

A GB300 NVL72 rack shares one pool of about 820 multicast objects across every
process of its NVLink domain; each `symm_mem.rendezvous` binds one unless
multicast is disabled for it. Checked here:

- `peer_rendezvous`: no multicast address, every peer's stores land through
  `buffer_ptrs` / `get_buffer`, and the process environment is restored;
- `multicast_rendezvous`: the multicast address on multicast platforms, and 0 on
  every rank when one rank's binding is skipped (the ranks agree on the path);
- the specialized decode's peer-store mailboxes (DCP partials, top-k exchange,
  MoE tail reduce-scatter) bind no multicast object and still map every peer;
- the specialized decode refuses TP ranks that span nodes without one NVLink fabric
  (CPU only: the topology is faked).

Runs on 2 GPUs (`mp.spawn`), or on every rank of a `torchrun` / `srun` launch
(e.g. TP8 across two GB300 NVL72 trays) when `RANK` and `WORLD_SIZE` are set.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

ENV = "TORCH_SYMM_MEM_DISABLE_MULTICAST"


def has_multicast() -> bool:
    from torch._C._autograd import DeviceType
    from torch._C._distributed_c10d import _SymmetricMemory

    index = torch.accelerator.current_device_index()
    return bool(_SymmetricMemory.has_multicast_support(DeviceType.CUDA, index))


def check_peer_stores(handle, tensor, world, rank, tag):
    """Rank r stores r + 1 + 100 * tag into slot r of every peer's buffer."""
    tensor.view(torch.int32)[:world].zero_()
    torch.accelerator.synchronize()
    dist.barrier()
    for peer in range(world):
        handle.get_buffer(peer, [world], torch.int32, 0)[rank].fill_(
            rank + 1 + 100 * tag
        )
    torch.accelerator.synchronize()
    dist.barrier()
    got = tensor.view(torch.int32)[:world].tolist()
    assert got == [r + 1 + 100 * tag for r in range(world)], (tag, rank, got)


def check(world: int) -> None:
    import torch.distributed._symmetric_memory as symm_mem

    from vllm.models.kimi_k3.nvidia.specialized_decode.symm import (
        multicast_rendezvous,
        peer_rendezvous,
    )

    group = get_tp_group()
    rank = group.rank_in_group
    name = group.device_group.group_name
    device = torch.device("cuda", torch.accelerator.current_device_index())
    multicast = has_multicast()

    # peer_rendezvous: no multicast address, peer stores land, env restored.
    for previous in (None, "0"):
        if previous is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = previous
        t = symm_mem.empty(1 << 16, dtype=torch.int32, device=device)
        handle = peer_rendezvous(t, name)
        # Restored ("0", which binds like an unset variable, where it was unset).
        assert os.environ.get(ENV) == (previous or "0"), (previous, os.environ.get(ENV))
        assert int(handle.multicast_ptr or 0) == 0
        assert all(handle.buffer_ptrs) and len(handle.buffer_ptrs) == world
        check_peer_stores(handle, t, world, rank, 1)
    os.environ.pop(ENV, None)

    # multicast_rendezvous: the address where every rank bound it, else 0 on all.
    t = symm_mem.empty(1 << 16, dtype=torch.int32, device=device)
    _, mc = multicast_rendezvous(t, group.device_group, "test")
    assert bool(mc) == multicast, (mc, multicast)
    if multicast:
        if rank == world - 1:
            os.environ[ENV] = "1"  # this rank skips the binding
        t = symm_mem.empty(1 << 16, dtype=torch.int32, device=device)
        handle, mc = multicast_rendezvous(t, name, "test")
        os.environ.pop(ENV, None)
        assert mc == 0, (rank, mc)
        check_peer_stores(handle, t, world, rank, 2)

    # The decode's peer-store mailboxes bind no multicast object and map every peer.
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.topk_exchange import (  # noqa: E501
        TopkExchange,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
        DcpMailbox,
    )

    box = DcpMailbox(group.device_group, device, world, 4, 2, "bf16")
    assert box.peers.ne(0).all() and box.lse_peers.ne(0).all()
    topk = TopkExchange(group.device_group, device, world, rank, 4, 64)
    assert topk.peers.ne(0).all()
    if multicast:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.prepared import (  # noqa: E501
            OprojMailbox,
        )

        owner = group.device_group, device, world, rank
        rs = OprojMailbox(*owner, (3, world, 4, 64), multicast=False)
        latent = OprojMailbox(*owner, (3, world, 4, 64))
        assert rs.mc == 0 and all(rs.handle.buffer_ptrs)
        assert latent.mc != 0
    torch.accelerator.synchronize()
    group.barrier()


def worker(rank: int, world: int, port: int) -> None:
    update_environment_variables(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    run_rank(world)


def run_rank(world: int) -> None:
    with set_current_vllm_config(VllmConfig()):
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.accelerator.set_device_index(local_rank)
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world)
        check(world)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
def test_symm_multicast_budget():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        run_rank(int(os.environ["WORLD_SIZE"]))
        return
    world = 2
    if torch.accelerator.device_count() < world:
        pytest.skip("needs 2 GPUs")
    from vllm.utils.network_utils import get_open_port

    mp.spawn(worker, args=(world, get_open_port()), nprocs=world)


@pytest.mark.parametrize(
    ("tp", "topology", "refused"),
    [
        (1, None, False),
        (16, "INTRA_NODE", False),
        (16, "CROSS_NODE_NVLINK", False),
        (16, "UNSUPPORTED", True),
    ],
)
def test_specialized_decode_requires_shared_fabric(monkeypatch, tp, topology, refused):
    """TP ranks on different nodes without a shared NVLink fabric would block in the
    first mailbox rendezvous until the store timeout (600 s): refuse them up front."""
    import vllm.distributed
    from vllm.distributed.device_communicators import nvlink_fabric
    from vllm.models.kimi_k3.nvidia.specialized_decode import runtime

    queried = []

    def classify(group):
        queried.append(group)
        return nvlink_fabric.SymmetricMemoryTopology[topology]

    tp_group = SimpleNamespace(cpu_group=object())
    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(nvlink_fabric, "get_symmetric_memory_topology", classify)
    monkeypatch.setattr(runtime, "unsupported", lambda _config: None)
    config = SimpleNamespace(
        additional_config={"decode_backend": "latency_specialized"},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, decode_context_parallel_size=1
        ),
        speculative_config=None,
        model_config=SimpleNamespace(
            max_model_len=4096, hf_text_config=SimpleNamespace(dsa_config=None)
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2]),
    )
    if refused:
        with pytest.raises(ValueError, match="without a shared NVLink fabric"):
            runtime.specialized_decode_config(config)
    else:
        assert runtime.specialized_decode_config(config) is not None
    assert queried == ([] if tp == 1 else [tp_group.cpu_group])
