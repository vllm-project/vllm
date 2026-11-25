# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner buffers survive skewed ranks, graph-size switches and repeated reuse."""

import random
from datetime import timedelta
from functools import partial

import pytest
import torch
import torch.distributed as dist

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both"])
@pytest.mark.parametrize(
    "connector,extra,device,expected",
    [
        ("NixlConnector", {}, "cuda", True),
        ("NixlConnector", {"transfer_mode": "pull"}, "cuda", True),
        ("NixlConnector", {"transfer_mode": "push"}, "cuda", True),
        ("NixlPullConnector", {}, "cuda", True),
        ("NixlPushConnector", {}, "cuda", True),
        ("NixlConnector", {}, "cpu", True),
        ("OffloadingConnector", {}, "cuda", True),
        ("SimpleCPUOffloadConnector", {}, "cuda", True),
        ("LMCacheConnectorV1", {"use_layerwise": False}, "cuda", True),
        ("LMCacheConnectorV1", {"use_layerwise": True}, "cuda", False),
    ],
)
def test_dep_kv_transfer_scope(role, connector, extra, device, expected):
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.config import (
        supports_kv_transfer,
    )

    config = KVTransferConfig(
        kv_connector=connector,
        kv_role=role,
        kv_connector_extra_config=extra,
        kv_buffer_device=device,
    )
    assert supports_kv_transfer(config) is expected


@pytest.mark.parametrize("empty_config", [False, True])
def test_dep_no_kv_transfer(empty_config):
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.config import (
        supports_kv_transfer,
    )

    assert supports_kv_transfer(KVTransferConfig() if empty_config else None)


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_dep_multi_connector_checks_every_child(layerwise, nested):
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.config import (
        supports_kv_transfer,
    )

    child = {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"use_layerwise": layerwise},
    }
    if nested:
        child = {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"connectors": [child]},
        }
    config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "NixlPushConnector", "kv_role": "kv_consumer"},
                child,
            ]
        },
    )
    assert supports_kv_transfer(config) is (not layerwise)


@pytest.mark.parametrize("layerwise", [False, True])
def test_dep_external_connector_uses_graph_capability(layerwise):
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.config import (
        supports_kv_transfer,
    )

    config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_module_path=(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"
        ),
        kv_connector_extra_config={"use_layerwise": layerwise},
    )
    assert supports_kv_transfer(config) is (not layerwise)


def test_dep_kv_transfer_preserves_factory_validation():
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.config import (
        supports_kv_transfer,
    )

    config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config={"transfer_mode": "invalid"},
    )
    with pytest.raises(ValueError, match="Invalid NixlConnector transfer_mode"):
        supports_kv_transfer(config)


@pytest.mark.parametrize("width,limit", [(2, 128), (4, 128), (8, 384)])
def test_dep_row_limit(width, limit):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep import (
        MAX_ROWS_BY_WIDTH,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.owner import (
        owner_slice,
        uniform_owner_layout,
    )

    assert MAX_ROWS_BY_WIDTH[width] == limit
    for rows in (1, 128, 129, 192, 384, 385):
        assert uniform_owner_layout([rows] * width, rows, width) is (rows <= limit)
        if rows <= limit:
            assert owner_slice(rows * width - 1, rows, width) == (width - 1, rows - 1)
        else:
            with pytest.raises(ValueError, match="uniform EP"):
                owner_slice(0, rows, width)
    assert not uniform_owner_layout([0] * width, 0, width)
    assert not uniform_owner_layout([limit] * (width - 1) + [0], limit, width)


@pytest.mark.skipif(
    not current_platform.is_device_capability(103), reason="needs SM103"
)
def test_local_mhc_cluster_entry():
    """Also run under racecheck: DSM peers must exist before the first write."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.shared_boundary import (
        FusedPostPre,
    )

    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
        norm = RMSNorm(5120, eps=1e-6).to("cuda", torch.bfloat16)
        x = torch.full((1, 5120), 0.5, device="cuda", dtype=torch.bfloat16)
        residual = torch.full((1, 4, 5120), 0.25, device="cuda", dtype=torch.bfloat16)
        post = torch.full((1, 4, 1), 0.5, device="cuda")
        comb = torch.eye(4, device="cuda").unsqueeze(0)
        pre = torch.full((1, 4), 0.25, device="cuda")
        call = FusedPostPre()
        args = (x, residual, post, comb, pre, norm, None, True)
        call(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call(*args)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.all(output[0] == 0.5)
        assert torch.all(output[1] == 1)
        assert torch.all(output[2] == 0.5)


def _worker(rank, width, port, capacity):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary import (
        empty_quantized_layer_input,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep import shared_boundary
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.owner import (
        EPReduceScatterMHC,
        uniform_owner_layout,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.transport import (
        FinalizeOwnerPublish,
        PeerFrontDispatch,
    )

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=width,
        timeout=timedelta(seconds=300),
    )
    device = torch.device("cuda", rank)
    try:
        with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
            owner = EPReduceScatterMHC(dist.group.WORLD, device, capacity)
            storage = PeerFrontDispatch(device, dist.group.WORLD, capacity)
            norm = RMSNorm(5120, eps=1e-6).to(device=device, dtype=torch.bfloat16)
            reference = shared_boundary.FusedPostPre()
            errors = torch.zeros((), dtype=torch.int64, device=device)
            counter = torch.zeros((), dtype=torch.int32, device=device)
            graphs = []
            row_counts = (1, 6, 12, 48, 128)
            if capacity > 128:
                row_counts = (1, 126, 129, 192, 288, 384)
            for rows in row_counts:
                assert uniform_owner_layout([rows] * width, rows, width)
                assert not uniform_owner_layout([0] + [rows] * (width - 1), rows, width)
                assert not uniform_owner_layout(
                    [rows + 1] + [rows] * (width - 1), rows, width
                )
                total = rows * width
                channel = (
                    PeerFrontDispatch(device, dist.group.WORLD, rows, storage)
                    if rows % 2 == 0
                    else None
                )
                residual = torch.full(
                    (rows, 4, 5120), 0.25, dtype=torch.bfloat16, device=device
                )
                post = torch.full((rows, 4, 1), 0.5, dtype=torch.float32, device=device)
                comb = torch.eye(4, device=device).expand(rows, -1, -1).contiguous()
                pre = torch.full((rows, 4), 0.25, dtype=torch.float32, device=device)
                shared = torch.full(
                    (rows, 5120), 0.125, dtype=torch.bfloat16, device=device
                )
                partials = torch.empty(
                    (total, 5120), dtype=torch.bfloat16, device=device
                )
                expected = torch.empty(
                    (rows, 5120), dtype=torch.bfloat16, device=device
                )
                pattern = (
                    torch.arange(total * 6, device=device)[:, None] % 16 * 0.125
                    + torch.arange(5120, device=device)[None, :] % 8 * 0.03125
                )
                local_pattern = pattern[:total][rank * rows : (rank + 1) * rows]
                local_routes = pattern.view(total, 6, 5120)[
                    rank * rows : (rank + 1) * rows
                ]
                expected_sum = torch.empty(
                    (rows, 5120), dtype=torch.float32, device=device
                )
                routed = UnfinalizedMoEOutput(
                    torch.empty((total * 6, 5120), dtype=torch.bfloat16, device=device),
                    torch.full((total, 6), 0.125, dtype=torch.float32, device=device),
                    torch.arange(total * 6, dtype=torch.int32, device=device).view(
                        total, 6
                    ),
                )
                # Each source owns disjoint expert contributions; include absent
                # routes and owners receiving zero contributions from some peers.
                route_ids = torch.arange(6, device=device)[None, :].expand(total, -1)
                routed.expanded_idx_to_permuted_idx.masked_fill_(
                    route_ids % width != rank, -1
                )
                publisher = FinalizeOwnerPublish(owner, routed, rows)
                q = empty_quantized_layer_input(rows, 5120, device)
                qref = empty_quantized_layer_input(rows, 5120, device)
                for deferred in (False, True):
                    for shared_input in (None, shared):
                        quant = (q.data.view(torch.uint8), q.scale)
                        quant_ref = (qref.data.view(torch.uint8), qref.scale)
                        call = owner.prepare(
                            partials,
                            residual,
                            post,
                            comb,
                            pre,
                            norm,
                            quant,
                            True,
                            shared=shared_input,
                            publish_override=partial(publisher, routed)
                            if deferred
                            else None,
                        )
                        reference(
                            expected,
                            residual,
                            post,
                            comb,
                            pre,
                            norm,
                            quant_ref,
                            True,
                            shared=shared_input,
                        )
                        dist.barrier()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            counter.add_(1)
                            value = (counter % 16).float() * 0.125
                            if channel is not None:
                                for plane in channel.local_output:
                                    plane.view(torch.uint8).fill_(rank + 1)
                                gathered = channel.gather_before_owner()
                                for plane in gathered:
                                    for source in range(width):
                                        errors.add_(
                                            (
                                                plane[
                                                    source * rows : (source + 1) * rows
                                                ].view(torch.uint8)
                                                != source + 1
                                            ).sum()
                                        )
                            torch.cuda._sleep(20_000 * rank)
                            expected_sum.zero_()
                            if deferred:
                                routed.gemm2_permuted.copy_(
                                    pattern + value + rank * 0.125
                                )
                                for source in range(width):
                                    owned = (
                                        torch.arange(6, device=device) % width == source
                                    )[None, :, None]
                                    contribution = torch.where(
                                        owned,
                                        (local_routes + value + source * 0.125)
                                        .bfloat16()
                                        .float()
                                        * 0.125,
                                        0.0,
                                    ).sum(dim=1)
                                    expected_sum.add_(contribution.bfloat16().float())
                            else:
                                partials.copy_(pattern[:total] + value + rank * 0.125)
                                for source in range(width):
                                    expected_sum.add_(
                                        (local_pattern + value + source * 0.125)
                                        .bfloat16()
                                        .float()
                                    )
                            expected.copy_(expected_sum)
                            got = call()
                            want = reference(
                                expected,
                                residual,
                                post,
                                comb,
                                pre,
                                norm,
                                quant_ref,
                                True,
                                shared=shared_input,
                            )
                            for actual, ref in zip(got, want):
                                errors.add_((actual != ref).sum())
                            errors.add_(
                                (
                                    q.data.view(torch.uint8)
                                    != qref.data.view(torch.uint8)
                                ).sum()
                            )
                            errors.add_((q.scale != qref.scale).sum())
                        # Keep all graph input/output allocations alive.
                        graphs.append(
                            (
                                graph,
                                (
                                    call,
                                    got,
                                    want,
                                    routed,
                                    channel,
                                    q,
                                    qref,
                                    partials,
                                    expected,
                                    shared,
                                    residual,
                                    post,
                                    comb,
                                    pre,
                                    pattern,
                                    expected_sum,
                                ),
                            )
                        )
            order = random.Random(927).choices(range(len(graphs)), k=200)
            dist.barrier()
            for index in order:
                graphs[index][0].replay()
            torch.accelerator.synchronize()
            assert errors.item() == 0, (rank, width, errors.item())
            print(
                f"DEP{width} rank {rank}: 200 graph replays, zero mismatches",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.distributed
@pytest.mark.skipif(
    not current_platform.is_device_capability(103), reason="needs SM103"
)
def test_owner_transport_graph_reuse(width):
    if torch.accelerator.device_count() < width:
        pytest.skip(f"needs {width} isolated GPUs")
    torch.multiprocessing.spawn(
        _worker, args=(width, get_open_port(), 128), nprocs=width
    )


@pytest.mark.distributed
@pytest.mark.skipif(
    not current_platform.is_device_capability(103), reason="needs SM103"
)
def test_owner_transport_large_graph_reuse():
    width = 8
    if torch.accelerator.device_count() < width:
        pytest.skip(f"needs {width} isolated GPUs")
    torch.multiprocessing.spawn(
        _worker, args=(width, get_open_port(), 384), nprocs=width
    )
