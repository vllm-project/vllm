# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end test: ECMooncakeConnector producer (GPU0) publishes EC metadata and
serves pull requests; consumer (GPU1) fetches layout over HTTP and pulls the
tensor via Mooncake TransferEngine + ZMQ.

Requires: 2+ CUDA GPUs, mooncake-transfer-engine, pyzmq, httpx, fastapi, uvicorn.

Protocol: ``mooncake_protocol`` defaults to ``tcp`` in mocks unless you set
``MOONCAKE_EC_PROTOCOL=rdma`` (matches ``ec_connector_extra_config.mooncake_protocol``).
Example RDMA run::

    MOONCAKE_EC_PROTOCOL=rdma PYTHONPATH=. python tests/v1/ec_connector/integration/test_ec_mooncake_transfer_e2e.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from unittest.mock import Mock

import torch

try:
    import zmq  # noqa: F401
except ImportError as e:
    raise SystemExit("pyzmq is required: pip install pyzmq") from e
try:
    import mooncake  # noqa: F401
except ImportError as e:
    raise SystemExit("mooncake-transfer-engine is required") from e

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
)


def _find_free_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _mock_vllm_producer(registry_port: int) -> Mock:
    cfg = Mock(spec=VllmConfig)
    cfg.parallel_config = Mock()
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.ec_transfer_config = Mock()
    cfg.ec_transfer_config.is_ec_producer = True
    cfg.ec_transfer_config.is_ec_consumer = False
    cfg.ec_transfer_config.ec_buffer_device = "cuda"
    cfg.ec_transfer_config.ec_connector_extra_config = {
        "mooncake_protocol": os.environ.get("MOONCAKE_EC_PROTOCOL", "tcp"),
        "registry_http_port": registry_port,
    }
    return cfg


def _mock_vllm_consumer() -> Mock:
    cfg = Mock(spec=VllmConfig)
    cfg.parallel_config = Mock()
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.ec_transfer_config = Mock()
    cfg.ec_transfer_config.is_ec_producer = False
    cfg.ec_transfer_config.is_ec_consumer = True
    cfg.ec_transfer_config.ec_buffer_device = "cuda"
    cfg.ec_transfer_config.ec_connector_extra_config = {
        "mooncake_protocol": os.environ.get("MOONCAKE_EC_PROTOCOL", "tcp"),
        "remote_registry_url": "http://unused-on-worker",
    }
    return cfg


def _producer_entry(
    mm_hash: str,
    registry_port: int,
    ready: mp.Queue,
    done: mp.Event,
    barrier: mp.Barrier,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.init()
    cfg = _mock_vllm_producer(registry_port)
    conn = ECMooncakeConnector(cfg, ECConnectorRole.WORKER)
    torch.manual_seed(12345)
    tensor = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    cache = {mm_hash: tensor}
    conn.save_caches(cache, mm_hash)
    ready.put("ok")
    barrier.wait(timeout=120)
    # Hold process until consumer finishes transfer
    done.wait(timeout=180)


def _consumer_entry(
    mm_hash: str,
    registry_url: str,
    barrier: mp.Barrier,
    result_queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.init()
    barrier.wait(timeout=120)
    import httpx

    url = f"{registry_url.rstrip('/')}/ec/info/{mm_hash}"
    for _ in range(60):
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    else:
        result_queue.put({"ok": False, "err": "registry never ready"})
        return
    data = r.json()
    spec = ECMooncakeLoadSpec(
        mm_hash=mm_hash,
        num_token=1,
        nbytes=int(data["nbytes"]),
        shape=tuple(int(x) for x in data["shape"]),
        dtype=str(data["dtype"]),
        producer_zmq=str(data["producer_zmq"]),
    )
    meta = ECMooncakeConnectorMetadata()
    meta.add_load(spec)
    cfg = _mock_vllm_consumer()
    conn = ECMooncakeConnector(cfg, ECConnectorRole.WORKER)
    conn.bind_connector_metadata(meta)
    enc: dict[str, torch.Tensor] = {}
    try:
        conn.start_load_caches(enc)
    except Exception as e:
        result_queue.put({"ok": False, "err": repr(e)})
        return
    got = enc.get(mm_hash)
    if got is None:
        result_queue.put({"ok": False, "err": "missing tensor"})
        return
    torch.manual_seed(12345)
    expected = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    max_diff = (got.cpu() - expected.cpu()).abs().max().item()
    result_queue.put({"ok": True, "max_diff": max_diff})


def test_ec_mooncake_two_process_transfer():
    """Producer on cuda:0 and consumer on cuda:1 transfer one EC tensor."""
    mm_hash = "e2e_mm_test_hash"
    registry_port = _find_free_port()
    registry_url = f"http://127.0.0.1:{registry_port}"

    ctx = mp.get_context("spawn")
    ready: mp.Queue = ctx.Queue()
    result: mp.Queue = ctx.Queue()
    done = ctx.Event()
    barrier = ctx.Barrier(2)
    prod = ctx.Process(
        target=_producer_entry,
        args=(mm_hash, registry_port, ready, done, barrier),
        daemon=True,
    )
    cons = ctx.Process(
        target=_consumer_entry,
        args=(mm_hash, registry_url, barrier, result),
        daemon=True,
    )
    prod.start()
    assert ready.get(timeout=120) == "ok"
    cons.start()
    cons.join(timeout=180)
    done.set()
    prod.join(timeout=30)

    assert not cons.is_alive(), "consumer process hung"
    assert cons.exitcode == 0, f"consumer exit {cons.exitcode}"
    out = result.get(timeout=1)
    assert out["ok"], out.get("err", out)
    assert out["max_diff"] < 1e-4, f"tensor mismatch max_diff={out['max_diff']}"


def _main() -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("Need at least 2 CUDA devices for this e2e test.")
    test_ec_mooncake_two_process_transfer()
    print("ECMooncake two-process transfer e2e: PASSED")


if __name__ == "__main__":
    _main()
