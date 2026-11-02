# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest


@pytest.fixture
def p2p_control_peer():
    """A control-only peer: delay its handshake or fail fetches on demand."""
    import queue
    import socket
    import threading

    from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection
    from vllm.v1.kv_offload.tiering.p2p.control.zmq import ZmqTransport

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    handshake = threading.Event()
    reply_to_fetch = threading.Event()
    fetches: queue.Queue[dict] = queue.Queue()
    ready = threading.Event()
    stop = threading.Event()
    errors = []

    def serve():
        transport = None
        try:
            transport = ZmqTransport(f"127.0.0.1:{port}", "127.0.0.1", port)
            connections: list[ControlConnection] = []
            acknowledged = set()
            ready.set()
            while not stop.is_set():
                connections.extend(transport.poll())
                for conn in connections:
                    if handshake.is_set() and conn.peer_id not in acknowledged:
                        conn.send(
                            {"type": "connect_ack", "peer_id": f"127.0.0.1:{port}"}
                        )
                        acknowledged.add(conn.peer_id)
                    for msg in conn.recv():
                        kind = msg["type"]
                        if kind == "fetch" and msg["keys"]:
                            fetches.put(msg)
                            if not reply_to_fetch.is_set():
                                continue
                            conn.send(
                                {
                                    "type": "transfer_done",
                                    "kv_request_id": msg["kv_request_id"],
                                    "round_seq": msg["round_seq"],
                                    "success": False,
                                }
                            )
                        elif kind == "abort_fetch":
                            conn.send(
                                {
                                    "type": "abort_ack",
                                    "kv_request_id": msg["kv_request_id"],
                                    "round_seq": msg["round_seq"],
                                }
                            )
                stop.wait(0.001)
        except Exception as exc:
            errors.append(exc)
            ready.set()
        finally:
            if transport is not None:
                transport.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(10)
    assert not errors
    try:
        yield port, handshake, reply_to_fetch, fetches
    finally:
        stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert not errors


def test_inference_reuses_a_finished_kv_request_id(
    vllm_runner, monkeypatch, p2p_control_peer
):
    """A cancellation's late failure must not suppress a later request's fetch."""
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_P2P_HANDSHAKE_TIMEOUT_S", "120")
    peer_port, handshake, reply_to_fetch, fetches = p2p_control_peer
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        local_port = sock.getsockname()[1]
    config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 16 * 1024 * 1024,
            "secondary_tiers": [
                {"type": "p2p", "host": "127.0.0.1", "port": local_port}
            ],
        },
    )
    with vllm_runner(
        "HuggingFaceTB/SmolLM2-135M",
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.35,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_transfer_config=config,
    ) as runner:
        llm = runner.llm
        params = SamplingParams(
            temperature=0,
            max_tokens=4,
            ignore_eos=True,
            extra_args={
                "kv_transfer_params": {
                    "remote_prefiller": {
                        "remote_host": "127.0.0.1",
                        "remote_port": peer_port,
                        "kv_request_id": "retired",
                    }
                }
            },
        )
        prompt = {"prompt_token_ids": [3 + i % 100 for i in range(192)]}

        handshake.set()
        request_ids = llm.enqueue([prompt], params, use_tqdm=False)
        deadline = time.monotonic() + 30
        while fetches.empty():
            assert time.monotonic() < deadline, "no initial P2P fetch"
            llm.llm_engine.step()
        fetches.get_nowait()
        llm.llm_engine.abort_request(request_ids, internal=True)
        from vllm.v1.metrics.reader import Counter

        deadline = time.monotonic() + 30
        while True:
            llm.generate(
                [{"prompt_token_ids": [0]}],
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )
            failures = [
                metric.value
                for metric in llm.get_metrics()
                if isinstance(metric, Counter)
                and "promotion_job_failures" in metric.name
            ]
            if sum(failures) > 0:
                break
            assert time.monotonic() < deadline, (
                f"cancelled load was not drained: {failures}"
            )
        reply_to_fetch.set()
        second_prompt = {"prompt_token_ids": [103 + i % 100 for i in range(192)]}
        result = llm.generate([second_prompt], params, use_tqdm=False)[0]
        assert len(result.outputs[0].token_ids) == 4
        assert not fetches.empty(), "late failure poisoned the reused KV request id"
        assert fetches.get_nowait()["kv_request_id"] == "retired"
