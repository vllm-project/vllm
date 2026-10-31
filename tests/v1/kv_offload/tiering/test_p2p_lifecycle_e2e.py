# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time


def _p2p_inference_kwargs(port):
    from vllm.config import KVTransferConfig

    return dict(
        model="HuggingFaceTB/SmolLM2-135M",
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.30,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_load_failure_policy="recompute",
            kv_connector_extra_config={
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": 16 * 1024 * 1024,
                "secondary_tiers": [{"type": "p2p", "host": "127.0.0.1", "port": port}],
            },
        ),
    )


def _serve_p2p_inference_source(port, ready, stop):
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(**_p2p_inference_kwargs(port))
    try:
        prompt = TokensPrompt(prompt_token_ids=[3 + i % 100 for i in range(192)])
        params = SamplingParams(
            temperature=0,
            max_tokens=4,
            ignore_eos=True,
            extra_args={
                "kv_transfer_params": {
                    "remote_decoder": {
                        "kv_request_id": "buffered",
                    }
                }
            },
        )
        reference = llm.generate([prompt], params, use_tqdm=False)[0]
        ready.put(reference.outputs[0].token_ids)
        deadline = time.monotonic() + 180
        while not stop.is_set():
            assert time.monotonic() < deadline, "consumer did not finish"
            llm.generate(
                [TokensPrompt(prompt_token_ids=[0])],
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )
    finally:
        llm.llm_engine.engine_core.shutdown()


def test_inference_uses_completion_buffered_after_disconnect(vllm_runner, monkeypatch):
    """A completed NIXL transfer must not become a failed promotion on disconnect."""
    import multiprocessing
    import socket
    import threading

    import msgspec.msgpack
    import zmq
    import zmq.utils.monitor

    from vllm import SamplingParams, TokensPrompt
    from vllm.v1.metrics.reader import Counter

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    ports = []
    for _ in range(2):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    source_id = f"127.0.0.1:{ports[0]}".encode()
    target_endpoint = f"tcp://127.0.0.1:{ports[1]}".encode()
    buffered = threading.Event()
    recv = zmq.Socket.recv_multipart
    recv_monitor = zmq.utils.monitor.recv_monitor_message
    injected = False

    def receive_both_before_dispatch(sock, *args, **kwargs):
        frames = recv(sock, *args, **kwargs)
        if (
            sock.getsockopt(zmq.TYPE) == zmq.ROUTER
            and sock.getsockopt(zmq.LAST_ENDPOINT) == target_endpoint
            and frames[0] == source_id
            and msgspec.msgpack.decode(frames[-1]).get("type") == "transfer_done"
        ):
            buffered.set()
        return frames

    def disconnect_after_buffering(sock, *args, **kwargs):
        nonlocal injected
        if buffered.is_set() and not injected:
            injected = True
            return {"event": zmq.EVENT_DISCONNECTED}
        return recv_monitor(sock, *args, **kwargs)

    monkeypatch.setattr(zmq.Socket, "recv_multipart", receive_both_before_dispatch)
    monkeypatch.setattr(
        zmq.utils.monitor, "recv_monitor_message", disconnect_after_buffering
    )
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    stop = ctx.Event()
    producer = ctx.Process(
        target=_serve_p2p_inference_source, args=(ports[0], ready, stop)
    )
    producer.start()
    try:
        kwargs = _p2p_inference_kwargs(ports[1])
        model = kwargs.pop("model")
        with vllm_runner(model, **kwargs) as runner:
            reference = ready.get(timeout=180)
            prompt = TokensPrompt(prompt_token_ids=[3 + i % 100 for i in range(192)])
            params = SamplingParams(
                temperature=0,
                max_tokens=4,
                ignore_eos=True,
                extra_args={
                    "kv_transfer_params": {
                        "remote_prefiller": {
                            "remote_host": "127.0.0.1",
                            "remote_port": ports[0],
                            "kv_request_id": "buffered",
                        }
                    }
                },
            )
            result = runner.llm.generate([prompt], params, use_tqdm=False)[0]
            assert buffered.is_set()
            assert result.outputs[0].token_ids == reference
            failures = [
                metric.value
                for metric in runner.llm.get_metrics()
                if isinstance(metric, Counter)
                and "promotion_job_failures" in metric.name
            ]
            assert sum(failures) == 0, failures
    finally:
        stop.set()
        producer.join(timeout=30)
        if producer.is_alive():
            producer.terminate()
            producer.join(timeout=10)
        ready.close()
    assert producer.exitcode == 0
