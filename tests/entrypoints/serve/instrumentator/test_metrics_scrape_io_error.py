# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client.mmap_dict import MmapedDict

from vllm.entrypoints.serve.instrumentator.metrics import attach_router


@pytest.mark.parametrize("path", ["/metrics", "/metrics/"])
@pytest.mark.parametrize("failure", ["unreadable", "truncated"])
def test_multiprocess_scrape_reports_db_read_failure(
    tmp_path, monkeypatch, caplog, path, failure
):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    db_path = tmp_path / "counter_123.db"
    db = MmapedDict(str(db_path))
    try:
        db.write_value('["test_counter", "test_counter_total", {}, "A counter"]', 2, 0)
    finally:
        db.close()

    app = FastAPI()
    attach_router(app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    with TestClient(app, raise_server_exceptions=False) as client:
        good = client.get(path, follow_redirects=False)
        assert good.status_code == HTTPStatus.OK
        assert "test_counter_total 2.0" in good.text
        assert good.headers["content-type"].startswith("text/plain")

        if failure == "unreadable":
            db_path.unlink()
            db_path.mkdir()  # The real collector finds the .db but cannot read it.
        else:
            db_path.write_bytes(b"\0")

        bad = client.get(path, follow_redirects=False)
        assert bad.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "PROMETHEUS_MULTIPROC_DIR" in bad.text
        assert str(db_path) not in bad.text
        assert client.get("/health").json() == {"status": "ok"}

    assert "Prometheus metrics scrape failed" in caplog.text
    error_name = "IsADirectoryError" if failure == "unreadable" else "struct.error"
    assert error_name in caplog.text


def test_single_process_scrape_preserves_unexpected_collector_errors(monkeypatch):
    from prometheus_client import REGISTRY

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)

    class FailingCollector:
        def describe(self):
            return []

        def collect(self):
            raise OSError("unrelated collector failure")

    app = FastAPI()
    attach_router(app)
    collector = FailingCollector()
    REGISTRY.register(collector)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            assert (
                client.get("/metrics").status_code == HTTPStatus.INTERNAL_SERVER_ERROR
            )
    finally:
        REGISTRY.unregister(collector)


def test_openai_server_serves_completions_when_metrics_db_is_unreadable(tmp_path):
    """Exercise the scrape failure through the Python serving entrypoint."""
    import requests
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from tests.utils import RemoteOpenAIServer

    model_path = tmp_path / "model"
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        LlamaForCausalLM(config).save_pretrained(model_path)
    tokenizer = Tokenizer(
        WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    ).save_pretrained(model_path)

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with RemoteOpenAIServer(
        str(model_path),
        [
            "--enforce-eager",
            "--dtype",
            "float32",
            "--max-model-len",
            "64",
            "--max-num-seqs",
            "4",
        ],
        env_dict={
            "PROMETHEUS_MULTIPROC_DIR": str(metrics_dir),
            "VLLM_CPU_KVCACHE_SPACE": "1",
            "VLLM_USE_RUST_FRONTEND": "0",
        },
    ) as server:
        url = server.url_for("metrics")
        assert requests.get(url, timeout=10).status_code == HTTPStatus.OK

        (metrics_dir / "counter_123.db").mkdir()
        response = requests.get(url, timeout=10)
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "PROMETHEUS_MULTIPROC_DIR" in response.text

        completion = server.get_client().completions.create(
            model=str(model_path), prompt="hello", max_tokens=2, temperature=0
        )
        assert completion.usage is not None
        assert completion.usage.completion_tokens > 0
