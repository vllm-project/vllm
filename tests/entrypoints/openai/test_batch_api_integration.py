"""Integration tests for the Online Batch API.

These tests start a vLLM server and exercise the full batch workflow
via HTTP: upload file -> create batch -> poll status -> download results.

Requires a GPU and model to be available.
"""
import json
import time

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype", "float16",
        "--max-model-len", "256",
        # opt-125m's tokenizer defines no chat template; supply one so
        # /v1/chat/completions requests can be served.
        "--chat-template", "examples/template_chatml.jinja",
        "--enable-batch-api",
        "--batch-storage-dir", "/tmp/vllm-test-batches",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def base_url(server):
    return server.url_for("")


def test_full_batch_workflow(base_url):
    """Test: upload -> create batch -> poll -> get results."""

    # 1. Upload input file
    batch_input = "\n".join([
        json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": f"Say hello {i}"}],
                "max_tokens": 10,
            },
        })
        for i in range(3)
    ])

    upload_resp = requests.post(
        f"{base_url}/v1/files",
        files={"file": ("batch.jsonl", batch_input.encode())},
        data={"purpose": "batch"},
    )
    assert upload_resp.status_code == 200
    file_obj = upload_resp.json()
    assert file_obj["id"].startswith("file-")

    # 2. Create batch
    create_resp = requests.post(
        f"{base_url}/v1/batches",
        json={
            "input_file_id": file_obj["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert create_resp.status_code == 200
    batch_obj = create_resp.json()
    assert batch_obj["id"].startswith("batch-")
    assert batch_obj["status"] in ("validating", "in_progress")

    # 3. Poll until complete
    batch_id = batch_obj["id"]
    for _ in range(60):
        status_resp = requests.get(f"{base_url}/v1/batches/{batch_id}")
        assert status_resp.status_code == 200
        batch_obj = status_resp.json()
        if batch_obj["status"] in ("completed", "failed"):
            break
        time.sleep(1)

    assert batch_obj["status"] == "completed"
    assert batch_obj["request_counts"]["completed"] == 3
    assert batch_obj["request_counts"]["failed"] == 0
    assert batch_obj["output_file_id"] is not None

    # 4. Download results
    output_resp = requests.get(
        f"{base_url}/v1/files/{batch_obj['output_file_id']}/content")
    assert output_resp.status_code == 200
    lines = output_resp.text.strip().split("\n")
    assert len(lines) == 3

    for line in lines:
        output = json.loads(line)
        assert output["response"]["status_code"] == 200
        assert output["response"]["body"]["choices"][0]["message"]["content"]


def test_list_batches(base_url):
    resp = requests.get(f"{base_url}/v1/batches")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_list_files(base_url):
    resp = requests.get(f"{base_url}/v1/files")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"


def test_get_nonexistent_batch(base_url):
    resp = requests.get(f"{base_url}/v1/batches/batch-nonexistent")
    assert resp.status_code == 404


def test_get_nonexistent_file(base_url):
    resp = requests.get(f"{base_url}/v1/files/file-nonexistent")
    assert resp.status_code == 404


def test_create_batch_invalid_file(base_url):
    resp = requests.post(
        f"{base_url}/v1/batches",
        json={
            "input_file_id": "file-nonexistent",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert resp.status_code == 404


def test_create_batch_invalid_endpoint(base_url):
    upload_resp = requests.post(
        f"{base_url}/v1/files",
        files={"file": ("batch.jsonl", b'{"custom_id":"r1","method":"POST",'
                         b'"url":"/v1/chat/completions","body":{"model":"x",'
                         b'"messages":[]}}\n')},
        data={"purpose": "batch"},
    )
    file_id = upload_resp.json()["id"]

    resp = requests.post(
        f"{base_url}/v1/batches",
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/invalid",
            "completion_window": "24h",
        },
    )
    assert resp.status_code == 400


def test_delete_file(base_url):
    upload_resp = requests.post(
        f"{base_url}/v1/files",
        files={"file": ("delete_me.jsonl", b"test\n")},
        data={"purpose": "batch"},
    )
    file_id = upload_resp.json()["id"]

    del_resp = requests.delete(f"{base_url}/v1/files/{file_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    get_resp = requests.get(f"{base_url}/v1/files/{file_id}")
    assert get_resp.status_code == 404
