# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """Start a vLLM render server with the object storage routes enabled."""
    with RemoteLaunchRenderServer(MODEL_NAME, []) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    """Create an async HTTP client for the running server."""
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


@pytest.mark.asyncio
class TestUpload:
    """Tests for the upload endpoints (PUT /object_storage and /object_storage/{uuid})."""

    async def test_upload_auto(self, client: httpx.AsyncClient):
        """Upload a file without specifying a UUID; server generates one."""
        content = b"Hello, world!"
        files = {"file": ("test.txt", content, "text/plain")}
        resp = await client.put("/object_storage", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert "uuid" in data
        # Validate that the returned string is a proper UUID
        uuid.UUID(data["uuid"])

    async def test_upload_with_uuid(self, client: httpx.AsyncClient):
        """Upload a file with a user‑provided UUID."""
        custom_uuid = str(uuid.uuid4())
        content = b"Custom UUID content"
        files = {"file": ("custom.txt", content, "text/plain")}
        resp = await client.put(f"/object_storage/{custom_uuid}", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["uuid"] == custom_uuid

    async def test_upload_invalid_uuid(self, client: httpx.AsyncClient):
        """Upload with an invalid UUID format returns 400."""
        invalid = "not-a-uuid"
        files = {"file": ("bad.txt", b"data", "text/plain")}
        resp = await client.put(f"/object_storage/{invalid}", files=files)
        assert resp.status_code == 400
        assert "Invalid UUID format" in resp.text


@pytest.mark.asyncio
class TestDownload:
    """Tests for the download endpoint (GET /object_storage/{uuid})."""

    async def test_download_success(self, client: httpx.AsyncClient):
        """Upload a file, then download it and verify content."""
        custom_uuid = str(uuid.uuid4())
        content = b"Download test content"
        files = {"file": ("down.txt", content, "text/plain")}
        upload_resp = await client.put(f"/object_storage/{custom_uuid}", files=files)
        assert upload_resp.status_code == 200

        download_resp = await client.get(f"/object_storage/{custom_uuid}")
        assert download_resp.status_code == 200
        assert download_resp.content == content

    async def test_download_not_found(self, client: httpx.AsyncClient):
        """Download a non‑existent UUID returns 404."""
        non_existent = str(uuid.uuid4())
        resp = await client.get(f"/object_storage/{non_existent}")
        assert resp.status_code == 404
        assert "Object not found" in resp.text


@pytest.mark.asyncio
class TestDelete:
    """Tests for the delete endpoint (DELETE /object_storage/{uuid})."""

    async def test_delete_success(self, client: httpx.AsyncClient):
        """Upload, delete, then verify the object is gone."""
        custom_uuid = str(uuid.uuid4())
        content = b"Delete me"
        files = {"file": ("del.txt", content, "text/plain")}
        upload_resp = await client.put(f"/object_storage/{custom_uuid}", files=files)
        assert upload_resp.status_code == 200

        delete_resp = await client.delete(f"/object_storage/{custom_uuid}")
        assert delete_resp.status_code == 200

        # Confirm deletion
        get_resp = await client.get(f"/object_storage/{custom_uuid}")
        assert get_resp.status_code == 404

    async def test_delete_not_found(self, client: httpx.AsyncClient):
        """Delete a non‑existent UUID returns 404."""
        non_existent = str(uuid.uuid4())
        resp = await client.delete(f"/object_storage/{non_existent}")
        assert resp.status_code == 404
        assert "Object not found" in resp.text


@pytest.mark.asyncio
class TestInfo:
    """Tests for the info endpoint (HEAD /object_storage/{uuid})."""

    async def test_info_success(self, client: httpx.AsyncClient):
        """HEAD request returns status 200 for an existing object."""
        custom_uuid = str(uuid.uuid4())
        content = b"Info test"
        files = {"file": ("info.txt", content, "text/plain")}
        upload_resp = await client.put(f"/object_storage/{custom_uuid}", files=files)
        assert upload_resp.status_code == 200

        head_resp = await client.head(f"/object_storage/{custom_uuid}")
        assert head_resp.status_code == 200

    async def test_info_not_found(self, client: httpx.AsyncClient):
        """HEAD on a non‑existent UUID returns 404."""
        non_existent = str(uuid.uuid4())
        resp = await client.head(f"/object_storage/{non_existent}")
        assert resp.status_code == 404
