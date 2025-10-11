# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.connections import global_http_connection


@pytest.mark.asyncio
async def test_async_client_preserves_encoded_path():
    """
    Test that the AsyncHttpClient preserves encoded characters in the URL path.
    This is crucial for services like AWS S3 that use signed URLs.
    """
    # The path component '/path%2Fwith%2Fencoded%2Fslash' should be sent as-is.
    # httpbin.org/anything will echo back the request details.
    url_with_encoded_slash = "http://httpbin.org/anything/path%2Fwith%2Fencoded%2Fslash?query=1"
    expected_path = "/anything/path%2Fwith%2Fencoded%2Fslash"

    response = await global_http_connection.get_async_response(url_with_encoded_slash)
    
    response.raise_for_status()
    data = await response.json()

    # Assert that the path received by the server matches the original, encoded path.
    assert data.get("path") == expected_path, (
        f"URL path was not preserved. Expected '{expected_path}', "
        f"but got '{data.get('path')}'."
    )
