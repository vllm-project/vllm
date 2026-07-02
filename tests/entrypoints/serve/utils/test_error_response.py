# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

import pytest

from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMUnprocessableEntityError
from vllm.multimodal.media.connector import _handle_fetch_exception

pytestmark = pytest.mark.skip_global_cleanup


def test_vllm_unprocessable_entity_error_str():
    err = VLLMUnprocessableEntityError(
        "Failed to load image",
        parameter="image_url",
        value="http://invalid.com/img.png",
    )
    assert "Failed to load image" in str(err)
    assert "parameter=image_url" in str(err)
    assert "value=http://invalid.com/img.png" in str(err)


def test_create_error_response_for_unprocessable_entity():
    exc = VLLMUnprocessableEntityError(
        "Invalid image source",
        parameter="image_url",
        value="http://invalid.com/img.png",
    )
    resp = create_error_response(exc)

    assert resp.error.type == "UnprocessableEntityError"
    assert resp.error.code == HTTPStatus.UNPROCESSABLE_ENTITY.value
    assert resp.error.param == "image_url"
    assert "Invalid image source" in resp.error.message


def test_handle_fetch_exception_http_4xx():
    import requests

    # Create dummy response
    response_404 = requests.Response()
    response_404.status_code = 404
    e = requests.exceptions.HTTPError("404 Client Error", response=response_404)

    with pytest.raises(VLLMUnprocessableEntityError) as excinfo:
        _handle_fetch_exception(e, "http://404.com/img.jpg", "image_url")

    assert excinfo.value.parameter == "image_url"
    assert excinfo.value.value == "http://404.com/img.jpg"
    assert "Failed to fetch media from URL" in str(excinfo.value)


def test_handle_fetch_exception_http_5xx_propagates():
    import requests

    response_500 = requests.Response()
    response_500.status_code = 500
    e = requests.exceptions.HTTPError("500 Server Error", response=response_500)

    # 5xx server errors should not be converted to 422, they should propagate as is
    with pytest.raises(requests.exceptions.HTTPError):
        _handle_fetch_exception(e, "http://500.com/img.jpg", "image_url")


def test_handle_fetch_exception_connection_error():
    import requests

    e = requests.exceptions.ConnectionError("Failed to connect")

    with pytest.raises(VLLMUnprocessableEntityError) as excinfo:
        _handle_fetch_exception(e, "http://dns-failure.com/img.jpg", "image_url")

    assert excinfo.value.parameter == "image_url"
    assert "Failed to connect" in str(excinfo.value)
