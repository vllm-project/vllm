import hashlib
from starlette.datastructures import Headers

from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware


def make_middleware(tokens):
    return AuthenticationMiddleware(app=None, tokens=tokens)


def test_bearer_token_still_works():
    mw = make_middleware(["secret123"])
    headers = Headers({"Authorization": "Bearer secret123"})
    assert mw.verify_token(headers) is True


def test_x_api_key_now_works():
    mw = make_middleware(["secret123"])
    headers = Headers({"x-api-key": "secret123"})
    assert mw.verify_token(headers) is True
def test_wrong_scheme_falls_through_to_x_api_key():
    mw = make_middleware(["secret123"])
    headers = Headers({"Authorization": "Basic wrongscheme", "x-api-key": "secret123"})
    assert mw.verify_token(headers) is True

def test_wrong_x_api_key_rejected():
    mw = make_middleware(["secret123"])
    headers = Headers({"x-api-key": "wrong-token"})
    assert mw.verify_token(headers) is False


def test_no_headers_rejected():
    mw = make_middleware(["secret123"])
    headers = Headers({})
    assert mw.verify_token(headers) is False

import hashlib
from starlette.datastructures import Headers

from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware


def make_middleware(tokens):
    return AuthenticationMiddleware(app=None, tokens=tokens)


def test_bearer_token_still_works():
    mw = make_middleware(["secret123"])
    headers = Headers({"Authorization": "Bearer secret123"})
    assert mw.verify_token(headers) is True


def test_x_api_key_now_works():
    mw = make_middleware(["secret123"])
    headers = Headers({"x-api-key": "secret123"})
    assert mw.verify_token(headers) is True


def test_wrong_x_api_key_rejected():
    mw = make_middleware(["secret123"])
    headers = Headers({"x-api-key": "wrong-token"})
    assert mw.verify_token(headers) is False


def test_no_headers_rejected():
    mw = make_middleware(["secret123"])
    headers = Headers({})
    assert mw.verify_token(headers) is False
