import pytest

from vllm.utils.request_id import normalize_request_id


TRANSPORT_REQUEST_ID = (
    "___prefill_addr_10.0.0.1:31000___decode_addr_10.0.0.2:32000_deadbeef"
)


@pytest.mark.parametrize(
    ("request_id", "expected"),
    [
        (
            f"cmpl-{TRANSPORT_REQUEST_ID}-0-a1b2c3d4",
            TRANSPORT_REQUEST_ID,
        ),
        (
            f"chatcmpl-{TRANSPORT_REQUEST_ID}-a1b2c3d4",
            TRANSPORT_REQUEST_ID,
        ),
        (
            f"generate-tokens-{TRANSPORT_REQUEST_ID}-a1b2c3d4",
            TRANSPORT_REQUEST_ID,
        ),
        (TRANSPORT_REQUEST_ID, TRANSPORT_REQUEST_ID),
        (
            f"chatcmpl-{TRANSPORT_REQUEST_ID}_1-a1b2c3d4",
            TRANSPORT_REQUEST_ID,
        ),
        ("cmpl-not-a-transport-id-0-a1b2c3d4", "cmpl-not-a-transport-id-0-a1b2c3d4"),
        (
            "cmpl-___prefill_addr_10.0.0.1:31000___decode_addr_10.0.0.2:32000_-0-a1b2c3d4",
            "cmpl-___prefill_addr_10.0.0.1:31000___decode_addr_10.0.0.2:32000_-0-a1b2c3d4",
        ),
    ],
)
def test_normalize_request_id(request_id: str, expected: str) -> None:
    assert normalize_request_id(request_id) == expected
    assert normalize_request_id(normalize_request_id(request_id)) == expected
