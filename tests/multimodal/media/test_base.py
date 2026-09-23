# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vllm.multimodal.media import DecodeSpec, MediaIO, MediaRef
from vllm.multimodal.media.base import derive_media_key

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


class _CountingDecoder:
    """Decoder that records how many times it ran."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


class _StubMediaIO(MediaIO[str]):
    """MediaIO whose refs are built exactly like the production ones."""

    def load_bytes(self, data: bytes) -> str:
        return data.decode()

    def load_base64(self, media_type: str, data: str) -> str:
        raise NotImplementedError

    def load_file(self, filepath: Path) -> str:
        return self.load_bytes(filepath.read_bytes())


class _RawBytesMediaIO(MediaIO[bytes]):
    """MediaIO that hands the encoded payload straight to the processor.

    This is the shape a model uses when it decodes the media itself (e.g.
    Dots3Note reads frames *and* the audio track out of the video container).
    """

    def load_bytes(self, data: bytes) -> bytes:
        return data

    def load_base64(self, media_type: str, data: str) -> bytes:
        raise NotImplementedError

    def load_file(self, filepath: Path) -> bytes:
        return self.load_bytes(filepath.read_bytes())


def test_decoded_bytes_survive_release():
    """A decoder that yields the payload must not depend on `MediaRef.data`.

    `release()` drops the encoded bytes once hashing is done, so a model that
    needs them has to receive them as the *decoded* value. Reading `ref.data`
    instead makes the result depend on when release ran -- i.e. on cache
    eviction timing -- which is why it is not a supported way to get bytes.
    """
    payload = b"encoded-video-container"
    ref = _RawBytesMediaIO().load_bytes_ref(payload)

    assert ref.decode() == payload

    ref.release()
    assert ref.data == b""
    assert ref.decode() == payload


def test_media_ref_decodes_on_first_access():
    """A ref must not decode on construction, repr, or bytes access."""
    decoder = _CountingDecoder("decoded")
    ref = MediaRef(decoder, b"raw-bytes")

    assert not ref.is_decoded
    assert repr(ref) == "<MediaRef undecoded, 9 bytes>"
    assert ref.data == b"raw-bytes"
    assert decoder.calls == 0

    assert ref.decode() == "decoded"
    assert ref.is_decoded
    assert repr(ref) == "<MediaRef decoded, 9 bytes>"

    # Decoding is idempotent: the result is cached.
    assert ref.decode() == "decoded"
    assert decoder.calls == 1


def test_media_ref_concurrent_decode_runs_once():
    """Concurrent access from multiple threads decodes exactly once."""
    start = threading.Barrier(8)
    decoder_calls = 0
    lock = threading.Lock()

    def decode():
        nonlocal decoder_calls
        # Keep the decode window open so the other threads contend on it.
        time.sleep(0.05)
        with lock:
            decoder_calls += 1
        return "decoded"

    ref = MediaRef(decode, b"raw")

    def concurrent_decode():
        start.wait(timeout=10)
        return ref.decode()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: concurrent_decode(), range(8)))

    assert results == ["decoded"] * 8
    assert decoder_calls == 1


def test_media_ref_map_stacks_transform_and_shares_bytes():
    """map() defers the transform until decode and shares the encoded bytes."""
    decoder = _CountingDecoder(2)
    ref = MediaRef(decoder, b"raw")
    mapped = ref.map(lambda x: x * 10)

    assert mapped.data is ref.data
    assert mapped.key == ref.key  # no new decode settings, so no new identity
    assert decoder.calls == 0

    assert mapped.decode() == 20
    # The mapped ref decodes the underlying one exactly once.
    assert decoder.calls == 1
    assert ref.is_decoded
    assert ref.decode() == 2


def test_media_ref_map_extends_the_key():
    """Settings passed to map() join the spec, so the transform's parameters
    are part of the cache identity -- deterministically, and without
    re-digesting the payload."""
    ref = MediaRef(_CountingDecoder(1), b"raw", DecodeSpec({"a": 1}))
    mapped = ref.map(lambda x: x, b=2)

    assert mapped.spec.settings == {"a": 1, "b": 2}
    assert mapped.key != ref.key

    same = MediaRef(_CountingDecoder(1), b"raw", DecodeSpec({"a": 1}))
    assert same.map(lambda x: x, b=2).key == mapped.key
    assert same.map(lambda x: x, b=3).key != mapped.key


def test_media_ref_release_frees_the_encoded_bytes():
    """release() must drop every pin on the payload, not just one of them.

    The decode closure holds the bytes as well as the ref does, so clearing
    only the ref's own slot would leave the payload resident.
    """
    data = b"payload" * 64
    baseline = sys.getrefcount(data)

    ref = _StubMediaIO().load_bytes_ref(data)
    assert sys.getrefcount(data) > baseline

    ref.decode()
    ref.release()
    gc.collect()

    assert ref.data == b""
    assert sys.getrefcount(data) == baseline
    # The decoded value survives the release.
    assert ref.decode() == data.decode()


def test_media_ref_release_before_decode_is_final():
    """A released ref can no longer be decoded, and says so."""
    ref = MediaRef(_CountingDecoder("decoded"), b"raw-bytes")

    ref.release()
    assert ref.data == b""

    with pytest.raises(RuntimeError, match="released"):
        ref.decode()


def test_media_ref_key_is_stable_and_spec_scoped():
    data = b"encoded-payload"
    spec = DecodeSpec({"backend": "pyav"})

    def key(payload=data, decode_spec=None):
        return MediaRef(lambda: None, payload, decode_spec).key

    assert key() == key()
    assert key(decode_spec=spec) == key(decode_spec=DecodeSpec({"backend": "pyav"}))
    assert key(decode_spec=spec) != key(decode_spec=DecodeSpec({"backend": "av"}))
    assert key(decode_spec=spec) != key(payload=b"other-payload", decode_spec=spec)


def test_media_ref_key_setting_order_does_not_matter():
    spec_a = DecodeSpec({"fps": 2, "backend": "opencv"})
    spec_b = DecodeSpec({"backend": "opencv", "fps": 2})

    assert (
        MediaRef(lambda: None, b"x", spec_a).key
        == MediaRef(lambda: None, b"x", spec_b).key
    )


def test_derive_media_key_frames_its_inputs():
    """A spec change can never be offset by a bytes change."""
    spec = DecodeSpec({"a": "b"})
    shifted = DecodeSpec({"a": "bc"})

    assert derive_media_key(b"x", spec) != derive_media_key(b"xy", shifted)


def test_media_ref_pickle_roundtrip_materializes():
    """The decode closure cannot cross a process boundary, so pickling
    materializes the media and the restored ref keeps the key and spec that
    identify it."""
    ref = _StubMediaIO().load_bytes_ref(b"payload")

    restored = pickle.loads(pickle.dumps(ref))

    assert restored.decode() == "payload"
    assert restored.is_decoded
    assert restored.key == ref.key
    assert restored.spec == ref.spec
    assert restored.data == b""


def test_media_ref_pickle_does_not_redecode():
    decoder = _CountingDecoder("value")
    ref = MediaRef(decoder, b"payload")
    ref.decode()

    restored = pickle.loads(pickle.dumps(ref))

    assert decoder.calls == 1
    assert restored.decode() == "value"
    assert decoder.calls == 1


def test_media_ref_pickle_after_release_before_decode_raises():
    ref = MediaRef(lambda: "value", b"payload")
    ref.release()

    with pytest.raises(RuntimeError, match="released"):
        pickle.dumps(ref)
