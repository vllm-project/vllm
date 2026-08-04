# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for fs_l2_adapter key serialization helpers.

These helpers round-trip ObjectKey <-> filename. ``object_group_id`` is
embedded as a fixed field right after ``kv_rank``; ``cache_salt`` is
appended as a trailing field when non-empty. Unsalted keys use the
4-field shape and salted keys use the 5-field shape.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _filename_to_object_key,
    _object_key_to_filename,
)


class TestFilenameRoundtrip:
    """``_object_key_to_filename`` and ``_filename_to_object_key`` are
    exact inverses for both the 4-field (unsalted) and 5-field (salted)
    shapes."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "llama",
            "meta-llama/Llama-3",  # has '/', must survive PATH_SLASH_REPLACEMENT
        ],
    )
    @pytest.mark.parametrize("cache_salt", ["", "alice", "user-abc_123.xyz:42"])
    @pytest.mark.parametrize("object_group_id", [0, 1, 255])
    def test_roundtrip(self, model_name: str, cache_salt: str, object_group_id: int):
        key = ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name=model_name,
            kv_rank=42,
            object_group_id=object_group_id,
            cache_salt=cache_salt,
        )
        fn = _object_key_to_filename(key)
        assert fn.endswith(".data")
        # Salted filenames gain a trailing "@<salt>" before ".data".
        if cache_salt:
            assert fn.endswith("@" + cache_salt + ".data")
        parsed = _filename_to_object_key(fn)
        assert parsed == key

    def test_object_group_id_distinguishes_filenames(self):
        """Keys differing only in object_group_id must not collide."""
        fn0 = _object_key_to_filename(
            ObjectKey(
                chunk_hash=b"\xde\xad\xbe\xef",
                model_name="llama",
                kv_rank=42,
                object_group_id=0,
            )
        )
        fn1 = _object_key_to_filename(
            ObjectKey(
                chunk_hash=b"\xde\xad\xbe\xef",
                model_name="llama",
                kv_rank=42,
                object_group_id=1,
            )
        )
        assert fn0 != fn1

    def test_unsalted_format(self):
        """Unsalted keys use the 4-field shape."""
        fn = "llama@0x0000002a@0@deadbeef.data"
        parsed = _filename_to_object_key(fn)
        assert parsed == ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=42,
            object_group_id=0,
            cache_salt="",
        )

    def test_salted_format(self):
        """Salted keys append ``@<cache_salt>`` before the extension."""
        fn = "llama@0x0000002a@2@deadbeef@alice.data"
        parsed = _filename_to_object_key(fn)
        assert parsed == ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=42,
            object_group_id=2,
            cache_salt="alice",
        )

    def test_non_data_file_returns_none(self):
        assert _filename_to_object_key("not-a-data-file.txt") is None

    def test_too_few_fields_returns_none(self):
        assert _filename_to_object_key("just-one-field.data") is None

    def test_old_three_field_format_returns_none(self):
        """The pre-object_group_id 3-field shape is no longer accepted."""
        assert _filename_to_object_key("llama@0x0000002a@deadbeef.data") is None

    def test_too_many_fields_returns_none(self):
        assert _filename_to_object_key("a@b@c@d@e@f.data") is None

    def test_salt_with_forbidden_char_returns_none(self):
        # A filename that parses into 5 fields but whose trailing "salt"
        # slot contains a char ObjectKey.__post_init__ rejects (NUL here
        # is impossible in filenames, so use the length cap instead).
        too_long_salt = "x" * 129
        fn = f"llama@0x0000002a@0@deadbeef@{too_long_salt}.data"
        assert _filename_to_object_key(fn) is None


class TestIpcKeyToObjectKeys:
    """ipc_key_to_object_keys reads cache_salt from the ipc_key itself —
    there is no separate parameter, so callers cannot accidentally drop
    the salt."""

    def test_forwards_cache_salt_single_worker(self):
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        k = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_ids=[1, 2, 3],
            cache_salt="alice",
        )
        out = ipc_key_to_object_keys(k, [b"h1", b"h2"], [0])[0]
        assert len(out) == 2
        assert all(o.cache_salt == "alice" for o in out)

    def test_forwards_cache_salt_scheduler_path(self):
        """worker_id=None explodes one chunk into one ObjectKey per worker."""
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        k = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=4,
            worker_id=None,
            token_ids=[1, 2, 3],
            cache_salt="alice",
        )
        out = ipc_key_to_object_keys(k, [b"h1"], [0])[0]
        assert len(out) == 4
        assert all(o.cache_salt == "alice" for o in out)

    def test_empty_salt_passes_through(self):
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        k = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_ids=[1],
        )
        out = ipc_key_to_object_keys(k, [b"h1"], [0])[0]
        assert all(o.cache_salt == "" for o in out)

    def test_object_group_id_zero(self):
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        k = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_ids=[1, 2],
        )
        out = ipc_key_to_object_keys(k, [b"h1", b"h2"], [0])[0]
        assert all(o.object_group_id == 0 for o in out)

    def test_object_group_id_propagates_to_all_keys(self):
        """A non-zero object_group_id reaches every produced ObjectKey,
        including the worker-expansion (scheduler) path."""
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        k = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=4,
            worker_id=None,
            token_ids=[1, 2, 3],
        )
        out = ipc_key_to_object_keys(k, [b"h1"], [3])[0]
        assert len(out) == 4
        assert all(o.object_group_id == 3 for o in out)

    def test_multiple_object_groups(self):
        """Each requested object group gets its own positional key list."""
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        k = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=2,
            worker_id=None,
            token_ids=[1, 2, 3],
            cache_salt="alice",
        )
        out = ipc_key_to_object_keys(k, [b"h1", b"h2"], [0, 3])
        assert len(out) == 2
        # 2 chunks * 2 workers = 4 keys per group.
        assert all(len(group_keys) == 4 for group_keys in out)
        assert all(o.object_group_id == 0 for o in out[0])
        assert all(o.object_group_id == 3 for o in out[1])
        # The groups differ only in object_group_id.
        for first, second in zip(out[0], out[1], strict=True):
            assert first.chunk_hash == second.chunk_hash
            assert first.kv_rank == second.kv_rank
            assert first.cache_salt == second.cache_salt


class TestObjectKeyValidation:
    """``ObjectKey.__post_init__`` rejects invalid ``object_group_id``."""

    def test_negative_object_group_id_rejected(self):
        with pytest.raises(ValueError, match="object_group_id"):
            ObjectKey(
                chunk_hash=b"\xde\xad\xbe\xef",
                model_name="llama",
                kv_rank=0,
                object_group_id=-1,
            )


class TestIPCCacheServerKeyCacheSalt:
    """cache_salt on IPCCacheServerKey: validation + wire compat."""

    def test_reject_at_in_salt(self):
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        with pytest.raises(ValueError, match="cache_salt"):
            IPCCacheServerKey.from_token_ids(
                model_name="m",
                world_size=1,
                worker_id=0,
                token_ids=[1],
                cache_salt="a@b",
            )

    def test_reject_slash_in_salt(self):
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        with pytest.raises(ValueError, match="cache_salt"):
            IPCCacheServerKey.from_token_ids(
                model_name="m",
                world_size=1,
                worker_id=0,
                token_ids=[1],
                cache_salt="tenant/alice",
            )

    def test_no_worker_id_version_preserves_salt(self):
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        k = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=4,
            worker_id=2,
            token_ids=[1],
            cache_salt="alice",
        )
        k2 = k.no_worker_id_version()
        assert k2.worker_id is None
        assert k2.cache_salt == "alice"

    def test_wire_compat_old_payload_decodes(self):
        """An old 7-field msgspec payload must decode cleanly on new code
        with cache_salt defaulting to ""."""
        # Third Party
        import msgspec

        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        old_payload = {
            "model_name": "m",
            "world_size": 1,
            "worker_id": 0,
            "token_ids": (1, 2),
            "start": 0,
            "end": 2,
            "request_id": "r1",
        }
        wire = msgspec.msgpack.encode(old_payload)
        decoded = msgspec.msgpack.decode(wire, type=IPCCacheServerKey)
        assert decoded.cache_salt == ""

    def test_wire_compat_new_payload_roundtrip(self):
        # Third Party
        import msgspec

        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        k = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_ids=[1, 2],
            cache_salt="alice",
        )
        wire = msgspec.msgpack.encode(k)
        decoded = msgspec.msgpack.decode(wire, type=IPCCacheServerKey)
        assert decoded == k
        assert decoded.cache_salt == "alice"
