# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for ``vllm.transformers_utils.s3_utils``.

These tests cover the lazy-import behaviour added in the boto3 cold-start
optimization:

  * The module imports cleanly without boto3 actually being touched.
  * ``glob()`` and ``list_files()`` both lazy-import boto3 when the caller
    passes ``s3=None`` (defensive guard at every public entry point).
  * Missing-boto3 raises a clear ``ImportError`` (not a confusing
    ``AttributeError`` from a placeholder client construction).
  * The legacy module-level ``boto3`` attribute remains importable for
    external consumers that did ``from ... import boto3`` (resolved on
    demand via PEP 562 ``__getattr__``).
"""
import builtins
import importlib
import sys
from unittest.mock import MagicMock

import pytest

from vllm.transformers_utils import s3_utils


def test_module_imports_without_touching_boto3():
    """Importing s3_utils must not pull boto3 into ``sys.modules``.

    This is the core regression guard for the perf optimization: if anyone
    re-introduces a top-level ``import boto3`` the cold-start cost returns.
    """
    # Force a clean re-import without boto3 already cached.
    sys.modules.pop("vllm.transformers_utils.s3_utils", None)
    sys.modules.pop("boto3", None)
    importlib.import_module("vllm.transformers_utils.s3_utils")
    assert "boto3" not in sys.modules, (
        "s3_utils import pulled boto3 into sys.modules; the lazy-import "
        "optimization has regressed."
    )


def test_filter_helpers_pure_python():
    """Filter helpers must work without boto3 at all."""
    paths = ["a.bin", "b.safetensors", "c.txt"]
    assert s3_utils._filter_allow(paths, ["*.bin", "*.safetensors"]) == [
        "a.bin",
        "b.safetensors",
    ]
    assert s3_utils._filter_ignore(paths, ["*.txt"]) == ["a.bin", "b.safetensors"]


def test_list_files_with_explicit_client_does_not_import_boto3():
    """Passing a real S3 client must NOT trigger the lazy import path."""
    sys.modules.pop("boto3", None)

    s3 = MagicMock()
    s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "models/foo.bin"},
            {"Key": "models/bar.safetensors"},
            {"Key": "models/subdir/"},
        ]
    }

    bucket, prefix, paths = s3_utils.list_files(
        s3, "s3://my-bucket/models/", allow_pattern=["*.bin", "*.safetensors"]
    )

    assert bucket == "my-bucket"
    assert prefix == "models/"
    assert paths == ["models/foo.bin", "models/bar.safetensors"]
    # Critical: no lazy-import was triggered when caller supplied s3.
    assert "boto3" not in sys.modules


def test_glob_with_explicit_client():
    """``glob()`` happy-path with a caller-supplied client."""
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {
        "Contents": [{"Key": "models/foo.bin"}, {"Key": "models/bar.bin"}]
    }
    out = s3_utils.glob(s3=s3, path="s3://my-bucket/models", allow_pattern=["*.bin"])
    assert out == [
        "s3://my-bucket/models/foo.bin",
        "s3://my-bucket/models/bar.bin",
    ]


def test_glob_missing_boto3_raises_importerror(monkeypatch):
    """When ``s3=None`` and boto3 is unavailable, raise a clear ImportError.

    Mirrors the previous PlaceholderModule-on-attribute-access behaviour:
    callers that try to construct a default S3 client without boto3 installed
    must get an ImportError pointing at the install command — never a silent
    AttributeError on a stub.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3" or name.startswith("boto3."):
            raise ImportError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("boto3", None)

    with pytest.raises(ImportError, match="boto3 is required"):
        s3_utils.glob(s3=None, path="s3://my-bucket/models")


def test_list_files_missing_boto3_raises_importerror(monkeypatch):
    """The same lazy-guard applies to ``list_files()`` for safety.

    Without this guard, calling ``list_files(None, ...)`` would die with a
    confusing ``AttributeError: 'NoneType' object has no attribute
    'list_objects_v2'`` instead of a clean install hint.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3" or name.startswith("boto3."):
            raise ImportError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("boto3", None)

    with pytest.raises(ImportError, match="boto3 is required"):
        s3_utils.list_files(None, "s3://my-bucket/models/")


def test_module_level_boto3_attr_resolves_via_pep562():
    """``s3_utils.boto3`` must remain accessible for legacy external consumers.

    Some downstream code did ``from vllm.transformers_utils.s3_utils import
    boto3``; we preserve that surface via a module-level ``__getattr__``
    (PEP 562) so the perf win does not become a silent breaking-change.
    """
    # The attribute resolves to either the real boto3 module or a
    # PlaceholderModule, depending on whether boto3 is installed in this
    # environment. Either way, attribute access must not raise.
    boto3_attr = s3_utils.boto3
    assert boto3_attr is not None


def test_unknown_module_attr_still_raises_attributeerror():
    """The PEP 562 ``__getattr__`` must not over-eagerly swallow unrelated
    typos. Anything other than ``boto3`` should raise AttributeError, matching
    standard module-attribute semantics.
    """
    with pytest.raises(AttributeError, match="no attribute"):
        _ = s3_utils.does_not_exist  # type: ignore[attr-defined]
