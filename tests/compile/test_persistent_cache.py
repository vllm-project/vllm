# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.compilation.persistent_cache import (
    PersistentCompileCache,
    build_cache_manifest,
    manifest_key,
)
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import EngineArgs


class _Hashable(SimpleNamespace):
    def compute_hash(self):
        return getattr(self, "hash", "hash")


def _config(revision="main"):
    return SimpleNamespace(
        model_config=_Hashable(
            model="org/model",
            revision=revision,
            code_revision=None,
            dtype=torch.float16,
            quantization="fp8",
            max_model_len=4096,
            hash=f"model-{revision}",
        ),
        parallel_config=_Hashable(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=2,
            enable_expert_parallel=True,
            world_size=2,
        ),
        cache_config=_Hashable(cache_dtype="fp8"),
        scheduler_config=_Hashable(
            max_num_batched_tokens=8192,
            max_num_seqs=64,
        ),
        compilation_config=_Hashable(),
        attention_config=_Hashable(),
        kernel_config=_Hashable(),
    )


def _manifest(revision="main"):
    with patch.object(torch.cuda, "is_available", return_value=False):
        return build_cache_manifest(
            _config(revision),
            env_factors={"VLLM_TEST": "1"},
            config_hash=f"config-{revision}",
            compiler_hash="compiler",
            code_hash="code",
        )


def test_compile_cache_is_disabled_by_default():
    assert CompilationConfig().persistent_cache_enabled is False
    assert EngineArgs().compile_cache is False


def test_compile_cache_cli_flags_are_mutually_exclusive():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    assert parser.parse_args([]).compile_cache is False
    assert parser.parse_args(["--compile-cache"]).compile_cache is True
    assert parser.parse_args(["--no-compile-cache"]).compile_cache is False


def test_manifest_is_deterministic_and_isolates_mismatch():
    assert manifest_key(_manifest()) == manifest_key(_manifest())
    assert manifest_key(_manifest("main")) != manifest_key(_manifest("v2"))


def test_cache_miss(tmp_path):
    cache = PersistentCompileCache("a" * 64, 0, 0, "model")
    result = SimpleNamespace(returncode=1)
    with patch.object(cache, "_aws", return_value=result) as aws:
        assert cache.restore(str(tmp_path)) is False
    assert aws.call_args.args[0] == "cp"


def test_upload_then_cache_hit_reuses_artifact(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"compiled")
    uploaded = io.BytesIO()
    cache = PersistentCompileCache("b" * 64, 0, 0, "model")

    def fake_aws(command, source_arg, destination_arg, *_args):
        assert command == "cp"
        if source_arg.startswith("s3://"):
            Path(destination_arg).write_bytes(uploaded.getvalue())
        else:
            uploaded.seek(0)
            uploaded.truncate()
            uploaded.write(Path(source_arg).read_bytes())
        return SimpleNamespace(returncode=0)

    with patch.object(cache, "_aws", side_effect=fake_aws):
        assert cache.publish(str(source)) is True
        destination = tmp_path / "destination"
        assert cache.restore(str(destination)) is True

    assert (destination / "artifact.bin").read_bytes() == b"compiled"


def test_restore_rejects_path_traversal(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    cache = PersistentCompileCache("c" * 64, 0, 0, "model")

    def fake_aws(_command, _source, destination, *_args):
        Path(destination).write_bytes(payload.getvalue())
        return SimpleNamespace(returncode=0)

    with patch.object(cache, "_aws", side_effect=fake_aws):
        assert cache.restore(str(tmp_path / "destination")) is False
    assert not (tmp_path / "escape").exists()
