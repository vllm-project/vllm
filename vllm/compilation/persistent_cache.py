# SPDX-License-Identifier: Apache-2.0
"""Opt-in remote persistence for vLLM's torch compilation cache."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch

from vllm import __version__ as vllm_version
from vllm.logger import init_logger

logger = init_logger(__name__)

_BUCKET = "s3://eric-alcaide-dev/vllm_cache"
_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
_REGION = "eu-north1"
_SCHEMA = 1


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_cache_manifest(
    vllm_config: Any,
    *,
    env_factors: dict[str, object],
    config_hash: str,
    compiler_hash: str,
    code_hash: str,
) -> dict[str, Any]:
    """Build the exhaustive, human-auditable remote cache identity."""
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    scheduler = vllm_config.scheduler_config
    compilation = vllm_config.compilation_config
    hf_config = model.hf_config.to_dict()
    hf_config_hash = hashlib.sha256(
        json.dumps(hf_config, sort_keys=True, default=str).encode()
    ).hexdigest()
    capability = (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    )
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None

    return {
        "schema": _SCHEMA,
        "model": {
            "identity": model.model,
            "revision": model.revision,
            "code_revision": model.code_revision,
            "resolved_revision": getattr(model.hf_config, "_commit_hash", None),
            "hf_config_hash": hf_config_hash,
            "dtype": str(model.dtype),
            "attention_dtype": str(model.override_attention_dtype),
            "quantization": model.quantization,
            "max_model_len": model.max_model_len,
            "model_hash": model.compute_hash(),
        },
        "topology": {
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
            "data_parallel_size": parallel.data_parallel_size,
            "enable_expert_parallel": parallel.enable_expert_parallel,
            "world_size": parallel.world_size,
            "parallel_hash": parallel.compute_hash(),
        },
        "limits": {
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "max_num_seqs": scheduler.max_num_seqs,
            "scheduler_hash": scheduler.compute_hash(),
        },
        "kv_cache": {
            "dtype": str(cache.cache_dtype),
            "config_hash": cache.compute_hash(),
        },
        "backends": {
            "attention": str(vllm_config.attention_config.backend),
            "attention_hash": vllm_config.attention_config.compute_hash(),
            "moe": str(vllm_config.kernel_config.moe_backend),
            "kernel_hash": vllm_config.kernel_config.compute_hash(),
        },
        "toolchain": {
            "gpu_name": device_name,
            "gpu_capability": capability,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "cudnn": torch.backends.cudnn.version(),
            "triton": _package_version("triton"),
            "vllm": vllm_version,
            "flashinfer": _package_version("flashinfer-python"),
            "python_abi": platform.python_implementation()
            + "-"
            + platform.python_version(),
        },
        "compilation": {
            "config_hash": compilation.compute_hash(),
            "compiler_hash": compiler_hash,
            "source_code_hash": code_hash,
        },
        # The full hashes are intentionally retained in addition to the fields
        # above. They default-include graph-affecting config and environment
        # fields, so new factors isolate entries without changing this module.
        "vllm_config_hash": config_hash,
        "compile_environment": env_factors,
    }


def manifest_key(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class PersistentCompileCache:
    """Securely restore and publish one rank's compilation artifacts."""

    def __init__(self, key: str, rank: int, dp_rank: int, prefix: str) -> None:
        safe_prefix = hashlib.sha256(prefix.encode()).hexdigest()[:16]
        self.uri = (
            f"{_BUCKET}/v{_SCHEMA}/{key}/rank_{rank}_{dp_rank}/{safe_prefix}.tar.gz"
        )

    @staticmethod
    def _aws(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                "aws",
                "--endpoint-url",
                _ENDPOINT,
                "--region",
                _REGION,
                "s3",
                *args,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def restore(self, destination: str) -> bool:
        Path(destination).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vllm-compile-cache-") as tmp:
            archive = os.path.join(tmp, "cache.tar.gz")
            result = self._aws("cp", self.uri, archive, "--only-show-errors")
            if result.returncode != 0:
                logger.info(
                    "Persistent compile cache miss for key %s",
                    self.uri.split("/")[-3],
                )
                return False
            try:
                with tarfile.open(archive, "r:gz") as tar:
                    root = Path(destination).resolve()
                    for member in tar.getmembers():
                        target = (root / member.name).resolve()
                        if root not in target.parents and target != root:
                            raise ValueError("cache archive contains an unsafe path")
                        if member.issym() or member.islnk():
                            raise ValueError("cache archive contains a link")
                        if not (member.isfile() or member.isdir()):
                            raise ValueError("cache archive contains a special file")
                    tar.extractall(destination)
            except (OSError, tarfile.TarError, ValueError):
                logger.warning(
                    "Ignoring invalid persistent compile cache artifact", exc_info=True
                )
                return False
        logger.info("Persistent compile cache hit; restored compiled artifacts")
        return True

    def publish(self, source: str) -> bool:
        with tempfile.TemporaryDirectory(prefix="vllm-compile-cache-") as tmp:
            archive = os.path.join(tmp, "cache.tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                for path in sorted(Path(source).rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        tar.add(path, arcname=path.relative_to(source), recursive=False)
            result = self._aws("cp", archive, self.uri, "--only-show-errors")
        if result.returncode != 0:
            logger.warning("Failed to upload persistent compile cache artifact")
            return False
        logger.info("Uploaded persistent compile cache artifact")
        return True
