# SPDX-License-Identifier: Apache-2.0
"""Opt-in remote persistence for vLLM's torch compilation cache."""

import atexit
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
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
_SCHEMA = 2
_REGISTERED_CACHES: dict[str, tuple["PersistentCompileCache", dict[str, Path]]] = {}


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
    """Securely restore and publish one rank's compilation artifacts.

    Besides vLLM's torch compile directory, a bundle contains FlashInfer's JIT
    products (including ``fused_moe_trtllm_sm100``) and vLLM's FlashInfer
    autotuner results (including MoE and dense ``fp4_gemm`` profiles). CUDA
    graph executables themselves are process-local CUDA objects and cannot be
    serialized; their compilation inputs and exact capture configuration are
    persisted/keyed so capture is repeatable without recompilation/autotuning.
    """

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

    @staticmethod
    def cache_roots(torch_compile_dir: str) -> dict[str, Path]:
        flashinfer_jit = Path.home() / ".cache" / "flashinfer"
        flashinfer_autotune = (
            Path(os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"]).expanduser()
            if os.getenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR")
            else Path(os.environ.get("VLLM_CACHE_ROOT", Path.home() / ".cache/vllm"))
            / "flashinfer_autotune_cache"
        )
        return {
            "torch_compile": Path(torch_compile_dir),
            "flashinfer_jit": flashinfer_jit,
            "flashinfer_autotune": flashinfer_autotune,
        }

    def restore(self, roots: str | dict[str, Path]) -> bool:
        if isinstance(roots, str):
            roots = {"torch_compile": Path(roots)}
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
                    staging = (Path(tmp) / "extracted").resolve()
                    staging.mkdir()
                    for member in tar.getmembers():
                        target = (staging / member.name).resolve()
                        if staging not in target.parents and target != staging:
                            raise ValueError("cache archive contains an unsafe path")
                        if member.issym() or member.islnk():
                            raise ValueError("cache archive contains a link")
                        if not (member.isfile() or member.isdir()):
                            raise ValueError("cache archive contains a special file")
                    tar.extractall(staging)
                for name, destination in roots.items():
                    source = staging / "roots" / name
                    if source.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(source, destination, dirs_exist_ok=True)
            except (OSError, tarfile.TarError, ValueError):
                logger.warning(
                    "Ignoring invalid persistent compile cache artifact", exc_info=True
                )
                return False
        logger.info("Persistent compile cache hit; restored compiled artifacts")
        return True

    def publish(self, roots: str | dict[str, Path]) -> bool:
        if isinstance(roots, str):
            roots = {"torch_compile": Path(roots)}
        with tempfile.TemporaryDirectory(prefix="vllm-compile-cache-") as tmp:
            archive = os.path.join(tmp, "cache.tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                for name, source in sorted(roots.items()):
                    if not source.is_dir():
                        continue
                    for path in sorted(source.rglob("*")):
                        if path.is_file() and not path.is_symlink():
                            relative = path.relative_to(source)
                            tar.add(
                                path,
                                arcname=Path("roots") / name / relative,
                                recursive=False,
                            )
            result = self._aws("cp", archive, self.uri, "--only-show-errors")
        if result.returncode != 0:
            logger.warning("Failed to upload persistent compile cache artifact")
            return False
        logger.info("Uploaded persistent compile cache artifact")
        return True

    def publish_at_exit(self, roots: dict[str, Path]) -> None:
        """Register for post-warmup and normal-process-exit publication."""
        if self.uri not in _REGISTERED_CACHES:
            _REGISTERED_CACHES[self.uri] = (self, roots)
            atexit.register(self.publish, roots)


def publish_registered_caches() -> None:
    """Publish after model warmup, FlashInfer autotuning, and graph capture."""
    for cache, roots in _REGISTERED_CACHES.values():
        cache.publish(roots)
