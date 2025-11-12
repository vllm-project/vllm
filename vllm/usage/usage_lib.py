# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import json
import logging
import os
import platform
import time
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Any
from uuid import uuid4

import cpuinfo
import psutil
import requests
import torch

import vllm.envs as envs
from vllm.connections import global_http_connection
from vllm.logger import init_logger
from vllm.utils.platform_utils import cuda_get_device_properties
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

_config_home = envs.VLLM_CONFIG_ROOT
_USAGE_STATS_JSON_PATH = os.path.join(_config_home, "usage_stats.json")
_USAGE_STATS_DO_NOT_TRACK_PATH = os.path.join(_config_home, "do_not_track")
_USAGE_STATS_ENABLED = None
_USAGE_STATS_SERVER = envs.VLLM_USAGE_STATS_SERVER

_GLOBAL_RUNTIME_DATA = dict[str, str | int | bool]()

_USAGE_ENV_VARS_TO_COLLECT = [
    "VLLM_USE_MODELSCOPE",
    "VLLM_USE_TRITON_FLASH_ATTN",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_PP_LAYER_PARTITION",
    "VLLM_USE_TRITON_AWQ",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
]


def set_runtime_usage_data(key: str, value: str | int | bool) -> None:
    """Set global usage data that will be sent with every usage heartbeat."""
    _GLOBAL_RUNTIME_DATA[key] = value


def is_usage_stats_enabled():
    """Determine whether or not we can send usage stats to the server.
    The logic is as follows:
    - By default, it should be enabled.
    - Three environment variables can disable it:
        - VLLM_DO_NOT_TRACK=1
        - DO_NOT_TRACK=1
        - VLLM_NO_USAGE_STATS=1
    - A file in the home directory can disable it if it exists:
        - $HOME/.config/vllm/do_not_track
    """
    global _USAGE_STATS_ENABLED
    if _USAGE_STATS_ENABLED is None:
        do_not_track = envs.VLLM_DO_NOT_TRACK
        no_usage_stats = envs.VLLM_NO_USAGE_STATS
        do_not_track_file = os.path.exists(_USAGE_STATS_DO_NOT_TRACK_PATH)

        _USAGE_STATS_ENABLED = not (do_not_track or no_usage_stats or do_not_track_file)
    return _USAGE_STATS_ENABLED


def _get_current_timestamp_ns() -> int:
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)


def _detect_cloud_provider() -> str:
    # Try detecting through vendor file
    vendor_files = [
        "/sys/class/dmi/id/product_version",
        "/sys/class/dmi/id/bios_vendor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/chassis_asset_tag",
        "/sys/class/dmi/id/sys_vendor",
    ]
    # Mapping of identifiable strings to cloud providers
    cloud_identifiers = {
        "amazon": "AWS",
        "microsoft corporation": "AZURE",
        "google": "GCP",
        "oraclecloud": "OCI",
    }

    for vendor_file in vendor_files:
        path = Path(vendor_file)
        if path.is_file():
            file_content = path.read_text().lower()
            for identifier, provider in cloud_identifiers.items():
                if identifier in file_content:
                    return provider

    # Try detecting through environment variables
    env_to_cloud_provider = {
        "RUNPOD_DC_ID": "RUNPOD",
    }
    for env_var, provider in env_to_cloud_provider.items():
        if os.environ.get(env_var):
            return provider

    return "UNKNOWN"


class UsageContext(str, Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"


class UsageMessage:
    """Collect platform information and send it to the usage stats server."""

    def __init__(self) -> None:
        # NOTE: vLLM's server _only_ support flat KV pair.
        # Do not use nested fields.

        self.uuid = str(uuid4())

        # Environment Information
        self.provider: str | None = None
        self.num_cpu: int | None = None
        self.cpu_type: str | None = None
        self.cpu_family_model_stepping: str | None = None
        self.total_memory: int | None = None
        self.architecture: str | None = None
        self.platform: str | None = None
        self.cuda_runtime: str | None = None
        self.gpu_count: int | None = None
        self.gpu_type: str | None = None
        self.gpu_memory_per_device: int | None = None
        self.env_var_json: str | None = None

        # vLLM Information
        self.model_architecture: str | None = None
        self.vllm_version: str | None = None
        self.context: str | None = None

        # Metadata
        self.log_time: int | None = None
        self.source: str | None = None

    def report_usage(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any] | None = None,
    ) -> None:
        t = Thread(
            target=self._report_usage_worker,
            args=(model_architecture, usage_context, extra_kvs or {}),
            daemon=True,
        )
        t.start()

    def _report_usage_worker(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any],
    ) -> None:
        self._report_usage_once(model_architecture, usage_context, extra_kvs)
        self._report_continuous_usage()

    def _report_tpu_inference_usage(self) -> bool:
        try:
            from tpu_inference import tpu_info, utils

            self.gpu_count = tpu_info.get_num_chips()
            self.gpu_type = tpu_info.get_tpu_type()
            self.gpu_memory_per_device = utils.get_device_hbm_limit()
            self.cuda_runtime = "tpu_inference"
            return True
        except Exception:
            return False

    def _report_torch_xla_usage(self) -> bool:
        try:
            import torch_xla

            self.gpu_count = torch_xla.runtime.world_size()
            self.gpu_type = torch_xla.tpu.get_tpu_type()
            self.gpu_memory_per_device = torch_xla.core.xla_model.get_memory_info()[
                "bytes_limit"
            ]
            self.cuda_runtime = "torch_xla"
            return True
        except Exception:
            return False

    def _report_usage_once(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any],
    ) -> None:
        # Platform information
        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike():
            self.gpu_count = cuda_device_count_stateless()
            self.gpu_type, self.gpu_memory_per_device = cuda_get_device_properties(
                0, ("name", "total_memory")
            )
        if current_platform.is_cuda():
            self.cuda_runtime = torch.version.cuda
        if current_platform.is_tpu():  # noqa: SIM102
            if (not self._report_tpu_inference_usage()) and (
                not self._report_torch_xla_usage()
            ):
                logger.exception("Failed to collect TPU information")
        self.provider = _detect_cloud_provider()
        self.architecture = platform.machine()
        self.platform = platform.platform()
        self.total_memory = psutil.virtual_memory().total

        info = cpuinfo.get_cpu_info()
        self.num_cpu = info.get("count", None)
        self.cpu_type = info.get("brand_raw", "")
        self.cpu_family_model_stepping = ",".join(
            [
                str(info.get("family", "")),
                str(info.get("model", "")),
                str(info.get("stepping", "")),
            ]
        )

        # vLLM information
        self.context = usage_context.value
        self.vllm_version = VLLM_VERSION
        self.model_architecture = model_architecture

        # Environment variables
        self.env_var_json = json.dumps(
            {env_var: getattr(envs, env_var) for env_var in _USAGE_ENV_VARS_TO_COLLECT}
        )

        # Metadata
        self.log_time = _get_current_timestamp_ns()
        self.source = envs.VLLM_USAGE_SOURCE

        data = vars(self)
        if extra_kvs:
            data.update(extra_kvs)

        self._write_to_file(data)
        self._send_to_server(data)

    def _report_continuous_usage(self):
        """Report usage every 10 minutes.

        This helps us to collect more data points for uptime of vLLM usages.
        This function can also help send over performance metrics over time.
        """
        while True:
            time.sleep(600)
            data = {
                "uuid": self.uuid,
                "log_time": _get_current_timestamp_ns(),
            }
            data.update(_GLOBAL_RUNTIME_DATA)

            self._write_to_file(data)
            self._send_to_server(data)

    def _send_to_server(self, data: dict[str, Any]) -> None:
        try:
            global_http_client = global_http_connection.get_sync_client()
            global_http_client.post(_USAGE_STATS_SERVER, json=data)
        except requests.exceptions.RequestException:
            # silently ignore unless we are using debug log
            logging.debug("Failed to send usage data to server")

    def _write_to_file(self, data: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(_USAGE_STATS_JSON_PATH), exist_ok=True)
        Path(_USAGE_STATS_JSON_PATH).touch(exist_ok=True)
        with open(_USAGE_STATS_JSON_PATH, "a") as f:
            json.dump(data, f)
            f.write("\n")


usage_message = UsageMessage()
