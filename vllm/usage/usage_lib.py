import os
import torch
import json
import platform
import pkg_resources
import requests
import datetime
import psutil
import cpuinfo
from threading import Thread
from pathlib import Path
from typing import Optional
from enum import Enum

_xdg_config_home = os.getenv('XDG_CONFIG_HOME',
                             os.path.expanduser('~/.config'))
_vllm_internal_path = 'vllm/usage_stats.json'

_USAGE_STATS_FILE = os.path.join(
    _xdg_config_home,
    _vllm_internal_path)  #File path to store usage data locally
_USAGE_STATS_ENABLED = None
_USAGE_STATS_SERVER = os.environ.get('VLLM_USAGE_STATS_SERVER',
                                     'https://stats.vllm.ai')
_USAGE_STATS_URL = "https://vector-dev-server-uzyrqjjayq-uc.a.run.app"  #Placeholder for sending usage data to vector.dev http server


def is_usage_stats_enabled():
    """Determine whether or not we can send usage stats to the server.
    The logic is as follows:
    - By default, it should be enabled.
    - Two environment variables can disable it:
        - DO_NOT_TRACK=1
        - VLLM_NO_USAGE_STATS=1
    - A file in the home directory can disable it if it exists:
        - $HOME/.config/vllm/do_not_track
    """
    global _USAGE_STATS_ENABLED
    if _USAGE_STATS_ENABLED is None:
        do_not_track = os.environ.get('DO_NOT_TRACK', '0') == '1'
        no_usage_stats = os.environ.get('VLLM_NO_USAGE_STATS', '0') == '1'
        do_not_track_file = os.path.exists(
            os.path.expanduser('~/.config/vllm/do_not_track'))

        _USAGE_STATS_ENABLED = not (do_not_track or no_usage_stats
                                    or do_not_track_file)
    return _USAGE_STATS_ENABLED


def _get_current_timestamp_ns() -> int:
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)


def _detect_cloud_provider() -> str:
    # Try detecting through vendor file
    vendor_files = [
        '/sys/class/dmi/id/product_version', '/sys/class/dmi/id/bios_vendor',
        '/sys/class/dmi/id/product_name',
        '/sys/class/dmi/id/chassis_asset_tag', '/sys/class/dmi/id/sys_vendor'
    ]
    # Mapping of identifiable strings to cloud providers
    cloud_identifiers = {
        'amazon': "AWS",
        'microsoft corporation': "AZURE",
        'google': "GCP",
        'oraclecloud': "OCI",
    }

    for vendor_file in vendor_files:
        path = Path(vendor_file)
        if path.is_file():
            file_content = path.read_text().lower()
            for identifier, provider in cloud_identifiers.items():
                if identifier in file_content:
                    return provider
    return "UNKNOWN"


class UsageContext(Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"


class UsageMessage:

    def __init__(self) -> None:
        self.gpu_list: Optional[dict] = None
        self.provider: Optional[str] = None
        self.architecture: Optional[str] = None
        self.platform: Optional[str] = None
        self.model: Optional[str] = None
        self.vllm_version: Optional[str] = None
        self.context: Optional[str] = None
        self.log_time: Optional[int] = None
        #Logical CPU count
        self.num_cpu: Optional[int] = None
        self.cpu_type: Optional[str] = None
        self.total_memory: Optional[int] = None

    def report_usage(self, model: str, context: UsageContext) -> None:
        t = Thread(target=usage_message._report_usage, args=(model, context))
        t.start()

    def _report_usage(self, model: str, context: UsageContext) -> None:
        self.context = context.value
        self.gpu_list = []
        for i in range(torch.cuda.device_count()):
            device_property = torch.cuda.get_device_properties(i)
            name = device_property.name
            memory = device_property.total_memory
            self.gpu_list.append((name, memory))
        self.provider = _detect_cloud_provider()
        self.architecture = platform.machine()
        self.platform = platform.platform()
        self.vllm_version = pkg_resources.get_distribution("vllm").version
        self.model = model
        self.log_time = _get_current_timestamp_ns()
        self.num_cpu = os.cpu_count()
        self.cpu_type = cpuinfo.get_cpu_info()['brand_raw']
        self.total_memory = psutil.virtual_memory().total
        self._write_to_file()
        headers = {'Content-type': 'application/x-ndjson'}
        payload = json.dumps(vars(self))
        try:
            requests.post(_USAGE_STATS_URL, data=payload, headers=headers)
        except requests.exceptions.RequestException:
            print("Usage Log Request Failed")

    def _write_to_file(self):
        os.makedirs(os.path.dirname(_USAGE_STATS_FILE), exist_ok=True)
        with open(_USAGE_STATS_FILE, "w+") as outfile:
            json.dump(vars(self), outfile)


usage_message = UsageMessage()
