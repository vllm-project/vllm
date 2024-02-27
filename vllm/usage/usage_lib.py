import os
import torch
import json
import platform
import pkg_resources
import requests
import datetime
import psutil
from threading import Thread
from pathlib import Path
from typing import Optional
from enum import Enum

_USAGE_STATS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'usage_stats.json')  #File path to store usage data locally
_USAGE_STATS_ENABLED = None
_USAGE_STATS_SEVER = os.environ.get('VLLM_USAGE_STATS_SERVER',
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
    for vendor_file in vendor_files:
        path = Path(vendor_file)
        if path.is_file():
            if 'amazon' in path.read_text().lower():
                return "AWS"
            elif 'Microsoft Corporation' in path.read_text():
                return "AZURE"
            elif 'Google' in path.read_text():
                return "GCP"
            elif 'OracleCloud' in path.read_text():
                return "OCI"
    return "UNKNOWN"


class UsageContext(Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"


class UsageMessage:

    def __init__(self) -> None:
        self.gpu: Optional[dict] = None
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
        self.gpu = dict()
        for i in range(torch.cuda.device_count()):
            k = torch.cuda.get_device_properties(i).name
            if k in self.gpu:
                self.gpu[k] += 1
            else:
                self.gpu[k] = 1
        self.provider = _detect_cloud_provider()
        self.architecture = platform.machine()
        self.platform = platform.platform()
        self.vllm_version = pkg_resources.get_distribution("vllm").version
        self.model = model
        self.log_time = _get_current_timestamp_ns()
        self.num_cpu = os.cpu_count()
        self.cpu_type = platform.processor()
        self.total_memory = psutil.virtual_memory().total
        self._write_to_file()
        headers = {'Content-type': 'application/json'}
        payload = json.dumps(vars(self))
        try:
            requests.post(_USAGE_STATS_URL, data=payload, headers=headers)
        except requests.exceptions.RequestException:
            print("Usage Log Request Failed")

    def _write_to_file(self):
        with open(_USAGE_STATS_FILE, "w") as outfile:
            json.dump(vars(self), outfile)


usage_message = UsageMessage()
