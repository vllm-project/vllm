import requests
import os
import torch
import json
import platform
import sys
import pkg_resources
import requests
from cloud_detect import provider
from typing import Optional
from enum import Enum
from pathlib import Path
_USAGE_STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usage_stats.json')
_USAGE_STATS_ENABLED = None
_USAGE_STATS_SEVER = os.environ.get('VLLM_USAGE_STATS_SERVER', 'https://stats.vllm.ai')
_USAGE_STATS_URL = "http://127.0.0.1:1234"
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
        do_not_track_file = os.path.exists(os.path.expanduser('~/.config/vllm/do_not_track'))

        _USAGE_STATS_ENABLED = not (do_not_track or no_usage_stats or do_not_track_file)
    return _USAGE_STATS_ENABLED

class UsageContext(Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM = "LLM"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"


class UsageMessage:
    def __init__(self) -> None:
        self.gpu_name : Optional[str] = None
        self.provider : Optional[str] = None
        self.architecture : Optional[str] = None
        self.platform : Optional[str] = None
        self.model : Optional[str] = None
        self.vllm_version : Optional[str] = None
        self.context : Optional[str] = None
    def report_usage(self, model: str, context: UsageContext) -> None:
        self.context = context.value
        self.gpu_name = torch.cuda.get_device_name()
        self.provider = provider()
        self.architecture = platform.machine()
        self.platform = platform.platform()
        self.vllm_version = pkg_resources.get_distribution("vllm").version
        self.model = model
    def write_to_file(self):
        with open(_USAGE_STATS_FILE, "w") as outfile: 
            json.dump(vars(self), outfile)
    def send_to_server(self):
        headers = {'Content-type': 'application/json'}
        payload = json.dumps(vars(self))
        response = requests.post(_USAGE_STATS_URL, data=payload, headers=headers)
usage_message = UsageMessage()