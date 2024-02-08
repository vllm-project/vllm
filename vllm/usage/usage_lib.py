import requests
import os
import torch
import json
import platform
import sys
from cloud_detect import provider
from typing import Optional
_USAGE_STATS_FILE = 'usage_stats.json'
_USAGE_STATS_ENABLED = None
_USAGE_STATS_SEVER = os.environ.get('VLLM_USAGE_STATS_SERVER', 'https://stats.vllm.ai')

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


class UsageMessage:
    def __init__(self) -> None:
        self.gpu_name : Optional[str] = None
        self.provider : Optional[str] = None
        self.architecture : Optional[str] = None
        self.platform : Optional[str] = None
        self.model : Optional[str] = None
        self.entry_point : Optional[str] = None
    def report_usage(self) -> None:
        self.entry_point = sys.argv
        self.gpu_name = torch.cuda.get_device_name()
        self.provider = provider()
        self.architecture = platform.machine()
        self.platform = platform.platform()
    def update_model(self, model: str) -> None:
        self.model = model
    def write_to_file(self):
        with open(_USAGE_STATS_FILE, "w") as outfile: 
            json.dump(vars(self), outfile)
usage_message = UsageMessage()