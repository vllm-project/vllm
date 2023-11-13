import requests
import os

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




