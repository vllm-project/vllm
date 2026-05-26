# proton options

import os

_launch_metadata_allow_sync = None


def launch_metadata_allow_sync():
    global _launch_metadata_allow_sync
    if _launch_metadata_allow_sync is None:
        _launch_metadata_allow_sync = not (os.getenv("PROTON_LAUNCH_METADATA_NOSYNC") == "1")
    return _launch_metadata_allow_sync


def set_launch_metadata_allow_sync(allow_sync: bool):
    global _launch_metadata_allow_sync
    _launch_metadata_allow_sync = allow_sync
