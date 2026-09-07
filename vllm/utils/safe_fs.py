# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path
import stat
import tempfile


def get_user_root_dir() -> str:
    """Return the per-user vLLM runtime directory (under the home folder)."""
    return os.path.join(Path.home(), "vllm")


def _current_uid() -> int:
    """Return the effective uid of the current process, or -1 if unknown."""
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else -1


def _verify_dir_components(directory: str) -> None:
    """Reject symlinked components on the directory path below the temp root.

    Files written into a private directory can contain sensitive data, so a
    different local user must not be able to plant a symlink on the path to
    redirect them. Components at or above the system temp root are trusted
    (they are OS-managed, and the temp root itself may be a symlink, e.g.
    ``/tmp`` on macOS). For a directory outside the temp root
    (operator-configured), only the directory itself is checked.
    """
    temp_root = os.path.abspath(tempfile.gettempdir())
    path = os.path.abspath(directory)
    try:
        below_temp_root = os.path.commonpath([path, temp_root]) == temp_root
    except ValueError:  # different drives on Windows
        below_temp_root = False
    if below_temp_root:
        components = []
        cur = path
        while cur != temp_root:
            components.append(cur)
            parent = os.path.dirname(cur)
            if parent == cur:  # reached filesystem root
                break
            cur = parent
    else:
        components = [path]
    for comp in components:
        try:
            info = os.lstat(comp)
        except (FileNotFoundError, NotADirectoryError):
            # Not created yet (or reached through a non-directory, which
            # os.makedirs below will reject); components not verified here
            # are created privately.
            continue
        if stat.S_ISLNK(info.st_mode):
            raise OSError(
                f"Refusing symlinked component {comp} on directory path"
            )


def prepare_private_dir(directory: str) -> None:
    """Create a directory locked down to the current user.

    The path is first checked for symlinked components, then created with
    mode 0700 and re-checked (real directory, owned by the current user)
    before anything is written into it.
    """
    _verify_dir_components(directory)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    info = os.lstat(directory)
    if stat.S_ISLNK(info.st_mode):
        raise OSError(f"Refusing symlinked directory {directory}")
    if not stat.S_ISDIR(info.st_mode):
        raise OSError(f"Path {directory} is not a directory")
    uid = _current_uid()
    if uid != -1 and info.st_uid != uid:
        raise OSError(
            f"Directory {directory} is not owned by the current user"
        )


def safe_open_file(path: str, mode: str):
    """Open a file for writing with no-follow semantics.

    A symlinked file planted by another user is refused instead of followed
    (O_APPEND mirrors the "a" mode). O_NOFOLLOW is only defined on Windows
    since Python 3.12.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, mode)
