from enum import Enum, auto


class LauncherType(Enum):
    MPLauncher = auto()


def get_launcher_class(launcher_type: LauncherType):
    if launcher_type == LauncherType.MPLauncher:
        from vllm.implementations.launcher.mp_launcher import MPLauncher
        return MPLauncher
    else:
        raise NotImplementedError(
            f"Launcher type {launcher_type} not implemented")
