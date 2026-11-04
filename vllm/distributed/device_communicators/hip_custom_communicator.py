# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import atexit
import functools
import json
import os
import signal
import threading
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from enum import IntEnum
from pathlib import Path
from types import FrameType
from typing import Any, ClassVar, Literal, cast, final, get_args

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "HipCustomCommunicator",
    "AllReduceConfig",
    "AllReduceAddRmsNormConfig",
    "AllReduceAddAttnResRmsNormConfig",
    "get_hip_custom_communicator",
]

Op = Literal[
    "all_reduce",
    "all_reduce_add_rms_norm",
    "all_reduce_add_attn_res_rms_norm",
]

Quantization = Literal["exact", "int4"]

_QUANTIZATIONS: tuple[Quantization, ...] = ("exact", "int4")

AllReduceAlgorithm = Literal[
    "one_shot_pull",
    "one_shot_push",
    "two_shot_pull",
    "two_shot_push",
    "one_shot_push_staged",
]

AllReduceAddRmsNormAlgorithm = Literal[
    "one_shot_pull",
    "two_shot_pull_rows",
    "two_shot_pull_columns",
    "two_shot_push",
    "one_shot_pull_split",
    "one_shot_push_split",
    "one_shot_push_staged_split",
    "two_shot_pull_split",
    "two_shot_push_split",
]

AllReduceAddAttnResRmsNormAlgorithm = Literal[
    "one_shot_pull",
    "two_shot_pull_rows",
    "two_shot_pull_columns",
    "two_shot_push",
    "one_shot_pull_split",
    "one_shot_push_split",
    "one_shot_push_staged_split",
    "two_shot_pull_split",
    "two_shot_push_split",
]

Algorithm = Literal[
    "one_shot_pull",
    "one_shot_push",
    "two_shot_pull",
    "two_shot_pull_rows",
    "two_shot_pull_columns",
    "two_shot_push",
    "one_shot_push_staged",
    "one_shot_pull_split",
    "one_shot_push_split",
    "one_shot_push_staged_split",
    "two_shot_pull_split",
    "two_shot_push_split",
]

ABANDON_WAIT_S = 2.0

_EXIT_SIGNALS = (signal.SIGTERM, signal.SIGINT)

_exit_hooks_installed = False

SignalHandler = Callable[[int, FrameType | None], None] | int | None

_DTYPES: Mapping[str, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

_INPUT_DTYPES = (torch.float16, torch.bfloat16)

_WEIGHT_DTYPES = (torch.float32,)

_PACK_BYTES = 16

_TABLE = Path(__file__).with_name("hip_custom_communicator.json")

_TABLE_KEYS = ("inputs", "configs", "picks")

_QUALIFIED = "hip_custom_communicator."

_MODES: Mapping[Quantization, str] = {"exact": "exact", "int4": "quantized"}

_DTYPE_BYTES: Mapping[str, int] = {"bf16": 2, "f16": 2, "fp16": 2, "f32": 4}

_CONFIG_KEYS = (
    "kernel",
    "algorithm",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "waves_per_eu",
    "tile_m",
    "tile_n",
    "tile_k",
    "quantization",
    "pack_bytes",
)

_SPLIT_KERNELS: Mapping[str, str] = {
    "all_reduce_add_rms_norm": "add_rms_norm",
    "all_reduce_add_attn_res_rms_norm": "add_attn_res_rms_norm",
}
_LAUNCH_KEYS = (
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "waves_per_eu",
    "tile_m",
    "tile_n",
)
_SPLIT_KEYS: Mapping[str, tuple[str, ...]] = {
    "all_reduce_add_rms_norm": tuple(
        f"add_rms_norm_{k}" for k in (*_LAUNCH_KEYS, "pack_bytes")
    ),
    "all_reduce_add_attn_res_rms_norm": tuple(
        f"add_attn_res_rms_norm_{k}" for k in (*_LAUNCH_KEYS, "tile_k", "pack_bytes")
    ),
}

TORCH_COLLECTIVE_TIMEOUT_SECONDS = 600.0

Status = Literal["disabled", "open", "capturing", "broken", "abandoned", "closed"]

_open_communicators: "weakref.WeakSet[HipCustomCommunicator]" = weakref.WeakSet()


class Error(IntEnum):
    not_contiguous = 0
    dtype_not_built = 1
    world_not_built = 2
    row_not_packs = 3
    partials_too_small = 4
    grid_not_resident = 5
    staging_too_small = 6
    device_not_built = 7
    weight_not_built = 9
    no_such_kernel = 10
    not_built = 12
    launch_out_of_range = 13
    field_not_positive = 14
    field_not_this_templates = 15
    inner_stride_not_one = 17
    row_stride_not_packs = 18
    no_such_quantization = 19
    quantization_not_supported = 20
    block_too_large = 21
    lds_too_large = 22
    scratch_too_large = 23
    unfenced_too_small = 24
    pack_out_of_range = 25
    hip_failed = 26
    not_on_device = 27
    shape_mismatch = 28
    dtype_mismatch = 29
    not_aligned = 30
    inputs_overlap_outputs = 31
    field_out_of_range = 32
    view_past_region = 33
    tables_full = 34
    ranks_disagree = 35
    not_open = 36
    field_missing = 37
    no_such_group = 38
    exchange_failed = 39
    peer_not_accessible = 40
    peers_not_opened = 41

    no_problem_size = 100
    no_tuned_table = 101
    groups_disagree = 102
    disabled = 103
    broken = 104
    not_two_d = 105
    forced_incomplete = 106
    capture_not_open = 107
    close_while_capturing = 108
    wrong_device = 109
    tuned_table_unsound = 113
    capture_raised = 119


@dataclass(frozen=True)
class Build:
    algorithm: str
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    tile_k: int
    quantization: Quantization
    pack_bytes: int

    @classmethod
    def of(cls, config: "Config") -> "Build":
        return cls(
            algorithm=config.algorithm,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
            quantization=config.quantization,
            pack_bytes=config.pack_bytes,
        )


@dataclass(frozen=True)
class AllReduceConfig:
    kernel: ClassVar[str] = "all_reduce"
    algorithms: ClassVar[tuple[str, ...]] = get_args(AllReduceAlgorithm)
    tile_k: ClassVar[int] = 1
    algorithm: AllReduceAlgorithm
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    quantization: Quantization
    pack_bytes: int


@dataclass(frozen=True)
class AllReduceAddRmsNormConfig:
    kernel: ClassVar[str] = "all_reduce_add_rms_norm"
    algorithms: ClassVar[tuple[str, ...]] = get_args(AllReduceAddRmsNormAlgorithm)
    tile_k: ClassVar[int] = 1
    algorithm: AllReduceAddRmsNormAlgorithm
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    quantization: Quantization
    pack_bytes: int
    add_rms_norm_grid_x: int | None = None
    add_rms_norm_grid_y: int | None = None
    add_rms_norm_grid_z: int | None = None
    add_rms_norm_block_x: int | None = None
    add_rms_norm_block_y: int | None = None
    add_rms_norm_block_z: int | None = None
    add_rms_norm_waves_per_eu: int | None = None
    add_rms_norm_tile_m: int | None = None
    add_rms_norm_tile_n: int | None = None
    add_rms_norm_pack_bytes: int | None = None

    @property
    def split(self) -> bool:
        return self.algorithm.endswith("_split")


@dataclass(frozen=True)
class AllReduceAddAttnResRmsNormConfig:
    kernel: ClassVar[str] = "all_reduce_add_attn_res_rms_norm"
    algorithms: ClassVar[tuple[str, ...]] = get_args(
        AllReduceAddAttnResRmsNormAlgorithm
    )
    algorithm: AllReduceAddAttnResRmsNormAlgorithm
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    tile_k: int
    quantization: Quantization
    pack_bytes: int
    add_attn_res_rms_norm_grid_x: int | None = None
    add_attn_res_rms_norm_grid_y: int | None = None
    add_attn_res_rms_norm_grid_z: int | None = None
    add_attn_res_rms_norm_block_x: int | None = None
    add_attn_res_rms_norm_block_y: int | None = None
    add_attn_res_rms_norm_block_z: int | None = None
    add_attn_res_rms_norm_waves_per_eu: int | None = None
    add_attn_res_rms_norm_tile_m: int | None = None
    add_attn_res_rms_norm_tile_n: int | None = None
    add_attn_res_rms_norm_tile_k: int | None = None
    add_attn_res_rms_norm_pack_bytes: int | None = None

    @property
    def split(self) -> bool:
        return self.algorithm.endswith("_split")


Config = AllReduceConfig | AllReduceAddRmsNormConfig | AllReduceAddAttnResRmsNormConfig

_CONFIGS: Mapping[str, type[Config]] = {
    c.kernel: c
    for c in (
        AllReduceConfig,
        AllReduceAddRmsNormConfig,
        AllReduceAddAttnResRmsNormConfig,
    )
}


def _packs(config: Config) -> tuple[int, ...]:
    second = _SPLIT_KERNELS.get(config.kernel)
    pack = getattr(config, f"{second}_pack_bytes", None) if second else None
    return (config.pack_bytes,) if pack is None else (config.pack_bytes, pack)


@dataclass(frozen=True)
class TunedInput:
    op: Op
    residual: bool | None
    mode: str
    world: int
    hidden: int
    tokens: int
    dtype: str


@dataclass(frozen=True)
class TunedTable:
    picks: dict[TunedInput, Config]
    configs: tuple[Config, ...]
    answers: dict[tuple, tuple[TunedInput, Config]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def nearest(
        self,
        op: Op,
        residual: bool | None,
        quantization: Quantization,
        world: int,
        rows: int,
        cols: int,
    ) -> tuple[TunedInput, Config]:
        """The config tuned for the input nearest to this call."""
        question = (op, residual, quantization, world, rows, cols)
        if question in self.answers:
            return self.answers[question]
        tuned = [
            (i, c)
            for i, c in self.picks.items()
            if i.op == op
            and i.residual == residual
            and i.mode == _MODES[quantization]
            and i.world == world
        ]
        fitting = [
            (i, c)
            for i, c in tuned
            if all(cols * _DTYPE_BYTES[i.dtype] % pack == 0 for pack in _packs(c))
        ]
        tuned = fitting or tuned
        width = min((i.hidden for i, _ in tuned), key=lambda h: abs(h - cols))
        answer = min(
            ((i, c) for i, c in tuned if i.hidden == width),
            key=lambda ic: (abs(ic[0].tokens * width - rows * cols), ic[0].tokens),
        )
        self.answers[question] = answer
        return answer

    def kernel_config(
        self,
        kernel: str,
        algorithm: Algorithm,
        rows: int,
        cols: int,
        quantization: Quantization,
        world: int,
    ) -> Config | None:
        assert kernel in _CONFIGS, f"no op's kernel is {kernel}"
        builds = {Build.of(c) for c in self.configs if c.kernel == kernel}

        def takes(c: Config, moving: Quantization) -> bool:
            return (
                c.kernel == kernel
                and c.algorithm == algorithm
                and c.quantization == moving
                and replace(Build.of(c), quantization=quantization) in builds
            )

        for moving in (
            (quantization, "exact") if quantization == "int4" else (quantization,)
        ):
            found = []
            for i, pick in self.picks.items():
                if i.world == world and takes(pick, moving):
                    found.append((i, pick))
            if found:
                width = min((i.hidden for i, _ in found), key=lambda h: abs(h - cols))
                _, config = min(
                    ((i, c) for i, c in found if i.hidden == width),
                    key=lambda ic: (
                        abs(ic[0].tokens * ic[0].hidden - rows * cols),
                        ic[0].tokens,
                    ),
                )
            else:
                first = next((c for c in self.configs if takes(c, moving)), None)
                if first is None:
                    continue
                config = first
            if config is not None:
                return replace(config, quantization=quantization)
        return None


@dataclass
class Lifecycle:
    status: Status = "closed"
    error: Error | None = None
    detail: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    teardown: threading.Lock = field(default_factory=threading.Lock)

    def holds(self) -> bool:
        return self.status in ("open", "capturing", "broken")


@dataclass(frozen=True)
class CommsState:
    handle: int
    arch: str
    tuned_table: TunedTable


@dataclass(frozen=True, eq=False)
class Arguments:
    cpu_group: ProcessGroup
    device_group: ProcessGroup
    device: torch.device
    timeout_seconds: float
    max_rows: int
    max_cols: int
    quantization: Quantization
    tuned_path: Path

    @functools.cached_property
    def world(self) -> int:
        return self.device_group.size()

    @functools.cached_property
    def rank(self) -> int:
        return self.device_group.rank()


def parse_tuned_table(path: Path, arch: str) -> TunedTable | Error:
    """Read and check the tuned table for this arch."""
    try:
        tables = json.loads(path.read_text())
    except OSError as e:
        logger.error(
            "HipCustomCommunicator open: %s: %s",
            Error.no_tuned_table.name,
            f"{path} cannot be read ({e})",
        )
        return Error.no_tuned_table
    except ValueError as e:
        logger.error(
            "HipCustomCommunicator open: %s: %s",
            Error.tuned_table_unsound.name,
            f"{path} is not JSON ({e})",
        )
        return Error.tuned_table_unsound
    try:
        return _table_of(tables, path, arch)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.error(
            "HipCustomCommunicator open: %s: %s",
            Error.tuned_table_unsound.name,
            f"{path}: {type(e).__name__}: {e}",
        )
        return Error.tuned_table_unsound


def _table_of(tables: Any, path: Path, arch: str) -> TunedTable | Error:
    if arch not in tables:
        logger.error(
            "HipCustomCommunicator open: %s: %s",
            Error.no_tuned_table.name,
            f"{path} has no table for {arch}",
        )
        return Error.no_tuned_table
    raw = tables[arch]
    if list(raw) != list(_TABLE_KEYS):
        logger.error(
            "HipCustomCommunicator open: %s: %s",
            Error.tuned_table_unsound.name,
            f"the tuned table's keys are {list(raw)}, not {list(_TABLE_KEYS)}",
        )
        return Error.tuned_table_unsound
    configs: list[Config] = []
    for i, c in enumerate(raw["configs"]):
        where = f"the tuned table: configs[{i}]"
        kernel = c.get("kernel")
        if kernel not in _CONFIGS or c.get("op") != _QUALIFIED + kernel:
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: {kernel} is no kernel of {c.get('op')}",
            )
            return Error.tuned_table_unsound
        cls = _CONFIGS[c["kernel"]]
        own = {f.name for f in fields(cls)}
        split_keys = _SPLIT_KEYS.get(kernel, ())
        keys = ["op", *_CONFIG_KEYS, *split_keys]
        if list(c) != keys:
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where} has keys {list(c)}, not {keys}",
            )
            return Error.tuned_table_unsound
        if c["algorithm"] not in cls.algorithms:
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: {c['kernel']} has no algorithm {c['algorithm']}",
            )
            return Error.tuned_table_unsound
        for k in ("algorithm", "tile_k"):
            if k not in own and c[k] != getattr(cls, k):
                logger.error(
                    "HipCustomCommunicator open: %s: %s",
                    Error.tuned_table_unsound.name,
                    f"{where}: {c['kernel']}'s {k} is {getattr(cls, k)}, not {c[k]}",
                )
                return Error.tuned_table_unsound
        second = [c[k] for k in split_keys]
        split = str(c["algorithm"]).endswith("_split")
        if (split and (not second or None in second)) or (
            not split and second != [None] * len(second)
        ):
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: {split_keys} are a split's, every one, a split's alone",
            )
            return Error.tuned_table_unsound
        config = cls(**{k: c[k] for k in keys if k in own})
        configs.append(config)
    picks: dict[TunedInput, Config] = {}
    for i, k in enumerate(raw["picks"]):
        where = f"the tuned table: picks[{i}]"
        at = k.get("config")
        if not (
            list(k) == ["input", "config"]
            and isinstance(k["input"], int)
            and 0 <= k["input"] < len(raw["inputs"])
            and isinstance(at, int)
            and 0 <= at < len(configs)
        ):
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: {k} names no one input and its config",
            )
            return Error.tuned_table_unsound
        e = raw["inputs"][k["input"]]
        name = str(e.get("op", "")).removeprefix(_QUALIFIED)
        if not str(e.get("op", "")).startswith(_QUALIFIED) or name not in get_args(Op):
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: {e.get('op')} is no tuned op of the communicator's",
            )
            return Error.tuned_table_unsound
        if e.get("mode") not in _MODES.values() or not isinstance(e.get("world"), int):
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: no such mode, or no world: {e}",
            )
            return Error.tuned_table_unsound
        if ("residual" in e) != (name == "all_reduce_add_rms_norm") or (
            "residual" in e and not isinstance(e["residual"], bool)
        ):
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: a residual (a bool) is all_reduce_add_rms_norm's alone: {e}",
            )
            return Error.tuned_table_unsound
        if e.get("hidden", 0) < 1 or e.get("tokens", 0) < 1:
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: no width or no tokens",
            )
            return Error.tuned_table_unsound
        tuned = TunedInput(
            op=cast(Op, name),
            residual=e.get("residual"),
            mode=e["mode"],
            world=e["world"],
            hidden=e["hidden"],
            tokens=e["tokens"],
            dtype=e["dtype"],
        )
        if tuned in picks:
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: {tuned} is tuned twice",
            )
            return Error.tuned_table_unsound
        config = configs[at]
        if config.kernel != name:
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: its config is {config.kernel}'s, not {name}'s",
            )
            return Error.tuned_table_unsound
        if tuned.mode == "exact" and config.quantization != "exact":
            logger.error(
                "HipCustomCommunicator open: %s: %s",
                Error.tuned_table_unsound.name,
                f"{where}: an exact input moves INT4",
            )
            return Error.tuned_table_unsound
        picks[tuned] = config
    for name in get_args(Op):
        for residual in (False, True) if name == "all_reduce_add_rms_norm" else (None,):
            for world in sorted({i.world for i in picks}):
                for mode in _MODES.values():
                    if not any(
                        i.op == name
                        and i.residual == residual
                        and i.world == world
                        and i.mode == mode
                        for i in picks
                    ):
                        logger.error(
                            "HipCustomCommunicator open: %s: %s",
                            Error.tuned_table_unsound.name,
                            f"the tuned table: {name} (residual {residual}): no pick "
                            f"at world {world}, {mode}",
                        )
                        return Error.tuned_table_unsound
    return TunedTable(picks, tuple(configs))


def open_state(
    cpu_group: ProcessGroup,
    device_group: ProcessGroup,
    device: torch.device,
    max_rows: int,
    max_cols: int,
    timeout_seconds: float,
    tuned_path: Path,
) -> CommsState | Error:
    if (
        cpu_group.size() != device_group.size()
        or cpu_group.rank() != device_group.rank()
    ):
        logger.error(
            "HipCustomCommunicator open: %s: %s",
            Error.groups_disagree.name,
            "the CPU and device groups differ",
        )
        return Error.groups_disagree
    if torch.accelerator.current_device_index() != device.index:
        logger.error(
            "HipCustomCommunicator open: %s: %s",
            Error.wrong_device.name,
            f"the current device is {torch.accelerator.current_device_index()}, "
            f"not the communicator's {device.index}",
        )
        return Error.wrong_device
    error, handle, arch = torch.ops._rocm_C.hip_custom_communicator_open(
        group=cpu_group.group_name,
        device=device.index,
        max_rows=max_rows,
        max_cols=max_cols,
        timeout_seconds=timeout_seconds,
    )
    if error is not None:
        logger.error("HipCustomCommunicator open: C++'s open: %s", Error(error).name)
        return Error(error)
    tuned_table = parse_tuned_table(tuned_path, arch)
    if not isinstance(tuned_table, TunedTable):
        close_state(handle, alone=False)
        return tuned_table
    global _exit_hooks_installed
    if not _exit_hooks_installed:
        _exit_hooks_installed = True
        atexit.register(abandon_open)
        if threading.current_thread() is not threading.main_thread():
            logger.warning(
                "HipCustomCommunicator opened off the main thread: no SIGTERM/SIGINT "
                "teardown (process exit still runs it)"
            )
        else:

            def on_exit_signal(
                previous: SignalHandler, signum: int, frame: FrameType | None
            ) -> None:
                abandon_open()
                if callable(previous):
                    previous(signum, frame)
                elif previous != signal.SIG_IGN:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)

            for sig in _EXIT_SIGNALS:
                signal.signal(
                    sig, functools.partial(on_exit_signal, signal.getsignal(sig))
                )
    return CommsState(handle=handle, arch=arch, tuned_table=tuned_table)


def get_hip_custom_communicator() -> "HipCustomCommunicator | None":
    """The TP group's HipCustomCommunicator, or None."""
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )

    if get_tensor_model_parallel_world_size() <= 1:
        return None
    communicator = getattr(
        get_tp_group().device_communicator, "hip_custom_communicator", None
    )
    return None if communicator is None or communicator.disabled else communicator


def close_state(handle: int, alone: bool) -> None:
    error = torch.ops._rocm_C.hip_custom_communicator_close(state=handle, alone=alone)
    if error is not None:
        logger.error("HipCustomCommunicator close: %s", Error(error).name)


def abandon_open(wait_s: float = ABANDON_WAIT_S) -> None:
    for communicator in list(_open_communicators):
        communicator.abandon(wait_s)


class HipCustomCommunicator:
    """A TP all-reduce on HIP kernels that can fuse the op after it."""

    arguments: Arguments
    lifecycle: Lifecycle
    state: CommsState | None = None

    @final
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: int | str | torch.device,
        *,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        quantization: Quantization | None = None,
        tuned_path: Path = _TABLE,
    ) -> None:
        self.lifecycle = Lifecycle()
        self.state = None
        device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        if max_rows is None or max_cols is None:
            from vllm.config import get_current_vllm_config_or_none

            config = get_current_vllm_config_or_none()
            served = None
            if config is not None and config.model_config is not None:
                widest = config.model_config.get_hidden_size()
                spec = config.speculative_config
                draft = getattr(spec, "draft_model_config", None) if spec else None
                if draft is not None:
                    widest = max(widest, draft.get_hidden_size())
                served = config.scheduler_config.max_num_batched_tokens, widest
            if served is None:
                self.lifecycle.status = "disabled"
                self.lifecycle.error = Error.no_problem_size
                logger.info("%s disabled: no_problem_size", type(self).__name__)
                return
            served_rows, served_cols = served
            max_rows = served_rows if max_rows is None else max_rows
            max_cols = served_cols if max_cols is None else max_cols
        if timeout_seconds is None:
            try:
                timeout_seconds = device_group._get_backend(
                    device
                ).options._timeout.total_seconds()
            except (AttributeError, RuntimeError):
                timeout_seconds = TORCH_COLLECTIVE_TIMEOUT_SECONDS
        if quantization is None:
            import vllm.envs as envs

            asked = envs.VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION
            quantization = "int4" if asked == "INT4" else "exact"
        self.arguments = Arguments(
            cpu_group=cpu_group,
            device_group=device_group,
            device=device,
            timeout_seconds=timeout_seconds,
            max_rows=max_rows,
            max_cols=max_cols,
            quantization=quantization,
            tuned_path=tuned_path,
        )
        opened = open_state(
            cpu_group,
            device_group,
            device,
            max_rows,
            max_cols,
            timeout_seconds,
            tuned_path,
        )
        if not isinstance(opened, CommsState):
            error = opened
            if error in (
                Error.wrong_device,
                Error.no_such_group,
                Error.exchange_failed,
                Error.ranks_disagree,
                Error.peer_not_accessible,
                Error.peers_not_opened,
                Error.tuned_table_unsound,
            ):
                raise RuntimeError(f"{error.name}: opening the communicator (logged)")
            self.lifecycle.status, self.lifecycle.error = "disabled", error
            logger.info("%s disabled: %s", type(self).__name__, error.name)
            return
        self.state = opened
        self.lifecycle.status = "open"
        _open_communicators.add(self)
        logger.info(
            "HipCustomCommunicator ready: exchange sized for %d rows of %d, "
            "quantization=%s",
            self.arguments.max_rows,
            self.arguments.max_cols,
            self.arguments.quantization,
        )

    @final
    @property
    def disabled(self) -> bool:
        life = self.lifecycle
        if life.status == "broken":
            assert life.error is not None
            raise RuntimeError(
                f"broken: {life.error.name}: {life.detail}. Close it and build a "
                f"new one on every rank."
            )
        return life.status not in ("open", "capturing")

    @final
    def should_allreduce(self, input: torch.Tensor) -> bool:
        """Whether this communicator takes the all-reduce of `input`."""
        return (
            not self.disabled
            and self.state is not None
            and input.dtype in _INPUT_DTYPES
            and input.is_contiguous()
            and input.dim() == 2
            and input.numel() > 0
            and input.shape[1] * input.element_size() % _PACK_BYTES == 0
            and input.shape[0] <= self.arguments.max_rows
            and input.shape[1] <= self.arguments.max_cols
        )

    @final
    def all_reduce(
        self, input: torch.Tensor, *, config: AllReduceConfig | None = None
    ) -> torch.Tensor:
        """Sum `input` over the TP group."""
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: not {tuple(input.shape)}")

        output = torch.empty_like(input)
        capturing = torch.cuda.is_current_stream_capturing()
        if self.lifecycle.status == "capturing" and not capturing:
            return output

        if config is None:
            rows, cols = input.shape
            _, picked = self.state.tuned_table.nearest(
                "all_reduce",
                None,
                self.arguments.quantization,
                self.arguments.world,
                rows,
                cols,
            )
            assert isinstance(picked, AllReduceConfig), picked
            config = picked
        if self.arguments.quantization == "exact" and config.quantization != "exact":
            raise AssertionError(
                f"an exact communicator given an INT4 config: {config}"
            )

        error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
            state=self.state.handle,
            input=input,
            output=output,
            algorithm=config.algorithm,
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            grid_z=config.grid_z,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            quantization=config.quantization,
            pack_bytes=config.pack_bytes,
        )
        if error is not None:
            raise RuntimeError(f"{Error(error).name}: {config} on {input.shape}")
        return output

    @final
    def should_allreduce_add_rms_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> bool:
        """Whether this communicator takes the fused all-reduce and RMSNorm."""
        return (
            not self.disabled
            and self.state is not None
            and input.dtype in _INPUT_DTYPES
            and input.is_contiguous()
            and input.dim() == 2
            and input.numel() > 0
            and weight.dtype in (input.dtype, *_WEIGHT_DTYPES)
            and weight.dim() == 1
            and weight.shape[0] == input.shape[1]
            and weight.is_contiguous()
            and weight.device == input.device
            and weight.data_ptr()
            % (_PACK_BYTES // input.element_size() * weight.element_size())
            == 0
            and (
                residual is None
                or (
                    residual.shape == input.shape
                    and residual.dtype == input.dtype
                    and residual.is_contiguous()
                    and residual.device == input.device
                )
            )
            and input.shape[1] * input.element_size() % _PACK_BYTES == 0
            and input.shape[0] <= self.arguments.max_rows
            and input.shape[1] <= self.arguments.max_cols
        )

    @final
    def all_reduce_add_rms_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        residual: torch.Tensor | None = None,
        config: AllReduceAddRmsNormConfig | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """All-reduce `input`, then RMSNorm it, with an optional residual add."""
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: not {tuple(input.shape)}")

        add = residual is not None
        output = torch.empty_like(input)
        residual_output = torch.empty_like(input) if add else None
        if (
            self.lifecycle.status == "capturing"
            and not torch.cuda.is_current_stream_capturing()
        ):
            return (output, residual_output) if add else output

        if config is None:
            rows, cols = input.shape
            _, picked = self.state.tuned_table.nearest(
                "all_reduce_add_rms_norm",
                add,
                self.arguments.quantization,
                self.arguments.world,
                rows,
                cols,
            )
            assert isinstance(picked, AllReduceAddRmsNormConfig), picked
            config = picked
        if self.arguments.quantization == "exact" and config.quantization != "exact":
            raise AssertionError(
                f"an exact communicator given an INT4 config: {config}"
            )

        error = torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_rms_norm(
            state=self.state.handle,
            input=input,
            residual=residual,
            weight=weight,
            output=output,
            residual_output=residual_output,
            eps=eps,
            algorithm=config.algorithm,
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            grid_z=config.grid_z,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            quantization=config.quantization,
            pack_bytes=config.pack_bytes,
            add_rms_norm_grid_x=config.add_rms_norm_grid_x,
            add_rms_norm_grid_y=config.add_rms_norm_grid_y,
            add_rms_norm_grid_z=config.add_rms_norm_grid_z,
            add_rms_norm_block_x=config.add_rms_norm_block_x,
            add_rms_norm_block_y=config.add_rms_norm_block_y,
            add_rms_norm_block_z=config.add_rms_norm_block_z,
            add_rms_norm_waves_per_eu=config.add_rms_norm_waves_per_eu,
            add_rms_norm_tile_m=config.add_rms_norm_tile_m,
            add_rms_norm_tile_n=config.add_rms_norm_tile_n,
            add_rms_norm_pack_bytes=config.add_rms_norm_pack_bytes,
        )
        if error is not None:
            raise RuntimeError(f"{Error(error).name}: {config} on {input.shape}")
        return (output, residual_output) if add else output

    @final
    def should_allreduce_add_attn_res_rms_norm(
        self,
        input: torch.Tensor,
        prefix: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
        write_index: int,
    ) -> bool:
        """Whether this communicator takes the fused all-reduce and AttnRes."""
        e = input.element_size()
        weights = [norm_weight, qk_weight]
        if output_norm_weight is not None:
            weights.append(output_norm_weight)
        overlaps = (
            prefix is not None
            and write_index >= 0
            and prefix.device.type != "meta"
            and prefix.untyped_storage().data_ptr()
            == blocks.untyped_storage().data_ptr()
        )
        return (
            not self.disabled
            and self.state is not None
            and input.dtype in _INPUT_DTYPES
            and input.is_contiguous()
            and input.dim() == 2
            and input.numel() > 0
            and input.shape[1] * e % _PACK_BYTES == 0
            and input.shape[0] <= self.arguments.max_rows
            and input.shape[1] <= self.arguments.max_cols
            and blocks.dim() == 3
            and blocks.shape[0] == input.shape[0]
            and blocks.shape[2] == input.shape[1]
            and blocks.dtype == input.dtype
            and blocks.device == input.device
            and blocks.stride(2) == 1
            and blocks.stride(0) * e % _PACK_BYTES == 0
            and blocks.stride(1) * e % _PACK_BYTES == 0
            and blocks.data_ptr() % _PACK_BYTES == 0
            and 0 <= num_blocks <= blocks.shape[1]
            and (write_index < 0 or num_blocks <= write_index < blocks.shape[1])
            and all(
                w.dim() == 1
                and w.shape[0] == input.shape[1]
                and w.dtype == input.dtype
                and w.device == input.device
                and w.is_contiguous()
                for w in weights
            )
            and (
                prefix is None
                or (
                    prefix.shape == input.shape
                    and prefix.dtype == input.dtype
                    and prefix.device == input.device
                    and prefix.stride(1) == 1
                    and prefix.stride(0) * e % _PACK_BYTES == 0
                )
            )
            and not overlaps
        )

    @final
    def all_reduce_add_attn_res_rms_norm(
        self,
        input: torch.Tensor,
        prefix: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
        write_index: int,
        eps: float,
        output_eps: float,
        *,
        config: AllReduceAddAttnResRmsNormConfig | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce `input`, then AttnRes. Returns the new prefix and the output."""
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: not {tuple(input.shape)}")

        prefix_output, output = torch.empty_like(input), torch.empty_like(input)
        if (
            self.lifecycle.status == "capturing"
            and not torch.cuda.is_current_stream_capturing()
        ):
            return prefix_output, output

        if config is None:
            rows, cols = input.shape
            _, picked = self.state.tuned_table.nearest(
                "all_reduce_add_attn_res_rms_norm",
                None,
                self.arguments.quantization,
                self.arguments.world,
                rows,
                cols,
            )
            assert isinstance(picked, AllReduceAddAttnResRmsNormConfig), picked
            config = picked
        if self.arguments.quantization == "exact" and config.quantization != "exact":
            raise AssertionError(
                f"an exact communicator given an INT4 config: {config}"
            )

        block_output = blocks[:, write_index] if write_index >= 0 else None
        error = torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_attn_res_rms_norm(  # noqa: E501
            state=self.state.handle,
            input=input,
            prefix=prefix,
            blocks=blocks,
            norm_weight=norm_weight,
            qk_weight=qk_weight,
            output_norm_weight=output_norm_weight,
            prefix_output=prefix_output,
            output=output,
            block_output=block_output,
            num_blocks=num_blocks,
            eps=eps,
            output_eps=output_eps,
            algorithm=config.algorithm,
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            grid_z=config.grid_z,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
            quantization=config.quantization,
            pack_bytes=config.pack_bytes,
            add_attn_res_rms_norm_grid_x=config.add_attn_res_rms_norm_grid_x,
            add_attn_res_rms_norm_grid_y=config.add_attn_res_rms_norm_grid_y,
            add_attn_res_rms_norm_grid_z=config.add_attn_res_rms_norm_grid_z,
            add_attn_res_rms_norm_block_x=config.add_attn_res_rms_norm_block_x,
            add_attn_res_rms_norm_block_y=config.add_attn_res_rms_norm_block_y,
            add_attn_res_rms_norm_block_z=config.add_attn_res_rms_norm_block_z,
            add_attn_res_rms_norm_waves_per_eu=config.add_attn_res_rms_norm_waves_per_eu,
            add_attn_res_rms_norm_tile_m=config.add_attn_res_rms_norm_tile_m,
            add_attn_res_rms_norm_tile_n=config.add_attn_res_rms_norm_tile_n,
            add_attn_res_rms_norm_tile_k=config.add_attn_res_rms_norm_tile_k,
            add_attn_res_rms_norm_pack_bytes=config.add_attn_res_rms_norm_pack_bytes,
        )
        if error is not None:
            raise RuntimeError(f"{Error(error).name}: {config} on {input.shape}")
        return prefix_output, output

    @final
    @contextmanager
    def capture(self) -> Iterator[None]:
        """Wrap a CUDA graph capture. Registers its inputs with the peers on exit."""
        with self.lifecycle.lock:
            if self.lifecycle.status != "open":
                raise RuntimeError(
                    f"capture_not_open: capture() while {self.lifecycle.status}"
                )
            self.lifecycle.status = "capturing"
        raised: BaseException | None = None
        try:
            yield
        except BaseException as e:
            raised = e
        assert self.state is not None
        error = torch.ops._rocm_C.hip_custom_communicator_register_graph(
            state=self.state.handle
        )
        refused = (
            isinstance(raised, RuntimeError)
            and str(raised).split(":")[0] in Error.__members__
            and not str(raised).startswith(Error.hip_failed.name)
        )
        why = (
            f"{Error(error).name}: the captured inputs"
            if error is not None
            else None
            if raised is None or refused
            else f"{type(raised).__name__}: {raised}"
        )
        life = self.lifecycle
        with life.lock:
            if why is not None:
                if life.status not in ("broken", "abandoned", "closed"):
                    life.status, life.error = "broken", Error.capture_raised
                    life.detail = why
                    logger.error("%s broken: %s", type(self).__name__, why)
            elif life.status == "capturing":
                life.status = "open"
        if raised is not None:
            raise raised
        if error is not None:
            raise RuntimeError(f"{Error(error).name}: the captured inputs")

    @final
    def tuned(
        self, op: Op, rows: int, cols: int, residual: bool | None = None
    ) -> Config:
        assert self.state is not None, "an open communicator has its state"
        _, config = self.state.tuned_table.nearest(
            op, residual, self.arguments.quantization, self.arguments.world, rows, cols
        )
        return config

    @final
    def kernel_config(
        self,
        kernel: str,
        algorithm: Algorithm,
        rows: int,
        cols: int,
        quantization: Quantization = "exact",
    ) -> Config | None:
        assert self.state is not None, "an open communicator has its state"
        return self.state.tuned_table.kernel_config(
            kernel, algorithm, rows, cols, quantization, self.arguments.world
        )

    @final
    def close(self) -> None:
        """Close the communicator. Every rank must call it."""
        if self.lifecycle.status == "closed":
            return
        if self.lifecycle.status == "capturing":
            raise RuntimeError(
                "close_while_capturing: the graph being recorded would replay "
                "against released buffers; leave the capture first"
            )
        with self.lifecycle.lock:
            self.lifecycle.status = "closed"
        with self.lifecycle.teardown:
            if self.state is None:
                return
            try:
                torch.accelerator.synchronize(self.arguments.device)
            except RuntimeError as e:
                logger.error(
                    "HipCustomCommunicator close: the device did not finish: %s", e
                )
            close_state(self.state.handle, alone=False)
            self.state = None

    @final
    def abandon(self, wait_s: float = ABANDON_WAIT_S) -> None:
        with self.lifecycle.lock:
            if not self.lifecycle.holds():
                return
            self.lifecycle.status = "abandoned"
        if self.state is None:
            return
        if not self.lifecycle.teardown.acquire(blocking=False):
            return
        try:
            failed: list[BaseException] = []

            def synchronize() -> None:
                try:
                    torch.accelerator.synchronize(self.arguments.device)
                except BaseException as e:  # noqa: BLE001 -- reported as not finished
                    failed.append(e)

            waiter = threading.Thread(target=synchronize, daemon=True)
            waiter.start()
            waiter.join(wait_s)
            if not waiter.is_alive() and not failed:
                close_state(self.state.handle, alone=True)
                self.state = None
        finally:
            self.lifecycle.teardown.release()

    @final
    def __enter__(self) -> "HipCustomCommunicator":
        return self

    @final
    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if self.lifecycle.holds():
            warnings.warn(
                f"{type(self).__name__} was never closed; its peer handles and buffers "
                f"were left to garbage collection. Use `with make_communicator(...) as "
                f"comm:` "
                f"or call `close()`.",
                ResourceWarning,
                stacklevel=2,
            )
