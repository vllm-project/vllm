import enum
import functools
import json
import os
import random
import re
import socket
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cached_property
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Literal, Optional, get_args, get_type_hints

import torch
import torch.distributed as dist
import zmq


def get_device():
    from vllm.platforms import current_platform

    return torch.device(current_platform.device_type)


# config base


@dataclass(frozen=True)
class _BaseConfig(ABC):
    def __post_init__(self) -> None:
        self._verify_types()

    def _verify_types(self) -> None:
        hints = get_type_hints(type(self))
        cls_name = type(self).__name__
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            expected = self._unwrap_type(hints[f.name])
            if not isinstance(value, expected):
                raise TypeError(
                    f"{cls_name}.{f.name}: expected {expected.__name__}, "
                    f"got {type(value).__name__}"
                )

    @classmethod
    @abstractmethod
    def _env_prefix(cls) -> str: ...

    @classmethod
    def _env_name(cls, field_name: str) -> str:
        return f"{cls._env_prefix()}{field_name.upper()}"

    @classmethod
    def from_env(cls) -> "_BaseConfig":
        return cls(
            **{
                f.name: cls._parse_env_field(cls._env_name(f.name), f.default)
                for f in fields(cls)
            }
        )

    def with_defaults(self, **kwargs) -> "_BaseConfig":
        cls = type(self)
        actual = {
            key: value
            for key, value in kwargs.items()
            if os.getenv(cls._env_name(key)) is None
        }
        return replace(self, **actual) if actual else self

    @staticmethod
    def _unwrap_type(hint) -> type:
        args = get_args(hint)
        if args:
            return next(a for a in args if a is not type(None))
        return hint

    @classmethod
    def _parse_env_field(cls, env_name: str, default):
        return cls._parse_env_value(os.getenv(env_name), default)

    @staticmethod
    def _parse_env_value(raw, default):
        if raw is None or not raw.strip():
            return default
        if isinstance(default, bool):
            return raw.lower() in ("true", "1")
        if isinstance(default, int):
            return int(raw)
        return raw

    @classmethod
    def from_kv_pairs(cls, pairs: list[str] | None) -> "_BaseConfig":
        return cls(**cls._kv_pairs_to_dict(pairs))

    @classmethod
    def _kv_pairs_to_dict(cls, pairs: list[str] | None) -> dict:
        if not pairs:
            return {}

        missing = object()
        defaults = {f.name: f.default for f in fields(cls)}
        result: dict = {}

        for pair in pairs:
            key, sep, value = pair.partition("=")
            if not sep:
                raise ValueError(f"Invalid config pair (missing '='): {pair!r}")
            default = defaults.get(key, missing)
            if default is missing:
                raise ValueError(
                    f"Unknown config key {key!r}. Valid keys: {sorted(defaults)}"
                )
            try:
                result[key] = cls._parse_env_value(value, default)
            except (ValueError, TypeError) as exc:
                field_type = type(default).__name__
                raise TypeError(f"{key}: expected {field_type}, got {value!r}") from exc

        return result


_DEFAULT_EXP_NAME_PREFIX = "dump_"


@dataclass(frozen=True)
class DumperConfig(_BaseConfig):
    enable: bool = False
    filter: str | None = None
    dir: str = "/tmp/dumper"
    enable_output_file: bool = True
    enable_output_console: bool = True
    enable_value: bool = True
    enable_grad: bool = False
    enable_model_value: bool = False
    enable_model_grad: bool = False
    exp_name: str | None = None
    cleanup_previous: bool = False
    collective_timeout: int = 60
    server_port: str = "-1"
    non_intrusive_mode: str = "core"
    source_patcher_config: str | None = None
    grafter_enable: bool = False
    grafter_role: str = ""  # required if enabled: "baseline" or "target"
    grafter_b2t_filter: str | None = None  # names flowing baseline -> target
    grafter_t2b_filter: str | None = None  # names flowing target -> baseline
    grafter_master_address: str = ""  # required if enabled
    grafter_master_port: int = -1  # required if enabled (positive port)
    grafter_baseline_world_size: int = -1  # required if enabled
    grafter_target_world_size: int = -1  # required if enabled
    grafter_backend: str = "nccl"
    grafter_group_name: str = "graft"
    grafter_timeout: int = 300
    # Fully-qualified Python path "pkg.subpkg.module.fn_name"
    # None -> use the default identity-by-rank fallback in _Grafter._default_transform.
    grafter_transform_path: str | None = None
    # When True, append parallel-rank tags (pp_rank/tp_rank/...) to dump filenames so
    # tensors from different ranks do not collide when dumped into a shared directory.
    include_parallel_rank_in_filename: bool = False

    @classmethod
    def _env_prefix(cls) -> str:
        # NOTE: should not be `SGLANG_DUMPER_`, otherwise it is weird when dumping
        # Megatron in Miles
        return "DUMPER_"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.grafter_enable:
            assert self.grafter_role in ("baseline", "target"), (
                f"grafter_role must be 'baseline' or 'target' when "
                f"grafter_enable=True, "
                f"got {self.grafter_role!r}"
            )
            assert self.grafter_master_address, (
                "grafter_master_address must be set when grafter_enable=True"
            )
            assert self.grafter_master_port > 0, (
                f"grafter_master_port must be a positive port when "
                f"grafter_enable=True, "
                f"got {self.grafter_master_port}"
            )
            assert self.grafter_baseline_world_size > 0, (
                f"grafter_baseline_world_size must be > 0 when grafter_enable=True, "
                f"got {self.grafter_baseline_world_size}"
            )
            assert self.grafter_target_world_size > 0, (
                f"grafter_target_world_size must be > 0 when grafter_enable=True, "
                f"got {self.grafter_target_world_size}"
            )
            assert (
                self.grafter_b2t_filter is not None
                or self.grafter_t2b_filter is not None
            ), (
                "grafter_enable=True but neither grafter_b2t_filter nor "
                "grafter_t2b_filter is set; nothing would ever be grafted"
            )

    @property
    def server_port_parsed(self) -> int | Literal["reuse"] | None:
        raw = self.server_port
        if raw == "reuse":
            return "reuse"
        port = int(raw)
        if port <= 0:
            return None
        return port


# dumper core


@dataclass
class _DumperState:
    dump_index: int = 0
    step: int = 0
    global_ctx: dict = field(default_factory=dict)
    captured_output_data: dict | None = None
    cleanup_previous_handled: bool = False


class _Dumper:
    """Utility to dump tensors, which can be useful when comparison checking models.

    Example usage:
    dumper.dump("layer_start__hidden_states", hidden_states, layer_id=self.layer_id)
    dumper.step()

    Import from non-SGLang system:
    ```
    import sys

    sys.path.append("/YOUR_PATH/vllm/vllm/utils/debug_utils")
    from dumper import dumper
    ```

    Then run the program:
    `DUMPER_ENABLE=1 python ...`

    Auto-cleanup old dumps before first write:
    `DUMPER_CLEANUP_PREVIOUS=1 python ...`

    Alternatively, disable at startup and configure via HTTP:
    1. `python ...`
    2. vLLM mode:  `curl -X POST http://localhost:30000/dumper/configure -d
    '{"enable": true}'`
       standalone:   `curl -X POST http://localhost:40000/dumper/configure -d
       '{"enable": true}'`
    3. `curl -X POST http://localhost:30000/dumper/configure -d '{"enable": true,
    "filter": "layer_id=[0-3]"}'`
    4. `curl -X POST http://localhost:30000/dumper/reset`

    Related: `vllm.utils.debug_utils.dump_comparator` for dump comparison
    """

    def __init__(self, *, config: DumperConfig):
        self._config = config
        self._state = _DumperState()
        self._non_intrusives: list[_NonIntrusiveDumper] = []
        self._grafter = _Grafter(config=config)

    # ------------------------------- public :: core ---------------------------------

    @property
    def may_enable(self) -> bool:
        return self._config.enable or self._config.server_port_parsed is not None

    def step(self):
        """This should be called on all ranks at the end of each iteration."""
        self._http_manager  # noqa: B018

        if not self._config.enable:
            return

        # Users may want to `dump` only on some ranks, thus determine name here
        self._ensure_exp_name()

        self._state.step += 1
        _log(f"step={self._state.step}")

    def dump(
        self,
        name: str,
        value,
        save: bool = True,
        dims: str | None = None,
        dims_grad: str | None = None,
        grafter_extras: dict | None = None,
        **kwargs,
    ) -> None:
        value_meta: dict = {}
        grad_meta: dict = {}
        if dims is not None:
            value_meta["dims"] = dims
            grad_meta["dims"] = dims
        if dims_grad is not None:
            value_meta["dims_grad"] = dims_grad
            grad_meta["dims"] = dims_grad

        self._dump_inner(
            name=name,
            value=value,
            extra_kwargs=kwargs,
            save=save,
            enable_value=self._config.enable_value,
            enable_curr_grad=False,
            enable_future_grad=self._config.enable_grad,
            value_tag="Dumper.Value",
            grad_tag="Dumper.Grad",
            value_meta_only_fields=value_meta,
            grad_meta_only_fields=grad_meta,
            grafter_extras=grafter_extras,
        )

    def dump_model(
        self,
        model: "torch.nn.Module",
        name_prefix: str = "param",
        save: bool = True,
        get_grad: Callable | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        for param_name, param in model.named_parameters():
            for plugin in _plugins:
                param_name = (
                    plugin.transform_model_param_name(model, param_name) or param_name
                )
            self._dump_inner(
                name=f"{name_prefix}__{param_name}",
                value=param,
                extra_kwargs=kwargs,
                save=save,
                enable_value=self._config.enable_model_value,
                enable_curr_grad=self._config.enable_model_grad,
                enable_future_grad=False,
                value_tag="Dumper.ParamValue",
                grad_tag="Dumper.ParamGrad",
                get_grad=get_grad,
                step=step,
            )

    def dump_dict(self, name_prefix, data, save: bool = True, **kwargs):
        data = _obj_to_dict(data)
        for name, value in data.items():
            self.dump(f"{name_prefix}_{name}", value, save=save, **kwargs)

    def set_ctx(self, **kwargs):
        """Example:
        dumper.configure_default(filter='layer_id=[0-3]')
        dumper.set_ctx(layer_id=self.layer_id)
        ...
        dumper.set_ctx(layer_id=None)

        """
        self._state.global_ctx = {
            k: v for k, v in (self._state.global_ctx | kwargs).items() if v is not None
        }

    def ctx(
        self,
        _extractor: Callable[..., dict] | None = None,
        **static_ctx: Any,
    ) -> Callable:
        """Decorator that sets context before calling the wrapped function and clears
        it after.

        Two forms:
            @dumper.ctx(lambda self: dict(layer_id=self.layer_id))
            def forward(self, x): ...

            @dumper.ctx(phase="decode")
            def decode_step(self, x): ...
        """
        if _extractor is not None and static_ctx:
            raise ValueError("cannot mix lambda extractor with static kwargs")
        if _extractor is None and not static_ctx:
            raise ValueError("must provide either a lambda or static kwargs")

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx_dict: dict = _extractor(args[0]) if _extractor else static_ctx
                self.set_ctx(**ctx_dict)
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.set_ctx(**{k: None for k in ctx_dict})

            return wrapper

        return decorator

    def apply_source_patches(self) -> None:
        """Apply source patches from DUMPER_SOURCE_PATCHER_CONFIG if set.

        Automatically injects ``from vllm.utils.debug_utils.dumper import dumper``
        into every replacement block so users don't need to write it in YAML.
        """
        config_path = self._config.source_patcher_config
        if not config_path:
            return

        from vllm.utils.debug_utils.source_patcher import apply_patches_from_config

        yaml_content: str = Path(config_path).read_text()
        _log(f"[source_patcher] loading config from {config_path}")
        apply_patches_from_config(
            yaml_content,
            extra_imports=["from vllm.utils.debug_utils.dumper import dumper"],
        )

    def register_non_intrusive_dumper(
        self,
        model: "torch.nn.Module",
    ) -> Optional["_NonIntrusiveDumper"]:
        self._http_manager  # noqa: B018
        mode = self._config.non_intrusive_mode
        if mode == "off":
            return None
        non_intrusive = _NonIntrusiveDumper(dumper=self, model=model, mode=mode)
        self._non_intrusives.append(non_intrusive)
        return non_intrusive

    # public :: secondary

    def configure(self, **kwargs) -> None:
        self._config = replace(self._config, **kwargs)

    def control(self, method: str, body: dict | str | None = None) -> dict:
        if method == "get_state":
            return self.get_state()
        if method == "configure":
            if isinstance(body, str):
                body = json.loads(body)
            self.configure(**(body or {}))
        elif method == "reset":
            self.reset()
        else:
            raise ValueError(f"Unknown dumper control method: {method!r}")
        return {}

    def configure_default(self, **kwargs) -> None:
        self._config = self._config.with_defaults(**kwargs)

    def reset(self) -> None:
        for non_intrusive in self._non_intrusives:
            non_intrusive.remove()
        self._non_intrusives.clear()
        self._state = _DumperState()

    @contextmanager
    def capture_output(self):
        assert self._state.captured_output_data is None
        self._state.captured_output_data = {}
        try:
            yield self._state.captured_output_data
        finally:
            self._state.captured_output_data = None

    def get_state(self) -> dict:
        return {
            "config": asdict(self._config),
            "dump_index": self._state.dump_index,
            "step": self._state.step,
        }

    @cached_property
    def _http_manager(self) -> Optional["_DumperHttpManager"]:
        if self._config.server_port_parsed in (None, "reuse"):
            return None
        return _DumperHttpManager(self)

    # ------------------------- private :: related to dump -----------------------------

    def _dump_inner(
        self,
        *,
        name: str,
        value,
        extra_kwargs: dict,
        save: bool,
        enable_value: bool,
        enable_curr_grad: bool,
        enable_future_grad: bool,
        value_tag: str,
        grad_tag: str,
        value_meta_only_fields: dict | None = None,
        grad_meta_only_fields: dict | None = None,
        grafter_extras: dict | None = None,
        get_grad: Callable | None = None,
        step: int | None = None,
    ) -> None:
        self._http_manager  # noqa: B018

        if not self._config.enable:
            return

        recompute_status = _detect_recompute_status()
        tags = dict(
            name=name,
            recompute_status=recompute_status.value,
            **extra_kwargs,
            **self._state.global_ctx,
        )

        if (f := self._config.filter) is not None and not _evaluate_filter(f, tags):
            return

        if not (enable_value or enable_curr_grad or enable_future_grad):
            return

        recompute_meta = recompute_status.to_pseudo_parallel_meta()
        value = _materialize_value(value)
        self._grafter.maybe_intercept(value=value, tags=tags, extras=grafter_extras)

        if enable_value:
            self._dump_single(
                tag=value_tag,
                tags=tags,
                value=value,
                save=save,
                step=step,
                meta_only_fields={**(value_meta_only_fields or {}), **recompute_meta},
            )

        if enable_curr_grad and isinstance(value, torch.Tensor):
            g = get_grad(value) if get_grad is not None else value.grad
        else:
            g = None

        if g is not None:
            self._dump_single(
                tag=grad_tag,
                tags={**tags, "name": f"grad__{name}"},
                value=g,
                save=save,
                step=step,
                meta_only_fields={**(grad_meta_only_fields or {}), **recompute_meta},
            )

        if enable_future_grad:
            self._register_dump_grad_hook(
                name=name,
                tensor=value,
                extra_kwargs=extra_kwargs,
                save=save,
                meta_only_fields=grad_meta_only_fields or {},
            )

    def _register_dump_grad_hook(
        self,
        *,
        name: str,
        tensor,
        extra_kwargs: dict,
        save: bool,
        meta_only_fields: dict | None = None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        if not tensor.requires_grad:
            return

        captured_step = self._state.step
        captured_tags = dict(
            name=f"grad__{name}",
            **deepcopy(extra_kwargs),
        )
        captured_meta_only = meta_only_fields or {}

        def grad_hook(grad: torch.Tensor) -> None:
            self._dump_single(
                tag="Dumper.Grad",
                tags=captured_tags,
                value=grad,
                save=save,
                step=captured_step,
                meta_only_fields=captured_meta_only,
            )

        tensor.register_hook(grad_hook)

    def _dump_single(
        self,
        *,
        tag: str,
        tags: dict,
        value,
        save: bool,
        step: int | None = None,
        meta_only_fields: dict | None = None,
    ) -> None:
        self._ensure_exp_name()
        self._state.dump_index += 1

        rank = _get_rank()
        full_kwargs = dict(
            step=(step if step is not None else self._state.step),
            rank=rank,
            dump_index=self._state.dump_index,
            **tags,
        )
        if self._config.include_parallel_rank_in_filename:
            full_kwargs.update(_collect_parallel_rank_tags())
        full_filename = _format_tags(full_kwargs) + ".pt"
        path = Path(self._config.dir) / self._config.exp_name / full_filename

        if self._config.enable_output_console:
            _log(
                f"[{tag}] {path} "
                f"type={type(value)} "
                f"shape={value.shape if isinstance(value, torch.Tensor) else None} "
                f"dtype={value.dtype if isinstance(value, torch.Tensor) else None} "
                f"device={value.device if isinstance(value, torch.Tensor) else None} "
                f"id={id(value)} "
                f"sample_value={get_truncated_value(value)}"
            )

        capturing = self._state.captured_output_data is not None
        if save and (self._config.enable_output_file or capturing):
            output_data = {
                "value": value,
                "meta": dict(
                    **full_kwargs,
                    **self._static_meta,
                    **(meta_only_fields or {}),
                ),
            }

            if capturing:
                output_data["value"] = _deepcopy_or_clone(output_data["value"])
                self._state.captured_output_data[tags["name"]] = output_data
            else:
                if (
                    not self._state.cleanup_previous_handled
                    and self._config.cleanup_previous
                ):
                    self._state.cleanup_previous_handled = True
                    _cleanup_old_dumps(
                        Path(self._config.dir), exp_name=self._config.exp_name
                    )

                path.parent.mkdir(parents=True, exist_ok=True)
                _torch_save(output_data, str(path))

    # ------------------------------- private :: misc ---------------------------------

    @cached_property
    def _static_meta(self) -> dict:
        return _compute_static_meta()

    def _ensure_exp_name(self):
        if self._config.exp_name is None:
            name = _get_default_exp_name(
                timeout_seconds=self._config.collective_timeout
            )
            self.configure(exp_name=name)
            _log(f"Choose exp_name={name}")


# hook dumper


class _NonIntrusiveDumper:
    _NAME_PREFIX = "non_intrusive__"
    _LAYER_NAME_RE = re.compile(r"(?:.+\.)?layers\.(\d+)$")

    def __init__(
        self,
        dumper: _Dumper,
        model: "torch.nn.Module",
        mode: str,
    ):
        self._dumper = dumper
        self._mode = mode
        self._handles: list = []
        self._core_fields: frozenset[str] = frozenset().union(
            *(p.core_fields() for p in _plugins)
        )

        for module_name, module in model.named_modules():
            if ctx := self._detect_module_ctx(module_name, module):
                self._register_ctx_hooks(module, ctx=ctx)

            is_root = module_name == ""
            pre_hook = self._make_forward_pre_hook(
                module_name=module_name, is_root=is_root
            )
            hook = self._make_forward_hook(module_name=module_name, is_root=is_root)
            self._handles += _register_forward_hook_or_replace_fn(
                module,
                pre_hook=pre_hook,
                hook=hook,
                mode="replace_fn" if is_root else "hook",
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @classmethod
    def _detect_module_ctx(
        cls, module_name: str, module: "torch.nn.Module"
    ) -> dict | None:
        match = cls._LAYER_NAME_RE.fullmatch(module_name)
        if match:
            for plugin in _plugins:
                layer_id = plugin.detect_layer_id(module)
                if layer_id is not None:
                    return {"layer_id": layer_id}
            return {"layer_id": int(match.group(1))}
        return None

    def _register_ctx_hooks(self, module: "torch.nn.Module", *, ctx: dict) -> None:
        clear_ctx = {k: None for k in ctx}
        self._handles.append(
            module.register_forward_pre_hook(
                lambda _mod, _input, _ctx=ctx: self._dumper.set_ctx(**_ctx)
            )
        )
        self._handles.append(
            module.register_forward_hook(
                lambda _mod, _input, _output, _clear=clear_ctx: self._dumper.set_ctx(
                    **_clear
                )
            )
        )

    def _make_forward_pre_hook(self, *, module_name: str, is_root: bool):
        def _hook(_module, args, kwargs):
            for i, item in enumerate(args):
                self._dump_value(
                    module_name, item, sub_name=f"inputs.{i}", is_root=is_root
                )
            for name, value in kwargs.items():
                self._dump_value(
                    module_name,
                    value,
                    sub_name=f"inputs.{name}",
                    is_root=is_root,
                )

        return _hook

    def _make_forward_hook(self, *, module_name: str, is_root: bool):
        def _hook(_module, input, output):
            if output is not None:
                self._dump_value(module_name, output, sub_name="output", is_root=False)

        return _hook

    def _dump_value(
        self, module_name: str, value: Any, sub_name: str, *, is_root: bool
    ) -> None:
        for key, item in self._convert_value(
            value, skip_forward_batch=(not is_root)
        ).items():
            effective_key = key or sub_name.rsplit(".", 1)[-1]
            if effective_key in self._core_fields:
                self._dumper.dump(effective_key, item)
            elif self._mode == "all":
                parts = [p for p in (module_name, sub_name, key) if p]
                self._dumper.dump(self._NAME_PREFIX + ".".join(parts), item)

    @staticmethod
    def _convert_value(value, *, skip_forward_batch: bool = False) -> dict[str, Any]:
        if isinstance(value, torch.Tensor):
            return {"": value}

        if isinstance(value, (tuple, list)):
            tensors = [t for t in value if isinstance(t, torch.Tensor)]
            if len(tensors) == 1:
                return {"": tensors[0]}
            return {str(i): t for i, t in enumerate(tensors)}

        for plugin in _plugins:
            result = plugin.convert_value(value, skip_forward_batch=skip_forward_batch)
            if result is not None:
                return result

        return {}


def _register_forward_hook_or_replace_fn(
    module: "torch.nn.Module",
    *,
    pre_hook,
    hook,
    mode: str,
) -> list:
    """Attach pre/post forward hooks to *module*.

    mode="hook"       — standard ``register_forward_pre_hook`` /
    ``register_forward_hook``
                        (fires only via ``__call__``).
    mode="replace_fn" — monkey-patch ``module.forward`` so hooks fire even when
                        callers invoke ``.forward()`` directly (as sglang does for the
                        root model).

    Returns a list of handle objects with a ``.remove()`` method that undoes
    the registration.
    """
    if mode == "hook":
        return [
            module.register_forward_pre_hook(pre_hook, with_kwargs=True),
            module.register_forward_hook(hook),
        ]
    elif mode == "replace_fn":
        original_forward = module.forward

        @functools.wraps(original_forward)
        def _wrapped(*args, **kwargs):
            pre_hook(module, args, kwargs)
            output = original_forward(*args, **kwargs)
            hook(module, args, output)
            return output

        module.forward = _wrapped

        class _Handle:
            def remove(self) -> None:
                assert module.forward is _wrapped
                module.forward = original_forward

        return [_Handle()]
    else:
        raise ValueError(f"Unknown mode {mode!r}")


# grafter


class _GraftRole(enum.Enum):
    BASELINE = "baseline"
    TARGET = "target"


class _GraftDirection(enum.Enum):
    B2T = "b2t"  # name flows baseline -> target
    T2B = "t2b"  # name flows target -> baseline


@dataclass
class GraftTransformInput:
    """Single argument passed to a user-supplied transform function.

    User transforms have signature::

        def transform(graft_input: GraftTransformInput) -> torch.Tensor: ...

    The dataclass shape lets us add fields (e.g., direction, sender ranks)
    later without breaking existing transforms.
    """

    # Full dumper.dump tags dict (name + recompute_status + extra_kwargs + ctx).
    tags: "dict[str, Any]"
    # One tensor per sender rank, in sender-rank order.
    received_list: "list[torch.Tensor]"
    # Parallel list of per-sender `grafter_extras` (the dict passed to
    # dumper.dump on each sender; None if the sender omitted it).
    received_extras_list: "list[dict | None]"
    # Recv side's local tensor that will be copy_'d into.
    target: "torch.Tensor"


class _Grafter:
    """Cross-system tensor transplant. Triggered silently from dumper.dump.

    Both sides set the SAME grafter_b2t_filter (names that flow baseline ->
    target) and grafter_t2b_filter (names that flow target -> baseline). The
    only per-side difference is grafter_role ("baseline" | "target"), which
    determines whether a name match means send or recv on this side.

    Graft global rank layout: baseline occupies ranks 0..baseline_world-1;
    target occupies ranks baseline_world..baseline_world+target_world-1. Each
    side derives its own rank from its local default PG via dist.get_rank().

    Please refer to TestGrafterE2eExample in tests for an example.
    """

    def __init__(self, *, config: DumperConfig):
        self._config = config
        self._pg = None

    @property
    def enabled(self) -> bool:
        return self._config.grafter_enable

    def maybe_intercept(
        self, *, value: Any, tags: dict, extras: dict | None = None
    ) -> None:
        """Intercept a dumper.dump call. `extras` is per-call auxiliary data
        (e.g., shard layout, dtype hint) that the sender attaches and the
        recv side's transform receives as `received_extras_list`."""
        cfg = self._config
        if not cfg.grafter_enable:
            return

        direction = self._classify_direction(tags)
        if direction is None:
            return

        if not isinstance(value, torch.Tensor):
            _log(
                f"[Grafter] tags={tags} matched grafter_{direction.value}_filter but "
                f"value is not a torch.Tensor (got type={type(value).__name__}); "
                f"skipping graft. Common cause: dumper.dump called with a non-tensor "
                f"value (dict, list, ...) on this name. Either narrow the filter or "
                f"wrap the value in a tensor."
            )
            return

        self._ensure_group()
        role = _GraftRole(cfg.grafter_role)
        is_send = self._is_sender(role=role, direction=direction)

        # all-gather over the graft world; sender ranks contribute (value,
        # extras) tuples, recv ranks contribute None (their local target is
        # private and shouldn't leak). all_gather_object is pickle-routed,
        # so tensor shapes may differ across sender ranks.
        total_world = cfg.grafter_baseline_world_size + cfg.grafter_target_world_size
        my_contribution = (value, extras) if is_send else None
        gathered: list = [None] * total_world
        dist.all_gather_object(gathered, my_contribution, group=self._pg)

        if is_send:
            _log(
                f"[Grafter] send role={role.value} dir={direction.value} "
                f"tags={tags} extras={extras} local={get_tensor_info(value)}"
            )
            return

        sender_contribs = self._sender_slice(direction=direction, gathered=gathered)
        # Pickled CUDA tensors are restored on their original-device name;
        # that may not match this process's local device, so normalize.
        sender_tensors = [
            (c[0].to(value.device) if isinstance(c[0], torch.Tensor) else c[0])
            for c in sender_contribs
        ]
        sender_extras = [c[1] for c in sender_contribs]

        # Transform + copy_ are wrapped: a buggy user transform must NOT
        # crash the whole training/inference run. On error we log the full
        # traceback and skip this graft point; downstream sees the recv
        # side's original tensor unchanged.
        info_before_overridden = get_tensor_info(value)
        try:
            value_to_override = self._apply_transform(
                tags=tags,
                received_list=sender_tensors,
                received_extras_list=sender_extras,
                target=value,
            )
            diff = _compare_tensors_quick(value, value_to_override)
            _log(
                f"[Grafter] recv role={role.value} dir={direction.value} "
                f"tags={tags} n_senders={len(sender_tensors)} "
                f"sender_extras={sender_extras} "
                f"before_overridden={info_before_overridden} "
                f"to_override={get_tensor_info(value_to_override)} "
                f"diff_pre_vs_new={diff}"
            )
            value.copy_(value_to_override)
        except Exception as e:
            _log(
                f"[Grafter] recv role={role.value} dir={direction.value} "
                f"tags={tags} transform/copy_ raised {type(e).__name__}: {e}; "
                f"skipping graft for this call (target tensor unchanged)\n"
                f"{traceback.format_exc()}"
            )

    def _classify_direction(self, tags: dict) -> Optional["_GraftDirection"]:
        cfg = self._config
        match_b2t = self._match(cfg.grafter_b2t_filter, tags)
        match_t2b = self._match(cfg.grafter_t2b_filter, tags)
        if match_b2t and match_t2b:
            raise RuntimeError(
                f"[Grafter] tags={tags} matched BOTH grafter_b2t_filter and "
                f"grafter_t2b_filter"
            )
        if match_b2t:
            return _GraftDirection.B2T
        if match_t2b:
            return _GraftDirection.T2B
        return None

    @staticmethod
    def _is_sender(*, role: "_GraftRole", direction: "_GraftDirection") -> bool:
        # baseline is the sender for B2T names; target is the sender for T2B.
        return (role == _GraftRole.BASELINE) == (direction == _GraftDirection.B2T)

    def _sender_slice(self, *, direction: "_GraftDirection", gathered: list) -> list:
        cfg = self._config
        if direction == _GraftDirection.B2T:
            return gathered[: cfg.grafter_baseline_world_size]
        return gathered[cfg.grafter_baseline_world_size :]

    @staticmethod
    def _match(expr: str | None, tags: dict) -> bool:
        if expr is None:
            return False
        return _evaluate_filter(expr, tags)

    def _ensure_group(self) -> None:
        if self._pg is not None:
            return

        cfg = self._config
        assert dist.is_initialized(), (
            "[Grafter] default torch.distributed must be initialized"
        )
        role = _GraftRole(cfg.grafter_role)
        local_world = dist.get_world_size()
        local_rank = dist.get_rank()
        if role == _GraftRole.BASELINE:
            assert local_world == cfg.grafter_baseline_world_size, (
                f"[Grafter] "
                f"grafter_baseline_world_size={cfg.grafter_baseline_world_size} "
                f"but dist.get_world_size()={local_world}; they must match on "
                f"the baseline side"
            )
            global_rank = local_rank
        else:
            assert local_world == cfg.grafter_target_world_size, (
                f"[Grafter] grafter_target_world_size={cfg.grafter_target_world_size} "
                f"but dist.get_world_size()={local_world}; they must match on "
                f"the target side"
            )
            global_rank = cfg.grafter_baseline_world_size + local_rank
        total_world = cfg.grafter_baseline_world_size + cfg.grafter_target_world_size
        init_method = f"tcp://{cfg.grafter_master_address}:{cfg.grafter_master_port}"
        _log(
            f"[Grafter] init group: role={role.value} "
            f"baseline_world={cfg.grafter_baseline_world_size} "
            f"target_world={cfg.grafter_target_world_size} "
            f"rank={global_rank} init_method={init_method} "
            f"backend={cfg.grafter_backend} name={cfg.grafter_group_name}"
        )
        self._pg = _collective_with_timeout(
            lambda: _init_custom_process_group(
                backend=cfg.grafter_backend,
                init_method=init_method,
                world_size=total_world,
                rank=global_rank,
                group_name=cfg.grafter_group_name,
            ),
            operation_name="_init_custom_process_group in _Grafter",
            timeout_seconds=cfg.grafter_timeout,
        )

    def _apply_transform(
        self,
        *,
        tags: dict,
        received_list: list,
        received_extras_list: list,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: integrate with dump_comparator unsharder annotations once
        # full inverse (sharded -> global -> sharded) transforms exist.
        graft_input = GraftTransformInput(
            tags=tags,
            received_list=received_list,
            received_extras_list=received_extras_list,
            target=target,
        )
        path = self._config.grafter_transform_path
        fn = self._default_transform if path is None else _load_function(path)
        return fn(graft_input)

    @staticmethod
    def _default_transform(graft_input: GraftTransformInput) -> torch.Tensor:
        """Identity-by-rank fallback. Requires #senders == #recvs and
        shape(received_list[my_recv_rank]) == shape(target). Otherwise raises
        and asks the user for a transform."""
        received_list = graft_input.received_list
        target = graft_input.target
        my_recv_rank = dist.get_rank()
        recv_world_size = dist.get_world_size()
        if len(received_list) != recv_world_size:
            raise RuntimeError(
                _Grafter._default_transform_error(
                    f"requires #senders == #recvs but got "
                    f"#senders={len(received_list)} vs #recvs={recv_world_size}"
                )
            )
        candidate = received_list[my_recv_rank]
        if candidate.shape != target.shape:
            raise RuntimeError(
                _Grafter._default_transform_error(
                    f"requires matching shapes but "
                    f"received_list[{my_recv_rank}].shape={tuple(candidate.shape)} "
                    f"!= target.shape={tuple(target.shape)}"
                )
            )
        return candidate

    @staticmethod
    def _default_transform_error(detail: str) -> str:
        return (
            f"[Grafter] no grafter_transform_path set; default identity-by-rank "
            f"{detail}. Provide a transform via "
            f"DUMPER_GRAFTER_TRANSFORM_PATH=pkg.module.symbol defining "
            f"`transform(graft_input: GraftTransformInput) -> Tensor`."
        )


# util fn


def _torch_save(value, path: str):
    value = _clone_if_view(value)
    try:
        try:
            return torch.save(value, path)
        except RuntimeError as e:
            if "not pickleable" in str(e):
                stripped = _strip_parameter(value)
                if stripped is not value:
                    _log(f"Observe error={e} and try pickling .data")
                    return _torch_save(stripped, path)
            raise
    except Exception as e:
        _log(f"Observe error={e} when saving data, skip the tensor")


def _map_tensor(value, fn: Callable[[torch.Tensor], torch.Tensor]):
    if isinstance(value, dict):
        return {k: _map_tensor(v, fn) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return fn(value)
    return value


def _clone_if_view(value):
    def _fn(t: torch.Tensor) -> torch.Tensor:
        if t.untyped_storage().nbytes() > t.nelement() * t.element_size():
            return t.clone()
        return t

    return _map_tensor(value, _fn)


def _strip_parameter(value):
    def _fn(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.nn.Parameter):
            return t.data
        return t

    return _map_tensor(value, _fn)


def _collective_with_timeout(fn, operation_name: str, timeout_seconds: int = 60):
    completed = threading.Event()

    def watchdog():
        if not completed.wait(timeout=timeout_seconds):
            _log(
                f"WARNING: '{operation_name}' has not completed after "
                f"{timeout_seconds}s. This usually means not all ranks are "
                f"participating in this collective operation."
            )

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        completed.set()


def _get_default_exp_name(timeout_seconds: int = 60):
    rank = _get_rank()
    now = time.time()
    ms = int((now % 1) * 1000)
    rand_suffix = random.randint(0, 999)
    object_list = [
        (
            (
                f"{_DEFAULT_EXP_NAME_PREFIX}"
                f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime(now))}"
                f"_{ms:03d}{rand_suffix:03d}"
            )
            if rank == 0
            else None
        )
    ]

    if dist.is_initialized():
        _collective_with_timeout(
            lambda: dist.broadcast_object_list(object_list, device=get_device()),
            operation_name="broadcast_object_list in _get_default_exp_name",
            timeout_seconds=timeout_seconds,
        )

    return object_list[0]


def _cleanup_old_dumps(base_dir: Path, exp_name: str | None = None) -> None:
    import shutil

    if _get_rank() == 0:
        targets = {entry for entry in base_dir.glob(f"{_DEFAULT_EXP_NAME_PREFIX}*")}
        if exp_name:
            targets.add(base_dir / exp_name)
        targets = {d for d in targets if d.is_dir()}

        for entry in targets:
            shutil.rmtree(entry)
            _log(f"Cleaned up {entry}")

    if dist.is_initialized():
        _collective_with_timeout(
            dist.barrier,
            operation_name="barrier in _cleanup_old_dumps",
        )


def _get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def _get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def _log(msg: str) -> None:
    """Print a log line tagged with the current rank and wall-clock time."""
    print(f"[Dumper, rank={_get_rank()}, t={time.time():.3f}] {msg}", flush=True)


def _compare_tensors_quick(a: "torch.Tensor", b: "torch.Tensor") -> str:
    """One-line summary of how close two tensors are. Inspired by
    vllm.utils.debug_utils.dump_comparator._compute_and_print_diff;
    intentionally inlined here to keep dumper.py free of cross-file imports.

    Different dtypes are fine — we unify by casting both to fp32, which is
    enough for the order-of-magnitude diff summary we log."""
    if a.shape != b.shape:
        return f"shape mismatch (a={tuple(a.shape)} vs b={tuple(b.shape)})"
    if a.numel() == 0:
        return "empty"
    a_float = a.detach().to(torch.float32)
    b_float = b.detach().to(torch.float32)
    raw_abs = (a_float - b_float).abs()
    max_abs = raw_abs.max().item()
    mean_abs = raw_abs.mean().item()
    rel_diff = _calc_rel_diff(a_float, b_float).item()
    return f"rel_diff={rel_diff:.6g} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"


# Copied verbatim from vllm.utils.debug_utils.dump_comparator (originally from
# DeepGEMM). Kept inline here so dumper.py has no cross-file imports.
def _calc_rel_diff(x: "torch.Tensor", y: "torch.Tensor"):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _obj_to_dict(obj):
    if isinstance(obj, dict):
        return obj
    for plugin in _plugins:
        converted = plugin.convert_value(obj, skip_forward_batch=False)
        if converted is not None:
            return converted
    ret = {}
    for k in dir(obj):
        if k.startswith("__") and k.endswith("__"):
            continue
        try:
            v = getattr(obj, k)
            if not callable(v):
                ret[k] = v
        except Exception:
            # Skip attributes that raise an exception on access
            continue
    return ret


def _materialize_value(value):
    if callable(value):
        value = value()
    return value


_PARALLEL_RANK_KEYS = ("pp_rank", "tp_rank", "cp_rank", "ep_rank", "etp_rank")


def _collect_parallel_rank_tags() -> dict[str, int]:
    """Collect parallel-rank tags from framework plugins for use in dump filenames.

    Merges the ``_PARALLEL_RANK_KEYS`` reported by each plugin's
    ``collect_parallel_info()``; the first plugin to report a given key wins.
    """
    result: dict[str, int] = {}
    for plugin in _plugins:
        info = plugin.collect_parallel_info()
        if not info:
            continue
        for key in _PARALLEL_RANK_KEYS:
            if key in info and key not in result:
                result[key] = info[key]
    return result


def _format_tags(kwargs: dict) -> str:
    return "___".join(f"{k}={v}" for k, v in kwargs.items())


class _DefaultNoneDict(dict):
    """dict subclass that returns None for missing keys, for filter expression eval."""

    def __missing__(self, key: str):
        return None


_FILTER_BUILTINS: dict[str, Any] = {"search": re.search, "match": re.match}


def _evaluate_filter(filter_expr: str, tags: dict[str, Any]) -> bool:
    """Evaluate a Python filter expression against the tags dict.

    Unknown tag keys resolve to None, so `layer_id is None` works when layer_id is
    absent.
    `re.search` and `re.match` are available as `search()` and `match()`.
    """
    namespace = _DefaultNoneDict(tags)
    namespace.update(_FILTER_BUILTINS)
    return bool(eval(filter_expr, {"__builtins__": {}}, namespace))


def _deepcopy_or_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    return deepcopy(x)


# static meta


def _compute_static_meta():
    result = {
        "world_rank": _get_rank(),
        "world_size": _get_world_size(),
    }

    for plugin in _plugins:
        if info := plugin.collect_parallel_info():
            result[f"{plugin.name}_parallel_info"] = info

    for plugin in _plugins:
        tokenizer_path: str | None = plugin.get_tokenizer_path()
        if tokenizer_path is not None:
            result["tokenizer_path"] = tokenizer_path
            break

    return result


# http manager


class _DumperHttpManager:
    def __init__(self, dumper: "_Dumper"):
        self._dumper = dumper
        http_port = self._dumper._config.server_port_parsed

        rpc_broadcast = _create_zmq_rpc_broadcast(
            self,
            timeout_seconds=self._dumper._config.collective_timeout,
        )

        if _get_rank() == 0:
            assert rpc_broadcast is not None
            self._rpc_broadcast = rpc_broadcast

            if http_port == "reuse":
                _log("Standalone HTTP server disabled, reusing existing ports")
            else:
                _start_http_server(prefix="/dumper/", target=self, http_port=http_port)
                _log(f"HTTP server started on port {http_port}")

    # ------------------------------- public ---------------------------------

    def handle_request(self, *, method: str, body: dict[str, Any]) -> list[dict]:
        return self._rpc_broadcast._handle_request_inner(method=method, body=body)

    # ------------------------------- private ---------------------------------

    def _handle_request_inner(self, *, method: str, body: dict[str, Any]) -> dict:
        return self._dumper.control(method, body)


# http control server


def _start_http_server(*, prefix: str, target: object, http_port: int):
    handler_class = _make_http_handler(prefix=prefix, target=target)
    server = HTTPServer(("0.0.0.0", http_port), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def _make_http_handler(*, prefix: str, target):
    class _HTTPHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if not self.path.startswith(prefix):
                self.send_error(404)
                return
            method = self.path[len(prefix) :]
            try:
                req_body = self._get_request_body()
                _log(f"HTTP {self.path} {req_body=}")
                result = target.handle_request(method=method, body=req_body)
                resp_body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
            except Exception as e:
                self.send_error(400, str(e))

        def _get_request_body(self) -> dict:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return {}
            return json.loads(self.rfile.read(content_length))

    return _HTTPHandler


# zmq rpc


def _create_zmq_rpc_broadcast(
    handler, timeout_seconds: int = 60
) -> Optional["_ZmqRpcBroadcast"]:
    """A general-purpose minimal RPC to support broadcasting executions to multi
    processes
    """
    rank = _get_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:0")
    bound_port = int(sock.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
    local_addr = f"tcp://{_get_local_ip_by_remote()}:{bound_port}"

    def serve_loop():
        while True:
            try:
                req = sock.recv_pyobj()
                result = getattr(handler, req["method"])(*req["args"], **req["kwargs"])
                resp = {"result": result, "error": None}
            except Exception as e:
                _log(f"[ZmqRpc] error inside handler: {e}")
                resp = {"result": None, "error": str(e)}
            sock.send_pyobj(resp)

    thread = threading.Thread(target=serve_loop, daemon=True)
    thread.start()
    _log(f"[ZmqRpc] server started at {local_addr}")

    if dist.is_initialized():
        all_addresses = [None] * world_size
        _collective_with_timeout(
            lambda: dist.all_gather_object(all_addresses, local_addr),
            operation_name="all_gather_object in _create_zmq_rpc_broadcast",
            timeout_seconds=timeout_seconds,
        )
    else:
        all_addresses = [local_addr]
    _log(f"[ZmqRpc] all_addresses={all_addresses}")

    if rank == 0:
        handles = []
        for i, addr in enumerate(all_addresses):
            req_socket = ctx.socket(zmq.REQ)
            req_socket.connect(addr)
            handles.append(_ZmqRpcHandle(req_socket, debug_name=f"rank-{i}"))
        return _ZmqRpcBroadcast(handles)
    else:
        return None


class _ZmqRpcHandle:
    """Proxy object to call remote handler methods via ZMQ."""

    def __init__(self, socket, debug_name: str):
        self._socket = socket
        self._debug_name = debug_name

    def __getattr__(self, method_name: str):
        def call(*args, **kwargs):
            self._socket.send_pyobj(
                {
                    "method": method_name,
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            response = self._socket.recv_pyobj()
            if response["error"]:
                raise RuntimeError(
                    f"RPC error on {self._debug_name}: {response['error']}"
                )
            return response["result"]

        return call


class _RpcBroadcastBase:
    """Base for broadcasting method calls to dumper instance(s)."""

    def __getattr__(self, method_name: str):
        raise NotImplementedError

    def __init__(self, handles: list[_ZmqRpcHandle]):
        self._handles = handles


class _ZmqRpcBroadcast(_RpcBroadcastBase):
    """Broadcasts method calls to all ZMQ RPC handles.

    Returns a list of results, one per rank (ordered by rank).
    """

    def __init__(self, handles: list[_ZmqRpcHandle]):
        self._handles = handles

    def __getattr__(self, method_name: str):
        def call(*args, **kwargs):
            return [
                getattr(handle, method_name)(*args, **kwargs)
                for handle in self._handles
            ]

        return call


# copied code (avoid dependency)


def _get_local_ip_by_remote() -> str | None:
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
            return ip
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        _log("Can not get local ip by remote")
    return None


def _load_function(path: str) -> Callable:
    """Resolve a fully-qualified Python path 'pkg.module.symbol' to its object.

    Copied (verbatim, minus the function-registry branch) from
    miles.utils.misc.load_function — kept inline so dumper.py has no
    cross-package dependency.
    """
    import importlib

    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"_load_function expects 'pkg.module.symbol', got {path!r} "
            f"(missing dotted prefix)"
        )
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _init_custom_process_group(
    *,
    backend: str,
    init_method: str,
    world_size: int,
    rank: int,
    group_name: str,
    timeout=None,
):
    """Build a fresh torch.distributed process group, separate from the default
    one and any other custom groups (e.g. RLHF weight-update groups). Used by
    the grafter to bridge baseline and target systems.

    Adapted from sglang.srt.utils.common.init_custom_process_group; inlined
    here to keep dumper.py free of cross-file imports.
    """
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    if timeout is None:
        timeout = default_pg_timeout

    rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)
    store = PrefixStore(group_name, store)

    backend_obj = Backend(backend)
    # PyTorch 2.6 renamed `pg_options` to `backend_options`.
    torch_major_minor = tuple(
        int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
    )
    pg_options_param_name = (
        "backend_options" if torch_major_minor >= (2, 6) else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend_obj,
        store,
        group_name=group_name,
        **{pg_options_param_name: None},
        timeout=timeout,
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


# framework plugins


class _RecomputeStatus(enum.Enum):
    DISABLED = "disabled"
    ORIGINAL = "original"  # inside checkpoint, original forward
    RECOMPUTE = "recompute"  # inside checkpoint, recompute forward

    def to_pseudo_parallel_meta(self) -> dict[str, Any]:
        if self == _RecomputeStatus.DISABLED:
            return {}
        return {
            "recompute_pseudo_rank": 1 if self == _RecomputeStatus.RECOMPUTE else 0,
            "recompute_pseudo_size": 2,
        }


class _FrameworkPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def collect_parallel_info(self) -> dict: ...

    @abstractmethod
    def convert_value(
        self, value: Any, *, skip_forward_batch: bool
    ) -> dict[str, Any] | None:
        """Return converted dict, or None if this plugin doesn't handle the value."""
        ...

    @abstractmethod
    def detect_layer_id(self, module: "torch.nn.Module") -> int | None:
        """Return 0-indexed layer_id, or None if not detectable."""
        ...

    def core_fields(self) -> frozenset[str]:
        return frozenset()

    def get_tokenizer_path(self) -> str | None:
        return None

    def detect_recompute_status(self) -> _RecomputeStatus:
        return _RecomputeStatus.DISABLED

    def transform_model_param_name(
        self, model: "torch.nn.Module", param_name: str
    ) -> str | None:
        """Return a rewritten parameter name, or None to keep the original.

        Used by ``dump_model`` to canonicalize parameter names across parallel
        layouts (e.g. mapping pipeline-local layer indices to global ones).
        """
        return None


class _VLLMPlugin(_FrameworkPlugin):
    @property
    def name(self) -> str:
        return "vllm"

    def collect_parallel_info(self) -> dict:
        try:
            from vllm.distributed import parallel_state
        except ImportError:
            return {}

        info = {}
        for name in ("tp", "pp", "dp", "ep", "etp", "dcp", "pcp"):
            try:
                group = getattr(parallel_state, f"get_{name}_group")()
                info[f"{name}_rank"] = group.rank_in_group
                info[f"{name}_size"] = group.world_size
            except (AssertionError, RuntimeError):
                continue
        return info

    def convert_value(
        self, value: Any, *, skip_forward_batch: bool
    ) -> dict[str, Any] | None:
        if not type(value).__module__.startswith("vllm."):
            return None
        from vllm.sequence import IntermediateTensors
        from vllm.v1.worker.gpu.input_batch import InputBatch

        if isinstance(value, InputBatch):
            if skip_forward_batch:
                return {}
            return {
                "input_ids": value.input_ids[: value.num_tokens],
                "positions": value.positions[: value.num_tokens],
                "seq_lens": value.seq_lens[: value.num_reqs],
                "rids": value.req_ids,
                "extend_seq_lens": value.num_scheduled_tokens.tolist(),
            }
        if isinstance(value, IntermediateTensors):
            return dict(value.tensors)
        return None

    def detect_layer_id(self, module: "torch.nn.Module") -> int | None:
        return getattr(module, "layer_id", None)

    def core_fields(self) -> frozenset[str]:
        return frozenset({"input_ids", "positions", "seq_lens", "rids"})

    def get_tokenizer_path(self) -> str | None:
        try:
            from vllm.config import get_current_vllm_config

            return get_current_vllm_config().model_config.tokenizer
        except (ImportError, AssertionError, RuntimeError):
            return None


class _MegatronPlugin(_FrameworkPlugin):
    _available = True
    try:
        from megatron.core import parallel_state as _mpu
        from megatron.core.packed_seq_params import PackedSeqParams
    except ImportError:
        _available = False

    @property
    def name(self) -> str:
        return "megatron"

    def collect_parallel_info(self) -> dict:
        if not self._available:
            return {}

        info = {}
        try:
            info["tp_rank"] = self._mpu.get_tensor_model_parallel_rank()
            info["tp_size"] = self._mpu.get_tensor_model_parallel_world_size()
            info["pp_rank"] = self._mpu.get_pipeline_model_parallel_rank()
            info["pp_size"] = self._mpu.get_pipeline_model_parallel_world_size()
            info["dp_rank"] = self._mpu.get_data_parallel_rank()
            info["dp_size"] = self._mpu.get_data_parallel_world_size()
            info["cp_rank"] = self._mpu.get_context_parallel_rank()
            info["cp_size"] = self._mpu.get_context_parallel_world_size()
            info["vpp_rank"] = self._mpu.get_virtual_pipeline_model_parallel_rank()
            info["vpp_size"] = (
                self._mpu.get_virtual_pipeline_model_parallel_world_size()
            )
            info["ep_rank"] = self._mpu.get_expert_model_parallel_rank()
            info["ep_size"] = self._mpu.get_expert_model_parallel_world_size()
            info["etp_rank"] = self._mpu.get_expert_tensor_parallel_rank()
            info["etp_size"] = self._mpu.get_expert_tensor_parallel_world_size()
            info["edp_rank"] = self._mpu.get_expert_data_parallel_rank()
            info["edp_size"] = self._mpu.get_expert_data_parallel_world_size()
            info["tcp_rank"] = self._mpu.get_tensor_and_context_parallel_rank()
            info["tcp_size"] = self._mpu.get_tensor_and_context_parallel_world_size()
            info["etmp_rank"] = self._mpu.get_expert_tensor_and_model_parallel_rank()
            info["etmp_size"] = (
                self._mpu.get_expert_tensor_and_model_parallel_world_size()
            )
            info["tp_src_rank"] = self._mpu.get_tensor_model_parallel_src_rank()
            info["mp_src_rank"] = self._mpu.get_model_parallel_src_rank()
            info["dp_src_rank"] = self._mpu.get_data_parallel_src_rank()
        except (AttributeError, AssertionError):
            info["megatron_error"] = True

        # Megatron sequence parallel reuses the TP group (no dedicated parallel state
        # API).
        # When sequence_parallel=True, inject sp_rank/sp_size for the comparator
        # unsharder.
        try:
            from megatron.training.global_vars import get_args

            args = get_args()
            if getattr(args, "sequence_parallel", False) and "tp_rank" in info:
                info["sp_rank"] = info["tp_rank"]
                info["sp_size"] = info["tp_size"]
        except (ImportError, AssertionError, AttributeError):
            pass

        return info

    def convert_value(
        self, value: Any, *, skip_forward_batch: bool
    ) -> dict[str, Any] | None:
        if not self._available:
            return None
        if isinstance(value, self.PackedSeqParams):
            return {
                "cu_seqlens_q": value.cu_seqlens_q,
                "cu_seqlens_kv": value.cu_seqlens_kv,
                "qkv_format": value.qkv_format,
            }
        return None

    def detect_layer_id(self, module: "torch.nn.Module") -> int | None:
        if hasattr(module, "layer_number"):
            return module.layer_number - 1
        return None

    def core_fields(self) -> frozenset[str]:
        return frozenset(
            {"input_ids", "position_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"}
        )

    def detect_recompute_status(self) -> _RecomputeStatus:
        if not self._available:
            return _RecomputeStatus.DISABLED
        try:
            from megatron.core.tensor_parallel.random import is_checkpointing

            if not is_checkpointing():
                return _RecomputeStatus.DISABLED
            if torch.is_grad_enabled():
                return _RecomputeStatus.RECOMPUTE
            return _RecomputeStatus.ORIGINAL
        except (ImportError, AttributeError):
            return _RecomputeStatus.DISABLED

    def transform_model_param_name(
        self, model: "torch.nn.Module", param_name: str
    ) -> str | None:
        """Rewrite pipeline-local layer indices to global ones in a param name.

        With pipeline parallelism, ``model.named_parameters()`` reports layer
        indices local to the current PP stage (e.g. ``layers.0`` on every stage).
        Adding the stage's ``get_transformer_layer_offset`` makes the dumped
        names globally unique and comparable across stages. Returns None (keep the
        original name) when not applicable.
        """
        if not self._available:
            return None

        try:
            pp_size = self._mpu.get_pipeline_model_parallel_world_size()
        except (AttributeError, AssertionError):
            return None
        if pp_size <= 1:
            return None

        config = self._get_model_config(model)
        if config is None:
            return None

        offset = self._get_transformer_layer_offset(config)
        if not offset:
            return None

        def _add_offset(match: "re.Match") -> str:
            return f"layers.{int(match.group(1)) + offset}"

        return re.sub(r"layers\.(\d+)", _add_offset, param_name)

    @staticmethod
    def _get_transformer_layer_offset(config) -> int:
        """Return the PP-stage layer offset for ``config``, or 0 if unavailable."""
        try:
            from megatron.core.transformer.transformer_layer import (
                get_transformer_layer_offset,
            )

            return get_transformer_layer_offset(config)
        except (ImportError, AttributeError, AssertionError):
            return 0

    @staticmethod
    def _get_model_config(model: "torch.nn.Module"):
        """Unwrap nested ``.module`` wrappers to reach the Megatron model config."""
        inner = model
        for _ in range(10):
            if hasattr(inner, "config"):
                return inner.config
            if hasattr(inner, "module"):
                inner = inner.module
            else:
                break
        return None


_plugins: list[_FrameworkPlugin] = [_VLLMPlugin(), _MegatronPlugin()]


def _detect_recompute_status() -> _RecomputeStatus:
    for plugin in _plugins:
        info = plugin.detect_recompute_status()
        if info != _RecomputeStatus.DISABLED:
            return info
    return _RecomputeStatus.DISABLED


# singleton


dumper = _Dumper(config=DumperConfig.from_env())


# other utility functions


def get_truncated_value(value):
    if value is None:
        return None

    if isinstance(value, tuple):
        return [get_truncated_value(x) for x in value]

    if not isinstance(value, torch.Tensor):
        return value

    if value.numel() < 200:
        return value

    slices = [slice(0, 5) if dim_size > 50 else slice(None) for dim_size in value.shape]
    return value[tuple(slices)]


def get_tensor_info(x):
    """From vllm.utils.debug_utils.dumper import get_tensor_info"""
    if not isinstance(x, torch.Tensor):
        return f"type={type(x)} value={x}"
    min = x.float().min() if x.numel() > 0 else None
    max = x.float().max() if x.numel() > 0 else None
    mean = x.float().mean() if x.numel() > 0 else None
    torch.set_printoptions(precision=10)
    x_sample_head = str(x.flatten()[:5])
    x_sample_tail = str(x.flatten()[-5:])
    torch.set_printoptions(precision=4)
    return (
        f"type={type(x)} "
        f"shape={x.shape} "
        f"dtype={x.dtype} "
        f"device={x.device} "
        f"stride={x.stride()} "
        f"req_grad={x.requires_grad} "
        f"min={min} "
        f"max={max} "
        f"mean={mean} "
        f"x_sample_head={x_sample_head} "
        f"x_sample_tail={x_sample_tail}"
    )
