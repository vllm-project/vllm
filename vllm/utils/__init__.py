# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import datetime
import enum
import getpass
import inspect
import multiprocessing
import os
import signal
import sys
import tempfile
import threading
import traceback
import uuid
import warnings
import weakref
from collections.abc import Callable
from functools import cache, partial, wraps
from typing import TYPE_CHECKING, Any, TypeVar

import cloudpickle
import psutil
import torch

import vllm.envs as envs
from vllm.logger import enable_trace_function_call, init_logger
from vllm.ray.lazy_utils import is_in_ray_actor

# Import utilities from specialized modules for backward compatibility
from vllm.utils.argparse_utils import (
    FlexibleArgumentParser,
    SortedHelpFormatter,
    StoreBoolean,
)
from vllm.utils.math_utils import (
    cdiv,
    next_power_of_2,
    prev_power_of_2,
    round_down,
    round_up,
)
from vllm.utils.platform_utils import cuda_is_initialized, xpu_is_initialized

__all__ = [
    # Argparse utilities
    "FlexibleArgumentParser",
    "SortedHelpFormatter",
    "StoreBoolean",
    # Math utilities
    "cdiv",
    "next_power_of_2",
    "prev_power_of_2",
    "round_down",
    "round_up",
]

_DEPRECATED_MAPPINGS = {
    "cprofile": "profiling",
    "cprofile_context": "profiling",
    "get_open_port": "network_utils",
}


def __getattr__(name: str) -> Any:  # noqa: D401 - short deprecation docstring
    """Module-level getattr to handle deprecated utilities."""
    if name in _DEPRECATED_MAPPINGS:
        submodule_name = _DEPRECATED_MAPPINGS[name]
        warnings.warn(
            f"vllm.utils.{name} is deprecated and will be removed in a future version. "
            f"Use vllm.utils.{submodule_name}.{name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = __import__(f"vllm.utils.{submodule_name}", fromlist=[submodule_name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # expose deprecated names in dir() for better UX/tab-completion
    return sorted(list(globals().keys()) + list(_DEPRECATED_MAPPINGS.keys()))


if TYPE_CHECKING:
    from argparse import Namespace

    from vllm.config import ModelConfig, VllmConfig
else:
    Namespace = object

    ModelConfig = object
    VllmConfig = object

logger = init_logger(__name__)

# This value is chosen to have a balance between ITL and TTFT. Note it is
# not optimized for throughput.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
POOLING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120

# Constants related to forcing the attention backend selection

# String name of register which may be set in order to
# force auto-selection of attention backend by Attention
# wrapper
STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"

# Possible string values of STR_BACKEND_ENV_VAR
# register, corresponding to possible backends
STR_FLASHINFER_ATTN_VAL: str = "FLASHINFER"
STR_TORCH_SDPA_ATTN_VAL: str = "TORCH_SDPA"
STR_XFORMERS_ATTN_VAL: str = "XFORMERS"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"


# ANSI color codes
CYAN = "\033[1;36m"
RESET = "\033[0;0m"


T = TypeVar("T")
U = TypeVar("U")


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class LayerBlockType(enum.Enum):
    attention = "attention"
    mamba = "mamba"


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def update_environment_variables(envs: dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


@cache
def is_pin_memory_available() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_pin_memory_available()


@cache
def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available."""
    # UVA requires pinned memory.
    # TODO: Add more requirements for UVA if needed.
    return is_pin_memory_available()


# TODO: This function can be removed if transformer_modules classes are
# serialized by value when communicating between processes
def init_cached_hf_modules() -> None:
    """
    Lazy initialization of the Hugging Face modules.
    """
    from transformers.dynamic_module_utils import init_hf_modules

    init_hf_modules()


def enable_trace_function_call_for_thread(vllm_config: VllmConfig) -> None:
    """Set up function tracing for the current thread,
    if enabled via the VLLM_TRACE_FUNCTION environment variable
    """

    if envs.VLLM_TRACE_FUNCTION:
        tmp_dir = tempfile.gettempdir()
        # add username to tmp_dir to avoid permission issues
        tmp_dir = os.path.join(tmp_dir, getpass.getuser())
        filename = (
            f"VLLM_TRACE_FUNCTION_for_process_{os.getpid()}"
            f"_thread_{threading.get_ident()}_"
            f"at_{datetime.datetime.now()}.log"
        ).replace(" ", "_")
        log_path = os.path.join(
            tmp_dir, "vllm", f"vllm-instance-{vllm_config.instance_id}", filename
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        enable_trace_function_call(log_path)


def weak_bind(
    bound_method: Callable[..., Any],
) -> Callable[..., None]:
    """Make an instance method that weakly references
    its associated instance and no-ops once that
    instance is collected."""
    ref = weakref.ref(bound_method.__self__)  # type: ignore[attr-defined]
    unbound = bound_method.__func__  # type: ignore[attr-defined]

    def weak_bound(*args, **kwargs) -> None:
        if inst := ref():
            unbound(inst, *args, **kwargs)

    return weak_bound


class AtomicCounter:
    """An atomic, thread-safe counter"""

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value"""
        self._value = initial
        self._lock = threading.Lock()

    def inc(self, num=1):
        """Atomically increment the counter by num and return the new value"""
        with self._lock:
            self._value += num
            return self._value

    def dec(self, num=1):
        """Atomically decrement the counter by num and return the new value"""
        with self._lock:
            self._value -= num
            return self._value

    @property
    def value(self):
        return self._value


def kill_process_tree(pid: int):
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501
def set_ulimit(target_soft_limit=65535):
    if sys.platform.startswith("win"):
        logger.info("Windows detected, skipping ulimit adjustment.")
        return

    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/utils.py#L28 # noqa: E501
def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def _maybe_force_spawn():
    """Check if we need to force the use of the `spawn` multiprocessing start
    method.
    """
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
        return

    reasons = []
    if is_in_ray_actor():
        # even if we choose to spawn, we need to pass the ray address
        # to the subprocess so that it knows how to connect to the ray cluster.
        # env vars are inherited by subprocesses, even if we use spawn.
        import ray

        os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        reasons.append("In a Ray actor and can only be spawned")

    if cuda_is_initialized():
        reasons.append("CUDA is initialized")
    elif xpu_is_initialized():
        reasons.append("XPU is initialized")

    if reasons:
        logger.warning(
            "We must use the `spawn` multiprocessing start method. "
            "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
            "See https://docs.vllm.ai/en/latest/usage/"
            "troubleshooting.html#python-multiprocessing "
            "for more information. Reasons: %s",
            "; ".join(reasons),
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def get_mp_context():
    """Get a multiprocessing context with a particular method (spawn or fork).
    By default we follow the value of the VLLM_WORKER_MULTIPROC_METHOD to
    determine the multiprocessing method (default is fork). However, under
    certain conditions, we may enforce spawn and override the value of
    VLLM_WORKER_MULTIPROC_METHOD.
    """
    _maybe_force_spawn()
    mp_method = envs.VLLM_WORKER_MULTIPROC_METHOD
    return multiprocessing.get_context(mp_method)


def bind_kv_cache(
    ctx: dict[str, Any],
    kv_cache: list[list[torch.Tensor]],  # [virtual_engine][layer_index]
    shared_kv_cache_layers: dict[str, str] | None = None,
) -> None:
    # Bind the kv_cache tensor to Attention modules, similar to
    # ctx[layer_name].kv_cache[ve]=kv_cache[ve][extract_layer_index(layer_name)]
    # Special things handled here:
    # 1. Some models have non-attention layers, e.g., Jamba
    # 2. Pipeline parallelism, each rank only has a subset of layers
    # 3. Encoder attention has no kv cache
    # 4. Encoder-decoder models, encoder-decoder attention and decoder-only
    #    attention of the same layer (e.g., bart's decoder.layers.1.self_attn
    #    and decoder.layers.1.encoder_attn) is mapped to the same kv cache
    #    tensor
    # 5. Some models have attention layers that share kv cache with previous
    #    layers, this is specified through shared_kv_cache_layers
    if shared_kv_cache_layers is None:
        shared_kv_cache_layers = {}
    from vllm.attention import AttentionType
    from vllm.model_executor.models.utils import extract_layer_index

    layer_need_kv_cache = [
        layer_name
        for layer_name in ctx
        if (
            hasattr(ctx[layer_name], "attn_type")
            and ctx[layer_name].attn_type
            in (AttentionType.DECODER, AttentionType.ENCODER_DECODER)
        )
        and ctx[layer_name].kv_sharing_target_layer_name is None
    ]
    layer_index_sorted = sorted(
        set(extract_layer_index(layer_name) for layer_name in layer_need_kv_cache)
    )
    for layer_name in layer_need_kv_cache:
        kv_cache_idx = layer_index_sorted.index(extract_layer_index(layer_name))
        forward_ctx = ctx[layer_name]
        assert len(forward_ctx.kv_cache) == len(kv_cache)
        for ve, ve_kv_cache in enumerate(kv_cache):
            forward_ctx.kv_cache[ve] = ve_kv_cache[kv_cache_idx]
    if shared_kv_cache_layers is not None:
        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            assert extract_layer_index(target_layer_name) < extract_layer_index(
                layer_name
            ), "v0 doesn't support interleaving kv sharing"
            ctx[layer_name].kv_cache = ctx[target_layer_name].kv_cache


def run_method(
    obj: Any,
    method: str | bytes | Callable,
    args: tuple[Any],
    kwargs: dict[str, Any],
) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


def import_pynvml():
    """
    Historical comments:

    libnvml.so is the library behind nvidia-smi, and
    pynvml is a Python wrapper around it. We use it to get GPU
    status without initializing CUDA context in the current process.
    Historically, there are two packages that provide pynvml:
    - `nvidia-ml-py` (https://pypi.org/project/nvidia-ml-py/): The official
        wrapper. It is a dependency of vLLM, and is installed when users
        install vLLM. It provides a Python module named `pynvml`.
    - `pynvml` (https://pypi.org/project/pynvml/): An unofficial wrapper.
        Prior to version 12.0, it also provides a Python module `pynvml`,
        and therefore conflicts with the official one. What's worse,
        the module is a Python package, and has higher priority than
        the official one which is a standalone Python file.
        This causes errors when both of them are installed.
        Starting from version 12.0, it migrates to a new module
        named `pynvml_utils` to avoid the conflict.
    It is so confusing that many packages in the community use the
    unofficial one by mistake, and we have to handle this case.
    For example, `nvcr.io/nvidia/pytorch:24.12-py3` uses the unofficial
    one, and it will cause errors, see the issue
    https://github.com/vllm-project/vllm/issues/12847 for example.
    After all the troubles, we decide to copy the official `pynvml`
    module to our codebase, and use it directly.
    """
    import vllm.third_party.pynvml as pynvml

    return pynvml


def warn_for_unimplemented_methods(cls: type[T]) -> type[T]:
    """
    A replacement for `abc.ABC`.
    When we use `abc.ABC`, subclasses will fail to instantiate
    if they do not implement all abstract methods.
    Here, we only require `raise NotImplementedError` in the
    base class, and log a warning if the method is not implemented
    in the subclass.
    """

    original_init = cls.__init__

    def find_unimplemented_methods(self: object):
        unimplemented_methods = []
        for attr_name in dir(self):
            # bypass inner method
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(self, attr_name)
                # get the func of callable method
                if callable(attr):
                    attr_func = attr.__func__
            except AttributeError:
                continue
            src = inspect.getsource(attr_func)
            if "NotImplementedError" in src:
                unimplemented_methods.append(attr_name)
        if unimplemented_methods:
            method_names = ",".join(unimplemented_methods)
            msg = f"Methods {method_names} not implemented in {self}"
            logger.debug(msg)

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        find_unimplemented_methods(self)

    type.__setattr__(cls, "__init__", wrapped_init)
    return cls


# Only relevant for models using ALiBi (e.g, MPT)
def check_use_alibi(model_config: ModelConfig) -> bool:
    cfg = model_config.hf_text_config
    return (
        getattr(cfg, "alibi", False)  # Falcon
        or (
            "BloomForCausalLM" in getattr(model_config.hf_config, "architectures", [])
        )  # Bloom
        or getattr(cfg, "position_encoding_type", "") == "alibi"  # codellm_1b_alibi
        or (
            hasattr(cfg, "attn_config")  # MPT
            and (
                (
                    isinstance(cfg.attn_config, dict)
                    and cfg.attn_config.get("alibi", False)
                )
                or (
                    not isinstance(cfg.attn_config, dict)
                    and getattr(cfg.attn_config, "alibi", False)
                )
            )
        )
    )


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len
