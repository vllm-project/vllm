
from functools import lru_cache

@lru_cache(maxsize=None)
def is_hip() -> bool:
    return torch.version.hip is not None


@lru_cache(maxsize=None)
def is_cpu() -> bool:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return "cpu" in version("vllm")
    except PackageNotFoundError:
        return False


@lru_cache(maxsize=None)
def is_openvino() -> bool:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return "openvino" in version("vllm")
    except PackageNotFoundError:
        return False


@lru_cache(maxsize=None)
def is_neuron() -> bool:
    try:
        import transformers_neuronx
    except ImportError:
        transformers_neuronx = None
    return transformers_neuronx is not None


@lru_cache(maxsize=None)
def is_hpu() -> bool:
    from importlib import util
    return util.find_spec('habana_frameworks') is not None


@lru_cache(maxsize=None)
def is_tpu() -> bool:
    try:
        import libtpu
    except ImportError:
        libtpu = None
    return libtpu is not None


@lru_cache(maxsize=None)
def is_xpu() -> bool:
    from importlib.metadata import version
    is_xpu_flag = "xpu" in version("vllm")
    # vllm is not build with xpu
    if not is_xpu_flag:
        return False
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        _import_ipex = True
    except ImportError as e:
        logger.warning("Import Error for IPEX: %s", e.msg)
        _import_ipex = False
    # ipex dependency is not ready
    if not _import_ipex:
        logger.warning("not found ipex lib")
        return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()

if is_xpu():
    from ._ipex_ops import *
elif is_hpu():
    from ._hpu_ops import *
else:
    from ._cuda_ops import *