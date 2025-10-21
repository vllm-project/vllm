import builtins as _builtins
class _DType:
    def __init__(self, name: str, itemsize: int, floating: bool = False, complex_: bool = False):
        self.name = name
        self._itemsize = itemsize
        self.is_floating_point = _builtins.bool(floating)
        self.is_complex = _builtins.bool(complex_)

    def __repr__(self):
        return f"torch.dtype({self.name})"


float32 = _DType("float32", 4, floating=True)
float64 = _DType("float64", 8, floating=True)
half = _DType("float16", 2, floating=True)
bfloat16 = _DType("bfloat16", 2, floating=True)
float = float32
uint8 = _DType("uint8", 1, floating=False)
int8 = _DType("int8", 1, floating=False)
int32 = _DType("int32", 4, floating=False)
int64 = _DType("int64", 8, floating=False)
bool = _DType("bool", 1, floating=False)

# Placeholder for rarely-used dtypes in tests
float8_e4m3fn = _DType("float8_e4m3fn", 1, floating=True)


class _Tensor:
    def __init__(self, dtype=float32):
        self._dtype = dtype

    def element_size(self):
        return getattr(self._dtype, "_itemsize", 1)


def tensor(_, dtype=float32):
    return _Tensor(dtype=dtype)


def get_default_dtype():
    return float32


def set_default_dtype(_):
    return None


def get_num_threads():
    return 1


def set_num_threads(_):
    return None


class _iinfo:
    def __init__(self, dtype):
        self.min = 0
        self.max = 2 ** (8 * getattr(dtype, "_itemsize", 1)) - 1


class _finfo:
    def __init__(self, dtype):
        self.min = -1.0
        self.max = 1.0
        self.resolution = 1e-6


def iinfo(dtype):
    return _iinfo(dtype)


def finfo(dtype):
    return _finfo(dtype)


# Expose a minimal Tensor symbol
Tensor = _Tensor

__all__ = [
    "float32",
    "float64",
    "half",
    "bfloat16",
    "float",
    "uint8",
    "int8",
    "int32",
    "int64",
    "bool",
    "float8_e4m3fn",
    "tensor",
    "Tensor",
    "get_default_dtype",
    "set_default_dtype",
    "get_num_threads",
    "set_num_threads",
    "iinfo",
    "finfo",
]
