# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Optional, Union
import struct

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import parse_cache_key
from lmcache.v1.memory_management import MemoryFormat

if TYPE_CHECKING:
    # First Party
    from lmcache.utils import CacheEngineKey, LayerCacheEngineKey

logger = init_logger(__name__)


MAX_KEY_LENGTH = 150
REMOTE_METADATA_FMT: Optional[str] = None
REMOTE_METADATA_BYTES: Optional[int] = None


class ClientCommand(IntEnum):
    PUT = auto()
    GET = auto()
    EXIST = auto()
    LIST = auto()
    HEALTH = auto()


class ServerReturnCode(IntEnum):
    SUCCESS = 200
    FAIL = 400


DTYPE_TO_INT = {
    None: 0,
    torch.half: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float: 4,
    torch.float32: 4,
    torch.float64: 5,
    torch.double: 5,
    torch.uint8: 6,
    torch.float8_e4m3fn: 7,
    torch.float8_e5m2: 8,
}

INT_TO_DTYPE = {
    0: None,
    1: torch.half,
    2: torch.float16,
    3: torch.bfloat16,
    4: torch.float,
    5: torch.float64,
    6: torch.uint8,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}

# TODO (Jiayi): Add more backends
LOCATION_TO_INT = {
    None: 0,
    "LocalCPUBackend": 1,
    "LocalDiskBackend": 2,
}

INT_TO_LOCATION = {
    0: None,
    1: "LocalCPUBackend",
    2: "LocalDiskBackend",
}


def init_remote_metadata_info(num_groups: int):
    global REMOTE_METADATA_FMT
    global REMOTE_METADATA_BYTES
    # length, fmt, (dtype, shape0, shape1, shape2, shape3) * num_groups
    fmt_length = 2 + 5 * num_groups
    REMOTE_METADATA_FMT = "i" * fmt_length
    REMOTE_METADATA_BYTES = 4 * fmt_length
    logger.info(
        "init remote metadata info with groups: %s, "
        "remote metadata fmt: %s, remote metadata bytes: %s",
        num_groups,
        REMOTE_METADATA_FMT,
        REMOTE_METADATA_BYTES,
    )


def get_remote_metadata_bytes():
    global REMOTE_METADATA_BYTES
    assert REMOTE_METADATA_BYTES is not None
    return REMOTE_METADATA_BYTES


def pad_shape_to_4d(shape: torch.Size) -> list[int]:
    """Pad a shape with fewer than 4 dimensions to 4D using trailing
    zeros.

    Shapes that are already 4D are returned as-is.  For shapes with
    fewer dimensions the missing trailing slots are filled with ``0``.
    This is consistent with the convention used by
    :class:`BinaryMemoryObj` (``[length, 0, 0, 0]``).

    Args:
        shape: The original tensor shape (1-D to 4-D).

    Returns:
        A list of exactly 4 integers representing the padded shape.

    Raises:
        AssertionError: If the shape has more than 4 dimensions.
    """
    assert len(shape) <= 4, (
        f"Shape dimension must be <= 4 for serialization, got {len(shape)}"
    )
    if len(shape) == 4:
        return list(shape)

    padded = list(shape) + [0] * (4 - len(shape))
    return padded


def strip_shape_padding(
    dims: list[int],
    fmt: Optional[MemoryFormat] = MemoryFormat.UNDEFINED,
) -> torch.Size:
    """Strip trailing-zero padding that was added by
    :func:`pad_shape_to_4d`.

    Trailing zeros are removed so that the original dimensionality is
    restored.  At least one dimension is always preserved.

    For ``BINARY`` and ``BINARY_BUFFER`` formats, the shape is returned
    as-is because these formats inherently use 4-D shapes with zero
    padding (e.g., ``[length, 0, 0, 0]``).

    Args:
        dims: A list of 4 integers read from the serialized format.
        fmt: The memory format of the serialized data.

    Returns:
        A :class:`torch.Size` with the padding removed.
    """
    if fmt in (MemoryFormat.BINARY, MemoryFormat.BINARY_BUFFER):
        # These formats use 4D shapes with legitimate zero dimensions.
        # Skip stripping to preserve the original shape.
        return torch.Size(dims)

    end = len(dims)
    while end > 1 and dims[end - 1] == 0:
        end -= 1
    return torch.Size(dims[:end])


@dataclass
class RemoteMetadata:
    length: int
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    fmt: MemoryFormat

    def _prepare_params(self):
        params = [self.length, int(self.fmt.value)]
        for shape, dtype in zip(self.shapes, self.dtypes, strict=True):
            padded = pad_shape_to_4d(shape)
            params.append(DTYPE_TO_INT[dtype])
            params.extend(padded)
        return params

    def serialize_into(self, buffer):
        assert REMOTE_METADATA_FMT is not None
        params = self._prepare_params()
        struct.pack_into(REMOTE_METADATA_FMT, buffer, 0, *params)

    def serialize(self) -> bytes:
        assert REMOTE_METADATA_FMT is not None
        params = self._prepare_params()
        packed_bytes = struct.pack(REMOTE_METADATA_FMT, *params)
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "RemoteMetadata":
        assert REMOTE_METADATA_FMT is not None
        # length, fmt, (dtype, shape0, shape1, shape2, shape3) * num_groups
        result = struct.unpack_from(REMOTE_METADATA_FMT, s)
        length = result[0]
        memory_fmt = MemoryFormat(result[1])
        shapes = []
        dtypes = []
        for i in range(2, len(result), 5):
            dims = list(result[i + 1 : i + 5])
            shapes.append(strip_shape_padding(dims, memory_fmt))
            dtypes.append(INT_TO_DTYPE[result[i]])

        return RemoteMetadata(
            length,
            shapes,
            dtypes,
            memory_fmt,
        )


# TODO(Jiayi): Server and client message can be merged into one.


@dataclass
class ClientMetaMessage:
    """
    Request message from LMCache workers or servers.
    """

    command: ClientCommand
    key: Union["CacheEngineKey", "LayerCacheEngineKey"]
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        key_str = self.key.to_string()
        assert len(key_str) <= MAX_KEY_LENGTH, (
            f"Key length {len(key_str)} exceeds maximum {MAX_KEY_LENGTH}"
        )

        # NOTE(Jiayi): 4 is the maximum dimension of memory object.
        # Pass in shape [x, 0, 0, 0] if it is a bytes memory object
        padded = pad_shape_to_4d(self.shape)

        packed_bytes = struct.pack(
            f"iiiiiiiii{MAX_KEY_LENGTH}s",
            self.command.value,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            LOCATION_TO_INT[self.location],
            padded[0],
            padded[1],
            padded[2],
            padded[3],
            key_str.encode().ljust(MAX_KEY_LENGTH),
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "ClientMetaMessage":
        command, length, fmt, dtype, location, shape0, shape1, shape2, shape3, key = (
            struct.unpack(f"iiiiiiiii{MAX_KEY_LENGTH}s", s)
        )
        shape = strip_shape_padding([shape0, shape1, shape2, shape3], MemoryFormat(fmt))
        return ClientMetaMessage(
            ClientCommand(command),
            parse_cache_key(key.decode().strip()),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            shape,
            INT_TO_LOCATION[location],
        )

    @staticmethod
    def packlength() -> int:
        # NOTE: 9 is the number of integers
        return 4 * 9 + MAX_KEY_LENGTH


@dataclass
class ServerMetaMessage:
    """
    Reply message from LMCache workers or servers.
    """

    code: ServerReturnCode
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        padded = pad_shape_to_4d(self.shape)
        packed_bytes = struct.pack(
            "iiiiiiiii",
            self.code.value,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            padded[0],
            padded[1],
            padded[2],
            padded[3],
            LOCATION_TO_INT[self.location],
        )
        return packed_bytes

    @staticmethod
    def packlength() -> int:
        return 4 * 9

    @staticmethod
    def deserialize(s: bytes) -> "ServerMetaMessage":
        code, length, fmt, dtype, shape0, shape1, shape2, shape3, location = (
            struct.unpack("iiiiiiiii", s)
        )
        shape = strip_shape_padding([shape0, shape1, shape2, shape3], MemoryFormat(fmt))
        return ServerMetaMessage(
            ServerReturnCode(code),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            shape,
            INT_TO_LOCATION[location],
        )
