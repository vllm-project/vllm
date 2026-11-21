# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def bf16_conversion(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    directory = tmp_path_factory.mktemp("bf16_conversion")
    source = directory / "conversion.cpp"
    library = directory / "conversion.so"
    source.write_text(
        '#include "float_convert.hpp"\n'
        'extern "C" uint16_t convert(uint32_t bits) {\n'
        "  return float_to_bf16(std::bit_cast<float>(bits));\n"
        "}\n"
    )
    include = Path(__file__).resolve().parents[3] / "csrc" / "cpu"
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++20",
            "-shared",
            "-fPIC",
            f"-I{include}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = ctypes.CDLL(str(library))
    result.convert.argtypes = [ctypes.c_uint32]
    result.convert.restype = ctypes.c_uint16
    return result


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        (0x00000000, 0x0000),
        (0x80000000, 0x8000),
        (0x7F800000, 0x7F80),
        (0xFF800000, 0xFF80),
        (0x7F800001, 0x7FC0),
        (0xFFFFFFFF, 0x7FC0),
        (0x7FC00000, 0x7FC0),
        (0x3F807FFF, 0x3F80),
        (0x3F808000, 0x3F80),
        (0x3F808001, 0x3F81),
        (0x3F818000, 0x3F82),
        (0xBF808000, 0xBF80),
        (0xBF818000, 0xBF82),
        (0x00008000, 0x0000),
        (0x00008001, 0x0001),
        (0x00018000, 0x0002),
        (0x7F7F8000, 0x7F80),
        (0xFF7F8000, 0xFF80),
    ],
)
def test_software_bf16_rounding_and_special_values(
    bf16_conversion: ctypes.CDLL, bits: int, expected: int
) -> None:
    assert bf16_conversion.convert(bits) == expected
