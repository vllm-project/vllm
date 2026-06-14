"""
TurboQuant CUDA Extension Setup

Builds the turboquant_kernel PyTorch extension with CUDA support.
This is typically integrated into vLLM's main build system.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension
from torch.utils import cpp_extension


def build_turboquant_extension():
    """Build TurboQuant CUDA kernels as PyTorch extension."""
    
    csrc_dir = Path(__file__).parent / "csrc"
    
    # Source files for the extension
    sources = [
        str(csrc_dir / "turboquant_bindings.cpp"),
        str(csrc_dir / "turboquant_kernels.cu"),
    ]
    
    extension = cpp_extension.CUDAExtension(
        name="turboquant_kernel",
        sources=sources,
        include_dirs=[str(csrc_dir)],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '-arch=sm_70',  # Adjust for your GPU architecture
                '-gencode', 'arch=compute_70,code=sm_70',
                '--use_fast_math',
            ],
        },
    )
    
    return extension


if __name__ == "__main__":
    setup(
        name="turboquant_kernel",
        version="1.0.0",
        description="TurboQuant CUDA kernels for efficient KV cache quantization",
        ext_modules=[build_turboquant_extension()],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
    )
