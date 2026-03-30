"""Minimal setup.py for the vllm_xpu_kernels stub package.

This installs a pure-Python replacement for vllm_xpu_kernels that registers
torch.library fallback implementations for all ops needed by vLLM on XPU
simulators (JGS/CRI).  It replaces the compiled C++/SYCL wheel
(vllm_xpu_kernels-0.1.3) which cannot run on simulators.

Usage:
    pip install --force-reinstall /path/to/vllm_xpu_kernels_stub
"""

from setuptools import setup, find_packages

setup(
    name="vllm_xpu_kernels",
    version="0.0.1.dev0",
    description="Pure-Python vllm_xpu_kernels stub for XPU simulators",
    packages=["vllm_xpu_kernels"],
    package_dir={"vllm_xpu_kernels": "."},
    python_requires=">=3.10",
    install_requires=["torch"],
)
