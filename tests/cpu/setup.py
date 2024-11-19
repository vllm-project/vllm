import site

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# TODO(shawnd200) integrate ISAs and find out a better build and install
torch_path = site.getsitepackages()[0] + "/torch"
ext_modules = [
    Pybind11Extension("cputypes",
                      sources=["cpu_types_py_bindings.cpp"],
                      language="c++",
                      libraries=["torch"],
                      include_dirs=[
                          f"{torch_path}/include/torch/csrc/api/include",
                          f"{torch_path}/include"
                      ],
                      library_dirs=[f"{torch_path}/lib"],
                      extra_compile_args=["-std=c++17"])
]

setup(
    name="cputypes",
    version="0.1",
    description="vLLM cpu types interface for test",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
