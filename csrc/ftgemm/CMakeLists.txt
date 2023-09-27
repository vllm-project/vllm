# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(Python REQUIRED)
set(Torch_DIR "/usr/local/lib/python3.9/site-packages/torch/share/cmake/Torch/")
find_package(Torch REQUIRED)
set(pybind11_DIR "/usr/local/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)


include_directories(${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
set(CUDA_LIBRARIES "/usr/local/cuda/lib64")
set(TORCH_LIBRARIES "/usr/local/lib/python3.9/site-packages/torch/lib")
link_directories(${CUDA_LIBRARIES} ${TORCH_LIBRARIES})

add_library(cuda_utils STATIC cuda_utils.cc)
set_property(TARGET cuda_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cuda_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cuda_utils PUBLIC -lcudart)

add_library(cublasAlgoMap STATIC cublasAlgoMap.cc)
set_property(TARGET cublasAlgoMap PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasAlgoMap PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasAlgoMap PUBLIC -lcublas -lcudart -lcurand cuda_utils)

add_library(cublasMMWrapper STATIC cublasMMWrapper.cc)
set_property(TARGET cublasMMWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasMMWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasMMWrapper PUBLIC -lcublas -lcudart -lcurand cublasAlgoMap cuda_utils)

add_library(cublasINT8MMWrapper STATIC cublasINT8MMWrapper.cc)
set_property(TARGET cublasINT8MMWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasINT8MMWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasINT8MMWrapper PUBLIC -lcublasLt -lcudart -lcurand -lcublas cublasAlgoMap cublasMMWrapper cuda_utils)

add_library(transformLayout STATIC transform_layout.cu)
set_property(TARGET transformLayout PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET transformLayout PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(transformLayout PUBLIC -lcudart -lcurand -lcublas)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
pybind11_add_module(int8_gemm MODULE bindings.cpp)
target_link_libraries(int8_gemm PUBLIC -lpython3.9 -ltorch -ltorch_python -lcudart cublasINT8MMWrapper cublasAlgoMap transformLayout)





