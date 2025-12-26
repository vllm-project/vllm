cmake_minimum_required(VERSION 3.20)
project(gpu_poold LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)        # CUDA Driver API
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)               # for at::from_blob wrapper

add_library(vmm_pool SHARED
  vmm_pool_client.cc
  uds_rpc.cc
)
target_link_libraries(vmm_pool PRIVATE CUDA::cuda_driver)
target_include_directories(vmm_pool PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

add_executable(gpu_poold
  gpu_poold.cc
  uds_rpc.cc
)
target_link_libraries(gpu_poold PRIVATE CUDA::cuda_driver)

pybind11_add_module(vmm_pool_py MODULE
  vmm_pool_pybind.cc
  vmm_pool_client.cc
  uds_rpc.cc
)
target_link_libraries(vmm_pool_py
  PRIVATE CUDA::cuda_driver ${TORCH_LIBRARIES})
target_compile_definitions(vmm_pool_py PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)
set_target_properties(vmm_pool_py PROPERTIES OUTPUT_NAME "vmm_pool_py")
