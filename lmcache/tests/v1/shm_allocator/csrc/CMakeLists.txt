cmake_minimum_required(VERSION 3.10)
project(shm_file_worker_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(shm_file_worker shm_file_worker.cpp)
if(NOT APPLE)
    target_link_libraries(shm_file_worker PRIVATE rt)
endif()
