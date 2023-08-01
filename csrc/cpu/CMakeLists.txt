
# find_package(Torch REQUIRED)
# message("Torch CXX FLAGS: ${TORCH_CXX_FLAGS}")
# message("Torch LIBS: ${TORCH_LIBRARIES}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set(USE_STATIC_MKL OFF)
# find_package(MKL REQUIRED)
# message("MKL found: ${MKL_FOUND}, 
#         version: ${MKL_VERSION}, 
#         include dir: ${MKL_INCLUDE_DIR}, 
#         libs: ${MKL_LIBRARIES},
#         omp type: ${MKL_OPENMP_TYPE},
#         omp path: ${MKL_OPENMP_LIBRARY}")

macro(add_operator_target op_name source_file)
add_library(${op_name} SHARED ${source_file})
# target_link_libraries(${op_name} ${TORCH_LIBRARIES} ${MKL_LIBRARIES})
endmacro(add_operator_target)

add_operator_target(layernorm layernorm_impl.cpp)
add_operator_target(pos_encoding pos_encoding_impl.cpp)
add_operator_target(cache_op "cache_impl.cpp")
add_operator_target(activation activation_impl.cpp)
add_operator_target(attention attention_impl.cpp)
