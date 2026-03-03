/**
 * binding.cpp
 *
 * Single entry-point for the cuda_extension Python module.
 * All kernel bindings are registered automatically by the REGISTER_BINDING
 * macro in each kernel's *_binding.cpp file — do not add bindings here.
 *
 * Do NOT modify this file.
 */

#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(cuda_extension, m) {
    m.doc() = "Custom CUDA extension for CUDA-Agent optimized kernels.";
    BindingRegistry::getInstance().applyBindings(m);
}
