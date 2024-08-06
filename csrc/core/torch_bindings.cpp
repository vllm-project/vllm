#include <torch/library.h>

#include "scalar_type.hpp"
#include "registration.h"

// Note the CORE exstension will be built for (almost) all hardware targets so
// new additions must account for this. (currently not built for TPU and Neuron)

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, lib) {
  // ScalarType, a custom class for representing data types that supports
  // quantized types, declared here so it can be used when creating interfaces
  // for custom ops.
  vllm::ScalarTypeTorch::bind_class(lib);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
