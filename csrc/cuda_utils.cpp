#include <torch/extension.h>

int get_device_attribute(
    int attribute,
    int device_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "get_device_attribute",
    &get_device_attribute,
    "Gets the specified device attribute.");
}

