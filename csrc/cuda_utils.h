#pragma once

#include <torch/extension.h>

int get_device_attribute(
    int attribute,
    int device_id);
