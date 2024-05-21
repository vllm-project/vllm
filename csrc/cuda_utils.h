#pragma once

#include <torch/extension.h>

int get_device_attribute(
    int attribute,
    int device_id);

int get_max_shared_memory_per_block_device_attribute(
    int device_id);
