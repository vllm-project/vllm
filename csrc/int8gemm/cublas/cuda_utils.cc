/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_utils.h"

cudaError_t getSetDevice(int i_device, int *o_device) {
  int current_dev_id = 0;
  cudaError_t err = cudaSuccess;

  if (o_device != NULL) {
    err = cudaGetDevice(&current_dev_id);
    if (err != cudaSuccess) {
      return err;
    }
    if (current_dev_id == i_device) {
      *o_device = i_device;
    } else {
      err = cudaSetDevice(i_device);
      if (err != cudaSuccess) {
        return err;
      }
      *o_device = current_dev_id;
    }
  } else {
    err = cudaSetDevice(i_device);
    if (err != cudaSuccess) {
      return err;
    }
  }

  return cudaSuccess;
}
