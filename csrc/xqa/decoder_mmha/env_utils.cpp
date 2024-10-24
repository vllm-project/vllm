/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "env_utils.h"
#include <cstdlib>

static std::optional<int32_t> getIntEnv(char const* name) {
  char const* const env = std::getenv(name);
  if (env == nullptr) {
    return std::nullopt;
  }
  int32_t const val = std::atoi(env);
  if (val <= 0) {
    return std::nullopt;
  }
  return {val};
};

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels() {
  static bool const forceXQA = (getIntEnv("VLLM_FORCE_XQA").value_or(0) != 0);
  return forceXQA;
}