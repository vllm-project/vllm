/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once
#include <cstdint>
#include <optional>


// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels();

// Whether XQA JIT is enabled.
//
// Returns the value of TRTLLM_ENABLE_XQA_JIT env var. If such env var doesn't exist, std::nullopt is returned.
// std::optional<bool> getEnvEnableXQAJIT();

// Tune the number of blocks per sequence for accuracy/performance purpose.
// bool getEnvMmhaMultiblockDebug();

// int getEnvMmhaBlocksPerSequence();

// int getEnvMmhaKernelBlockSize();

// Whether PDL is enabled.
bool getEnvEnablePDL();
