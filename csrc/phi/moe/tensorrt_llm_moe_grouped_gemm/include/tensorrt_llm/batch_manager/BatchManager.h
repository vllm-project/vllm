/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <assert.h>
#include <chrono>
#include <iostream>
#include <list>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace tensorrt_llm::batch_manager
{
enum class BatchManagerErrorCode_t
{
    STATUS_SUCCESS = 0,
    STATUS_FAILED = 1,
    STATUS_NO_WORK = 2,
    STATUS_TERMINATE = 3
};

enum class TrtGptModelType
{
    V1,
    InflightBatching,
    InflightFusedBatching
};

} // namespace tensorrt_llm::batch_manager
