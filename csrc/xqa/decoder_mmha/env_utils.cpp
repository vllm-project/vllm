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

#include "env_utils.h"
#include <cstdlib>


static std::optional<int32_t> getIntEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    int32_t const val = std::atoi(env);
    if (val <= 0)
    {
        return std::nullopt;
    }
    return {val};
};

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels()
{
    static bool const forceXQA = (getIntEnv("VLLM_FORCE_XQA").value_or(0) != 0);
    return forceXQA;
}


// // Tune the number of blocks per sequence for accuracy/performance purpose.
// bool getEnvMmhaMultiblockDebug()
// {
//     static bool init = false;
//     static bool forceMmhaMaxSeqLenTile = false;
//     if (!init)
//     {
//         init = true;
//         char const* enable_mmha_debug_var = std::getenv("TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG");
//         if (enable_mmha_debug_var)
//         {
//             if (enable_mmha_debug_var[0] == '1' && enable_mmha_debug_var[1] == '\0')
//             {
//                 forceMmhaMaxSeqLenTile = true;
//             }
//         }
//     }
//     return forceMmhaMaxSeqLenTile;
// }

// int getEnvMmhaBlocksPerSequence()
// {
//     static bool init = false;
//     static int mmhaBlocksPerSequence = 0;
//     if (!init)
//     {
//         init = true;
//         char const* mmhaBlocksPerSequenceEnv = std::getenv("TRTLLM_MMHA_BLOCKS_PER_SEQUENCE");
//         if (mmhaBlocksPerSequenceEnv)
//         {
//             mmhaBlocksPerSequence = std::atoi(mmhaBlocksPerSequenceEnv);
//             if (mmhaBlocksPerSequence <= 0)
//             {
//                 TLLM_LOG_WARNING("Invalid value for TRTLLM_MMHA_BLOCKS_PER_SEQUENCE. Will use default values instead!");
//             }
//         }
//     }
//     return mmhaBlocksPerSequence;
// }

// int getEnvMmhaKernelBlockSize()
// {
//     static bool init = false;
//     static int mmhaKernelBlockSize = 0;
//     if (!init)
//     {
//         init = true;
//         char const* mmhaKernelBlockSizeEnv = std::getenv("TRTLLM_MMHA_KERNEL_BLOCK_SIZE");
//         if (mmhaKernelBlockSizeEnv)
//         {
//             mmhaKernelBlockSize = std::atoi(mmhaKernelBlockSizeEnv);
//             if (mmhaKernelBlockSize <= 0)
//             {
//                 TLLM_LOG_WARNING("Invalid value for TRTLLM_MMHA_KERNEL_BLOCK_SIZE. Will use default values instead!");
//             }
//         }
//     }
//     return mmhaKernelBlockSize;
// }

// bool getEnvEnablePDL()
// {
//     static bool init = false;
//     static bool enablePDL = false;
//     if (!init)
//     {
//         init = true;
//         // PDL only available when arch >= 90
//         if (getSMVersion() >= 90)
//         {
//             char const* enable_pdl = std::getenv("TRTLLM_ENABLE_PDL");
//             if (enable_pdl)
//             {
//                 // PDL will be enabled by setting the env variables `TRTLLM_ENABLE_PDL` to `1`
//                 if (enable_pdl[0] == '1' && enable_pdl[1] == '\0')
//                 {
//                     enablePDL = true;
//                 }
//             }
//         }
//     }
//     return enablePDL;
// }

