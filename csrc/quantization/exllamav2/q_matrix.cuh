/*
 * Adapted from https://github.com/turboderp/exllamav2
 * Copyright (c) 2024 turboderp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef _q_matrix_cuh
#define _q_matrix_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace vllm {
namespace exl2 {

#define MAX_SUPERGROUPS 16

class QMatrix {
 public:
  int device;
  bool is_gptq;

  int height;
  int width;
  int groups;
  int gptq_groupsize;

  int rows_8;
  int rows_6;
  int rows_5;
  int rows_4;
  int rows_3;
  int rows_2;

  uint32_t* cuda_q_weight = NULL;
  uint16_t* cuda_q_perm = NULL;
  uint16_t* cuda_q_invperm = NULL;
  uint32_t* cuda_q_scale = NULL;
  half* cuda_q_scale_max = NULL;
  uint16_t* cuda_q_groups = NULL;
  uint16_t* cuda_q_group_map = NULL;
  uint32_t* cuda_gptq_qzeros = NULL;
  half* cuda_gptq_scales = NULL;

  half* temp_dq;

  bool failed;

  QMatrix(const int _device, const int _height, const int _width,
          const int _groups,

          uint32_t* _q_weight, uint16_t* _q_perm, uint16_t* _q_invperm,
          uint32_t* _q_scale, half* _q_scale_max, uint16_t* _q_groups,
          uint16_t* _q_group_map);

  ~QMatrix();

  void reconstruct(half* out);
  bool make_sequential(const uint32_t* cpu_g_idx);

 private:
};

}  // namespace exl2
}  // namespace vllm

#endif