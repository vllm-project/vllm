/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Reference implementation for convolution in device-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/functional.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"

namespace cutlass {
namespace reference {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

////////////////////////////////////////////////////////////////////////////////////////////////////
///                                   Conv2d device reference kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

// Conv2d Fprop kernel - y = fprop(x, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>,
  int kThreadM = 2,       // shape of a thread's tile in the GEMM M dimension
  int kThreadN = 4,       // shape of a thread's tile in the GEMM N dimension
  int kCtaShapeM = 16,    // shape of a threadblock in units of threads
  int kCtaShapeN = 8      // shape of a threadblock in units of threads
>
__global__ void Conv2dFprop(
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_x,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_y_in,
  TensorRef<ElementC, LayoutC> tensor_y_out,
  ElementCompute alpha,
  ElementCompute beta
  ) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  ElementAccumulator element_A[kThreadM];
  ElementAccumulator element_B[kThreadN];
  ElementAccumulator accum[kThreadM][kThreadN];

  int64_t npq_start = int64_t(blockIdx.x) * kCtaShapeM * kThreadM + threadIdx.x * kThreadM;
  int k_start = blockIdx.y * kCtaShapeN * kThreadN + threadIdx.y * kThreadN;

  int thread_n[kThreadM];
  int thread_p[kThreadM];
  int thread_q[kThreadM];

  // Compute N, P, Q coordinates for each row of a thread's tile
  int64_t PQ = int64_t(problem_size.P) * problem_size.Q;

  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {

    int64_t npq = npq_start + m;

    thread_n[m] = int(npq / PQ);
    
    int64_t residual = npq % PQ;
    thread_p[m] = int(residual / problem_size.Q);
    thread_q[m] = int(residual % problem_size.Q);
  }

  // Clear accumulators
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kThreadN; ++n) {
      accum[m][n] = ElementAccumulator();
    }
  }

  int c_per_group = problem_size.C / problem_size.groups;
  int k_per_group = problem_size.K / problem_size.groups;

  // Compute convolution
  for (int R = 0; R < problem_size.R; ++R) {
    for (int S = 0; S < problem_size.S; ++S) {
      for (int C = 0; C < problem_size.C; ++C) {

        // Get group id of currnet channel
        int c_group_idx = C / c_per_group;

        // Load from activations tensor
        int filter_r = R;
        int filter_s = S;   

        if (problem_size.mode == cutlass::conv::Mode::kConvolution) {
          filter_r = problem_size.R - 1 - R;
          filter_s = problem_size.S - 1 - S;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kThreadM; ++m) {
          int h = thread_p[m] * problem_size.stride_h - problem_size.pad_h + filter_r * problem_size.dilation_h;
          int w = thread_q[m] * problem_size.stride_w - problem_size.pad_w + filter_s * problem_size.dilation_w;

          if (thread_n[m] < problem_size.N && h >= 0 && h < problem_size.H && w >= 0 && w < problem_size.W) {
            element_A[m] = ElementAccumulator(tensor_x.at({thread_n[m], h, w, C}));
          }
          else {
            element_A[m] = ElementAccumulator();
          }
        }

        // Load from filters tensor
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < kThreadN; ++n) {
          int thread_k = k_start + n;
          int k_group_idx = thread_k / k_per_group;

          if (thread_k < problem_size.K && k_group_idx == c_group_idx) {
            element_B[n] = ElementAccumulator(tensor_w.at({thread_k, R, S, C % c_per_group}));
          }
          else {
            element_B[n] = ElementAccumulator();
          }
        }

        // Accumulate matrix product
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kThreadM; ++m) {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < kThreadN; ++n) {
            accum[m][n] = inner_product_op(element_A[m], element_B[n], accum[m][n]);
          }
        }
      }
    }
  }

  // Write out the results
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    if (thread_n[m] < problem_size.N && thread_p[m] < problem_size.P && thread_q[m] < problem_size.Q) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kThreadN; ++n) {
        int thread_k = k_start + n;
        if (thread_k < problem_size.K) {

          ElementCompute c_ref = ElementCompute();
          if (beta != ElementCompute()) {
            c_ref = ElementCompute(tensor_y_in.at({thread_n[m], thread_p[m], thread_q[m], thread_k}));
          }

          tensor_y_out.at({thread_n[m], thread_p[m], thread_q[m], thread_k}) = convert_op(
            alpha * ElementCompute(accum[m][n]) + beta * c_ref);
        }
      } 
    }
  }
}

// Conv3d Fprop kernel - y = fprop(x, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator =  ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>,
  int kThreadM = 2,       // shape of a thread's tile in the GEMM M dimension
  int kThreadN = 4,       // shape of a thread's tile in the GEMM N dimension
  int kCtaShapeM = 16,    // shape of a threadblock in units of threads
  int kCtaShapeN = 8      // shape of a threadblock in units of threads
>
__global__ void Conv3dFprop(
  conv::Conv3dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_x,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_y_in,
  TensorRef<ElementC, LayoutC> tensor_y_out,
  ElementCompute alpha,
  ElementCompute beta
  ) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  ElementAccumulator element_A[kThreadM];
  ElementAccumulator element_B[kThreadN];
  ElementAccumulator accum[kThreadM][kThreadN];

  int64_t nzpq_start = int64_t(blockIdx.x) * kCtaShapeM * kThreadM + threadIdx.x * kThreadM;
  int k_start = blockIdx.y * kCtaShapeN * kThreadN + threadIdx.y * kThreadN;

  int thread_n[kThreadM];
  int thread_z[kThreadM];
  int thread_p[kThreadM];
  int thread_q[kThreadM];

  // Compute N, Z, P, Q coordinates for each row of a thread's tile
  int64_t PQ = int64_t(problem_size.P) * problem_size.Q;
  int64_t ZPQ = PQ * problem_size.Z;

  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {

    int64_t nzpq = nzpq_start + m;

    thread_n[m] = int(nzpq / ZPQ);
    
    int64_t residual = nzpq % ZPQ;
    thread_z[m] = int(residual / PQ);

    residual = residual % PQ;
    thread_p[m] = int(residual / problem_size.Q);
    thread_q[m] = int(residual % problem_size.Q);
  }

  // Clear accumulators
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kThreadN; ++n) {
      accum[m][n] = ElementAccumulator();
    }
  }

  // Compute convolution
  for (int T = 0; T < problem_size.T; ++T) {
    for (int R = 0; R < problem_size.R; ++R) {
      for (int S = 0; S < problem_size.S; ++S) {
        for (int C = 0; C < problem_size.C; ++C) {

          // Load from activations tensor
          int filter_t = T;
          int filter_r = R;
          int filter_s = S;   

          if (problem_size.mode == cutlass::conv::Mode::kConvolution) {
            filter_t = problem_size.T - 1 - T;
            filter_r = problem_size.R - 1 - R;
            filter_s = problem_size.S - 1 - S;
          }

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < kThreadM; ++m) {
            int d = thread_z[m] * problem_size.stride_d - problem_size.pad_d + filter_t * problem_size.dilation_d;
            int h = thread_p[m] * problem_size.stride_h - problem_size.pad_h + filter_r * problem_size.dilation_h;
            int w = thread_q[m] * problem_size.stride_w - problem_size.pad_w + filter_s * problem_size.dilation_w;

            if (thread_n[m] < problem_size.N && 
              d >= 0 && d < problem_size.D && 
              h >= 0 && h < problem_size.H && 
              w >= 0 && w < problem_size.W) {

              element_A[m] = ElementAccumulator(tensor_x.at({thread_n[m], d, h, w, C}));
            }
            else {
              element_A[m] = ElementAccumulator();
            }
          }

          // Load from filters tensor
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < kThreadN; ++n) {
            int thread_k = k_start + n;

            if (thread_k < problem_size.K) {
              element_B[n] = ElementAccumulator(tensor_w.at({thread_k, T, R, S, C}));
            }
            else {
              element_B[n] = ElementAccumulator();
            }
          }

          // Accumulate matrix product
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < kThreadM; ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < kThreadN; ++n) {
              accum[m][n] = inner_product_op(element_A[m], element_B[n], accum[m][n]);
            }
          }

        } // for (C)
      } // for (S)
    }  // for (R) 
  } // for (T)

  // Write out the results
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {

    if (thread_n[m] < problem_size.N && 
      thread_z[m] < problem_size.Z && 
      thread_p[m] < problem_size.P && 
      thread_q[m] < problem_size.Q) {

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kThreadN; ++n) {
        int thread_k = k_start + n;
        if (thread_k < problem_size.K) {

          ElementCompute c_ref = ElementCompute();
          if (beta != ElementCompute()) {
            c_ref = ElementCompute(tensor_y_in.at({thread_n[m], thread_z[m], thread_p[m], thread_q[m], thread_k}));
          }

          tensor_y_out.at({thread_n[m], thread_z[m], thread_p[m], thread_q[m], thread_k}) = convert_op(
            alpha * ElementCompute(accum[m][n]) + beta * c_ref);
        }
      } // for (n)
 
    }
  } // for (m)
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Conv2d dgrad kernel - dx = dgrad(dy, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>,
  int kThreadM = 2,       // shape of a thread's tile in the GEMM M dimension
  int kThreadN = 4,       // shape of a thread's tile in the GEMM N dimension
  int kCtaShapeM = 16,    // shape of a threadblock in units of threads
  int kCtaShapeN = 8      // shape of a threadblock in units of threads
>
__global__ void Conv2dDgrad(
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_dx_in,
  TensorRef<ElementC, LayoutC> tensor_dx_out,
  ElementCompute alpha,
  ElementCompute beta
  ) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  ElementAccumulator element_A[kThreadM];
  ElementAccumulator element_B[kThreadN];
  ElementAccumulator accum[kThreadM][kThreadN];

  int64_t nhw_start = int64_t(blockIdx.x) * kCtaShapeM * kThreadM + threadIdx.x * kThreadM;
  int c_start = blockIdx.y * kCtaShapeN * kThreadN + threadIdx.y * kThreadN;

  int thread_n[kThreadM];
  int thread_h[kThreadM];
  int thread_w[kThreadM];

  // Compute N, H, W coordinates for each row of a thread's tile
  int64_t HW = int64_t(problem_size.H) * problem_size.W;

  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {

    int64_t nhw = nhw_start + m;

    thread_n[m] = int(nhw / HW);
    
    int64_t residual = nhw % HW;
    thread_h[m] = int(residual / problem_size.W);
    thread_w[m] = int(residual % problem_size.W);
  }

  // Clear accumulators
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kThreadN; ++n) {
      accum[m][n] = ElementAccumulator();
    }
  }

  // Compute convolution
  for (int R = 0; R < problem_size.R; ++R) {
    for (int S = 0; S < problem_size.S; ++S) {
      for (int K = 0; K < problem_size.K; ++K) {

        // Load from activations tensor
        int filter_r = R;
        int filter_s = S;   

        if (problem_size.mode == cutlass::conv::Mode::kConvolution) {
          filter_r = problem_size.R - 1 - R;
          filter_s = problem_size.S - 1 - S;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kThreadM; ++m) {

          int p = thread_h[m] + problem_size.pad_h - filter_r * problem_size.dilation_h;
          int q = thread_w[m] + problem_size.pad_w - filter_s * problem_size.dilation_w;

          element_A[m] = ElementAccumulator();

          if (p >= 0 && !(p % problem_size.stride_h) && q >= 0 && !(q % problem_size.stride_w)) {

            p = p / problem_size.stride_h;
            q = q / problem_size.stride_w;

            if (thread_n[m] < problem_size.N && p < problem_size.P && q < problem_size.Q) {
              element_A[m] = ElementAccumulator(tensor_dy.at({thread_n[m], p, q, K}));  
            }
          }
        }

        // Load from filters tensor
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < kThreadN; ++n) {
          int thread_c = c_start + n;

          if (thread_c < problem_size.C) {
            element_B[n] = ElementAccumulator(tensor_w.at({K, R, S, thread_c}));
          }
          else {
            element_B[n] = ElementAccumulator();
          }
        }

        // Accumulate matrix product
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kThreadM; ++m) {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < kThreadN; ++n) {
            accum[m][n] = inner_product_op(element_A[m], element_B[n], accum[m][n]);
          }
        }
      }
    }
  }

  // Write out the results
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    
    if (thread_n[m] < problem_size.N && thread_h[m] < problem_size.H && thread_w[m] < problem_size.W) {
      
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kThreadN; ++n) {
        int thread_c = c_start + n;
        if (thread_c < problem_size.C) {

          ElementCompute c_ref = ElementCompute();
          if (beta != ElementCompute()) {
            c_ref = ElementCompute(tensor_dx_in.at({thread_n[m], thread_h[m], thread_w[m], thread_c}));
          }

          tensor_dx_out.at({thread_n[m], thread_h[m], thread_w[m], thread_c}) = convert_op(
            alpha * ElementCompute(accum[m][n]) + beta * c_ref);
        }
      } 
    }
  }
}

// Conv3d dgrad kernel - dx = dgrad(dy, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>,
  int kThreadM = 2,       // shape of a thread's tile in the GEMM M dimension
  int kThreadN = 4,       // shape of a thread's tile in the GEMM N dimension
  int kCtaShapeM = 16,    // shape of a threadblock in units of threads
  int kCtaShapeN = 8      // shape of a threadblock in units of threads
>
__global__ void Conv3dDgrad(
  conv::Conv3dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_dx_in,
  TensorRef<ElementC, LayoutC> tensor_dx_out,
  ElementCompute alpha,
  ElementCompute beta
  ) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  ElementAccumulator element_A[kThreadM];
  ElementAccumulator element_B[kThreadN];
  ElementAccumulator accum[kThreadM][kThreadN];

  int64_t ndhw_start = int64_t(blockIdx.x) * kCtaShapeM * kThreadM + threadIdx.x * kThreadM;
  int c_start = blockIdx.y * kCtaShapeN * kThreadN + threadIdx.y * kThreadN;

  int thread_n[kThreadM];
  int thread_d[kThreadM];
  int thread_h[kThreadM];
  int thread_w[kThreadM];

  // Compute N, H, W coordinates for each row of a thread's tile
  int64_t HW = int64_t(problem_size.H) * problem_size.W;
  int64_t DHW = HW * problem_size.D;

  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {

    int64_t ndhw = ndhw_start + m;

    thread_n[m] = int(ndhw / DHW);
    
    int64_t residual = ndhw % DHW;
    thread_d[m] = int(residual / HW);

    residual = residual % HW;
    thread_h[m] = int(residual / problem_size.W);
    thread_w[m] = int(residual % problem_size.W);
  }

  // Clear accumulators
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kThreadN; ++n) {
      accum[m][n] = ElementAccumulator();
    }
  }

  // Compute convolution
  for (int T = 0; T < problem_size.T; ++T) {
    for (int R = 0; R < problem_size.R; ++R) {
      for (int S = 0; S < problem_size.S; ++S) {
        for (int K = 0; K < problem_size.K; ++K) {

          // Load from activations tensor
          int filter_t = T;
          int filter_r = R;
          int filter_s = S;   

          if (problem_size.mode == cutlass::conv::Mode::kConvolution) {
            filter_t = problem_size.T - 1 - T;
            filter_r = problem_size.R - 1 - R;
            filter_s = problem_size.S - 1 - S;
          }

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < kThreadM; ++m) {

            int z = thread_d[m] + problem_size.pad_d - filter_t * problem_size.dilation_d;
            int p = thread_h[m] + problem_size.pad_h - filter_r * problem_size.dilation_h;
            int q = thread_w[m] + problem_size.pad_w - filter_s * problem_size.dilation_w;

            element_A[m] = ElementAccumulator();

            if (z >= 0 && !(z % problem_size.stride_d) && 
              p >= 0 && !(p % problem_size.stride_h) && 
              q >= 0 && !(q % problem_size.stride_w)) {

              z = z / problem_size.stride_d;
              p = p / problem_size.stride_h;
              q = q / problem_size.stride_w;

              if (thread_n[m] < problem_size.N && z < problem_size.Z && p < problem_size.P && q < problem_size.Q) {
                element_A[m] = ElementAccumulator(tensor_dy.at({thread_n[m], z, p, q, K}));  
              }
            }
          }

          // Load from filters tensor
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < kThreadN; ++n) {
            int thread_c = c_start + n;

            if (thread_c < problem_size.C) {
              element_B[n] = ElementAccumulator(tensor_w.at({K, T, R, S, thread_c}));
            }
            else {
              element_B[n] = ElementAccumulator();
            }
          }

          // Accumulate matrix product
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < kThreadM; ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < kThreadN; ++n) {
              accum[m][n] = inner_product_op(element_A[m], element_B[n], accum[m][n]);
            }
          }

        } // for (C)
      } // for (S)
    } // for (R)
  } // for (T)

  // Write out the results
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    
    if (thread_n[m] < problem_size.N && 
      thread_d[m] < problem_size.D && 
      thread_h[m] < problem_size.H && 
      thread_w[m] < problem_size.W) {
      
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kThreadN; ++n) {
        int thread_c = c_start + n;
        if (thread_c < problem_size.C) {

          ElementCompute c_ref = ElementCompute();
          if (beta != ElementCompute()) {
            c_ref = ElementCompute(tensor_dx_in.at({thread_n[m], thread_d[m], thread_h[m], thread_w[m], thread_c}));
          }

          tensor_dx_out.at({thread_n[m], thread_d[m], thread_h[m], thread_w[m], thread_c}) = convert_op(
            alpha * ElementCompute(accum[m][n]) + beta * c_ref);
        }
      } 
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Conv2d wgrad kernel - dw = wgrad(dy, x)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>,
  int kThreadM = 2,       // shape of a thread's tile in the GEMM M dimension
  int kThreadN = 4,       // shape of a thread's tile in the GEMM N dimension
  int kCtaShapeM = 8,     // shape of a threadblock in units of threads
  int kCtaShapeN = 16     // shape of a threadblock in units of threads
>
__global__ void Conv2dWgrad(
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_x,
  TensorRef<ElementC, LayoutC> tensor_dw_in,
  TensorRef<ElementC, LayoutC> tensor_dw_out,
  ElementCompute alpha,
  ElementCompute beta
  ) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  ElementAccumulator element_A[kThreadM];
  ElementAccumulator element_B[kThreadN];
  ElementAccumulator accum[kThreadM][kThreadN];

  int k_start = blockIdx.x * kCtaShapeM * kThreadM + threadIdx.x * kThreadM;
  int64_t rsc_start = int64_t(blockIdx.y) * kCtaShapeN * kThreadN + threadIdx.y * kThreadN;
  
  int thread_r[kThreadN];
  int thread_s[kThreadN];
  int thread_c[kThreadN];

  // Compute R, S, C coordinates for each row of a thread's tile
  int64_t SC = int64_t(problem_size.S) * problem_size.C;

  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < kThreadN; ++n) {

    int64_t rsc = rsc_start + n;
    int64_t residual = rsc % SC;

    thread_r[n] = int(rsc / SC);
    thread_s[n] = int(residual / problem_size.C);
    thread_c[n] = int(residual % problem_size.C);
  }

  // Clear accumulators
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kThreadN; ++n) {
      accum[m][n] = ElementAccumulator();
    }
  }

  // Compute convolution
  for (int N = 0; N < problem_size.N; ++N) {
    for (int P = 0; P < problem_size.P; ++P) {
      for (int Q = 0; Q < problem_size.Q; ++Q) {

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kThreadM; ++m) {
          int thread_k = k_start + m;

          element_A[m] = ElementAccumulator();

          if (thread_k < problem_size.K) {
            element_A[m] = ElementAccumulator(tensor_dy.at({N, P, Q, thread_k}));
          }
        }

        // Load from filters tensor
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < kThreadN; ++n) {
          
          // Load from activations tensor
          int filter_r = thread_r[n];
          int filter_s = thread_s[n];

          if (problem_size.mode == cutlass::conv::Mode::kConvolution) {
            filter_r = problem_size.R - 1 - filter_r;
            filter_s = problem_size.S - 1 - filter_s;
          }

          int h = P * problem_size.stride_h - problem_size.pad_h + filter_r * problem_size.dilation_h;
          int w = Q * problem_size.stride_w - problem_size.pad_w + filter_s * problem_size.dilation_w;

          element_B[n] = ElementAccumulator();

          if (h >= 0 && h < problem_size.H && w >= 0 && w < problem_size.W && thread_c[n] < problem_size.C) {
            element_B[n] = ElementAccumulator(tensor_x.at({N, h, w, thread_c[n]}));
          }
        }

        // Accumulate matrix product
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kThreadM; ++m) {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < kThreadN; ++n) {
            accum[m][n] = inner_product_op(element_A[m], element_B[n], accum[m][n]);
          }
        }
      }
    }
  }

  // Write out the results
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    int thread_k = k_start + m;

    if (thread_k < problem_size.K) {
      
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kThreadN; ++n) {

        if (thread_r[n] < problem_size.R && thread_s[n] < problem_size.S && thread_c[n] < problem_size.C) {

          ElementCompute c_ref = ElementCompute();

          if (beta != ElementCompute()) {
            c_ref = ElementCompute(tensor_dw_in.at({thread_k, thread_r[n], thread_s[n], thread_c[n]}));
          }

          tensor_dw_out.at({thread_k, thread_r[n], thread_s[n], thread_c[n]}) = convert_op(
            alpha * ElementCompute(accum[m][n]) + beta * c_ref);
        }
      } 
    }
  }
}

// Conv3d wgrad kernel - dw = wgrad(dy, x)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>,
  int kThreadM = 2,       // shape of a thread's tile in the GEMM M dimension
  int kThreadN = 4,       // shape of a thread's tile in the GEMM N dimension
  int kCtaShapeM = 8,     // shape of a threadblock in units of threads
  int kCtaShapeN = 16     // shape of a threadblock in units of threads
>
__global__ void Conv3dWgrad(
  conv::Conv3dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_x,
  TensorRef<ElementC, LayoutC> tensor_dw_in,
  TensorRef<ElementC, LayoutC> tensor_dw_out,
  ElementCompute alpha,
  ElementCompute beta
  ) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  ElementAccumulator element_A[kThreadM];
  ElementAccumulator element_B[kThreadN];
  ElementAccumulator accum[kThreadM][kThreadN];

  int k_start = blockIdx.x * kCtaShapeM * kThreadM + threadIdx.x * kThreadM;
  int64_t trsc_start = int64_t(blockIdx.y) * kCtaShapeN * kThreadN + threadIdx.y * kThreadN;
  
  int thread_t[kThreadN];
  int thread_r[kThreadN];
  int thread_s[kThreadN];
  int thread_c[kThreadN];

  // Compute R, S, C coordinates for each row of a thread's tile
  int64_t SC = int64_t(problem_size.S) * problem_size.C;
  int64_t RSC = SC * problem_size.R;

  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < kThreadN; ++n) {

    int64_t trsc = trsc_start + n;

    thread_t[n] = int(trsc / RSC);

    int64_t residual = trsc % RSC;
    thread_r[n] = int(residual / SC);

    residual = residual % SC; 
    thread_s[n] = int(residual / problem_size.C);
    thread_c[n] = int(residual % problem_size.C);
  }

  // Clear accumulators
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kThreadN; ++n) {
      accum[m][n] = ElementAccumulator();
    }
  }

  // Compute convolution
  for (int N = 0; N < problem_size.N; ++N) {
    for (int Z = 0; Z < problem_size.Z; ++Z) {
      for (int P = 0; P < problem_size.P; ++P) {
        for (int Q = 0; Q < problem_size.Q; ++Q) {

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < kThreadM; ++m) {
            int thread_k = k_start + m;

            element_A[m] = ElementAccumulator();

            if (thread_k < problem_size.K) {
              element_A[m] = ElementAccumulator(tensor_dy.at({N, Z, P, Q, thread_k}));
            }
          }

          // Load from filters tensor
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < kThreadN; ++n) {
            
            // Load from activations tensor
            int filter_t = thread_t[n];
            int filter_r = thread_r[n];
            int filter_s = thread_s[n];

            if (problem_size.mode == cutlass::conv::Mode::kConvolution) {
              filter_t = problem_size.T - 1 - filter_t;
              filter_r = problem_size.R - 1 - filter_r;
              filter_s = problem_size.S - 1 - filter_s;
            }

            int d = Z * problem_size.stride_d - problem_size.pad_w + filter_t * problem_size.dilation_d;
            int h = P * problem_size.stride_h - problem_size.pad_h + filter_r * problem_size.dilation_h;
            int w = Q * problem_size.stride_w - problem_size.pad_w + filter_s * problem_size.dilation_w;

            element_B[n] = ElementAccumulator();

            if (d >= 0 && d < problem_size.D && 
              h >= 0 && h < problem_size.H && 
              w >= 0 && w < problem_size.W && 
              thread_c[n] < problem_size.C) {

              element_B[n] = ElementAccumulator(tensor_x.at({N, d, h, w, thread_c[n]}));
            }
          }

          // Accumulate matrix product
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < kThreadM; ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < kThreadN; ++n) {
              accum[m][n] = inner_product_op(element_A[m], element_B[n], accum[m][n]);
            }
          }

        } // for (Q)
      } // for (P)
    } // for (Z)
  } // for (N)

  // Write out the results
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    int thread_k = k_start + m;

    if (thread_k < problem_size.K) {
      
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kThreadN; ++n) {

        if (thread_t[n] < problem_size.T && 
          thread_r[n] < problem_size.R &&
          thread_s[n] < problem_size.S && 
          thread_c[n] < problem_size.C) {

          ElementCompute c_ref = ElementCompute();

          if (beta != ElementCompute()) {
            c_ref = ElementCompute(tensor_dw_in.at({thread_k, thread_t[n], thread_r[n], thread_s[n], thread_c[n]}));
          }

          tensor_dw_out.at({thread_k, thread_t[n], thread_r[n], thread_s[n], thread_c[n]}) = convert_op(
            alpha * ElementCompute(accum[m][n]) + beta * c_ref);
        }
      } 
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conv2d Fprop dispatcher - y = fprop(x, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
Status Conv2dFprop(
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_x,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_y_in,
  TensorRef<ElementC, LayoutC> tensor_y_out,
  ElementCompute alpha,
  ElementCompute beta,
  cudaStream_t stream = nullptr) {

  //
  // Blocking factors improve performance of reference implementation
  //

  int const kThreadM = 4;       // shape of a thread's tile in the GEMM M dimension
  int const kThreadN = 4;       // shape of a thread's tile in the GEMM N dimension
  int const kCtaShapeM = 16;    // shape of a threadblock in units of threads
  int const kCtaShapeN = 8;     // shape of a threadblock in units of threads

  int64_t npq = int64_t(problem_size.N) * problem_size.P * problem_size.Q;
  int64_t blocks_m = (npq + (kCtaShapeM * kThreadM) - 1) / (kCtaShapeM * kThreadM);

  dim3 block(kCtaShapeM, kCtaShapeN);
  dim3 grid(uint32_t(blocks_m), (problem_size.K + (kCtaShapeN * kThreadN) - 1) / (kCtaShapeN * kThreadN));

  kernel::Conv2dFprop<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ElementAccumulator,
    ConvertOp,
    InnerProductOp,
    kThreadM,
    kThreadN,
    kCtaShapeM,
    kCtaShapeN
  ><<< grid, block, 0, stream >>>(
    problem_size,
    tensor_x,
    tensor_w,
    tensor_y_in,
    tensor_y_out,
    alpha,
    beta
  );

  cudaError_t result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    return Status::kErrorInternal;
  }

  return Status::kSuccess;
}

/// Conv3d Fprop dispatcher - y = fprop(x, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
Status Conv3dFprop(
  conv::Conv3dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_x,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_y_in,
  TensorRef<ElementC, LayoutC> tensor_y_out,
  ElementCompute alpha,
  ElementCompute beta,
  cudaStream_t stream = nullptr) {

  //
  // Blocking factors improve performance of reference implementation
  //

  int const kThreadM = 4;       // shape of a thread's tile in the GEMM M dimension
  int const kThreadN = 4;       // shape of a thread's tile in the GEMM N dimension
  int const kCtaShapeM = 16;    // shape of a threadblock in units of threads
  int const kCtaShapeN = 8;     // shape of a threadblock in units of threads

  int64_t nzpq = int64_t(problem_size.N) * problem_size.Z * problem_size.P * problem_size.Q;
  int64_t blocks_m = (nzpq + (kCtaShapeM * kThreadM) - 1) / (kCtaShapeM * kThreadM);

  dim3 block(kCtaShapeM, kCtaShapeN);
  dim3 grid(uint32_t(blocks_m), (problem_size.K + (kCtaShapeN * kThreadN) - 1) / (kCtaShapeN * kThreadN));

  kernel::Conv3dFprop<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ElementAccumulator,
    ConvertOp,
    InnerProductOp,
    kThreadM,
    kThreadN,
    kCtaShapeM,
    kCtaShapeN
  ><<< grid, block, 0, stream >>>(
    problem_size,
    tensor_x,
    tensor_w,
    tensor_y_in,
    tensor_y_out,
    alpha,
    beta
  );

  cudaError_t result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    return Status::kErrorInternal;
  }

  return Status::kSuccess;
}

/// Conv2d Dgrad dispatcher - dx = dgrad(dy, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
Status Conv2dDgrad(
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_dx_in,
  TensorRef<ElementC, LayoutC> tensor_dx_out,
  ElementCompute alpha,
  ElementCompute beta,
  cudaStream_t stream = nullptr) {

  //
  // Blocking factors improve performance of reference implementation
  //

  int const kThreadM = 2;       // shape of a thread's tile in the GEMM M dimension
  int const kThreadN = 4;       // shape of a thread's tile in the GEMM N dimension
  int const kCtaShapeM = 16;    // shape of a threadblock in units of threads
  int const kCtaShapeN = 8;     // shape of a threadblock in units of threads

  int64_t nhw = int64_t(problem_size.N) * problem_size.H * problem_size.W;
  int64_t blocks_m = (nhw + (kCtaShapeM * kThreadM) - 1) / (kCtaShapeM * kThreadM);

  dim3 block(kCtaShapeM, kCtaShapeN);
  dim3 grid(uint32_t(blocks_m), (problem_size.C + (kCtaShapeN * kThreadN) - 1) / (kCtaShapeN * kThreadN));

  kernel::Conv2dDgrad<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ElementAccumulator,
    ConvertOp,
    InnerProductOp,
    kThreadM,
    kThreadN,
    kCtaShapeM,
    kCtaShapeN
  ><<< grid, block, 0, stream >>>(
    problem_size,
    tensor_dy,
    tensor_w,
    tensor_dx_in,
    tensor_dx_out,
    alpha,
    beta
  );

  cudaError_t result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    return Status::kErrorInternal;
  }

  return Status::kSuccess;
}

/// Conv3d Dgrad dispatcher - dx = dgrad(dy, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
Status Conv3dDgrad(
  conv::Conv3dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_dx_in,
  TensorRef<ElementC, LayoutC> tensor_dx_out,
  ElementCompute alpha,
  ElementCompute beta,
  cudaStream_t stream = nullptr) {

  //
  // Blocking factors improve performance of reference implementation
  //

  int const kThreadM = 2;       // shape of a thread's tile in the GEMM M dimension
  int const kThreadN = 4;       // shape of a thread's tile in the GEMM N dimension
  int const kCtaShapeM = 16;    // shape of a threadblock in units of threads
  int const kCtaShapeN = 8;     // shape of a threadblock in units of threads

  int64_t ndhw = int64_t(problem_size.N) * problem_size.D * problem_size.H * problem_size.W;
  int64_t blocks_m = (ndhw + (kCtaShapeM * kThreadM) - 1) / (kCtaShapeM * kThreadM);

  dim3 block(kCtaShapeM, kCtaShapeN);
  dim3 grid(uint32_t(blocks_m), (problem_size.C + (kCtaShapeN * kThreadN) - 1) / (kCtaShapeN * kThreadN));

  kernel::Conv3dDgrad<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ElementAccumulator,
    ConvertOp,
    InnerProductOp,
    kThreadM,
    kThreadN,
    kCtaShapeM,
    kCtaShapeN
  ><<< grid, block, 0, stream >>>(
    problem_size,
    tensor_dy,
    tensor_w,
    tensor_dx_in,
    tensor_dx_out,
    alpha,
    beta
  );

  cudaError_t result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    return Status::kErrorInternal;
  }

  return Status::kSuccess;
}

/// Conv2d Wgrad dispatcher - dw = wgrad(dy, x)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
Status Conv2dWgrad(
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_x,
  TensorRef<ElementC, LayoutC> tensor_dw_in,
  TensorRef<ElementC, LayoutC> tensor_dw_out,
  ElementCompute alpha,
  ElementCompute beta,
  cudaStream_t stream = nullptr) {

  //
  // Blocking factors improve performance of reference implementation
  //

  int const kThreadM = 2;       // shape of a thread's tile in the GEMM M dimension
  int const kThreadN = 4;       // shape of a thread's tile in the GEMM N dimension
  int const kCtaShapeM = 8;     // shape of a threadblock in units of threads
  int const kCtaShapeN = 16;    // shape of a threadblock in units of threads

  int64_t rsc = int64_t(problem_size.R) * problem_size.S * problem_size.C;
  int64_t blocks_n = (rsc + (kCtaShapeN * kThreadN) - 1) / (kCtaShapeN * kThreadN);

  dim3 block(kCtaShapeM, kCtaShapeN);
  dim3 grid((problem_size.K + (kCtaShapeM * kThreadM) - 1) / (kCtaShapeM * kThreadM), uint32_t(blocks_n));

  kernel::Conv2dWgrad<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ElementAccumulator,
    ConvertOp,
    InnerProductOp,
    kThreadM,
    kThreadN,
    kCtaShapeM,
    kCtaShapeN
  ><<< grid, block, 0, stream >>>(
    problem_size,
    tensor_dy,
    tensor_x,
    tensor_dw_in,
    tensor_dw_out,
    alpha,
    beta
  );

  cudaError_t result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    return Status::kErrorInternal;
  }

  return Status::kSuccess;
}

/// Conv3d Wgrad dispatcher - dw = wgrad(dy, x)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
Status Conv3dWgrad(
  conv::Conv3dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_x,
  TensorRef<ElementC, LayoutC> tensor_dw_in,
  TensorRef<ElementC, LayoutC> tensor_dw_out,
  ElementCompute alpha,
  ElementCompute beta,
  cudaStream_t stream = nullptr) {

  //
  // Blocking factors improve performance of reference implementation
  //

  int const kThreadM = 2;       // shape of a thread's tile in the GEMM M dimension
  int const kThreadN = 4;       // shape of a thread's tile in the GEMM N dimension
  int const kCtaShapeM = 8;     // shape of a threadblock in units of threads
  int const kCtaShapeN = 16;    // shape of a threadblock in units of threads

  int64_t trsc = int64_t(problem_size.T) * problem_size.R * problem_size.S * problem_size.C;
  int64_t blocks_n = (trsc + (kCtaShapeN * kThreadN) - 1) / (kCtaShapeN * kThreadN);

  dim3 block(kCtaShapeM, kCtaShapeN);
  dim3 grid((problem_size.K + (kCtaShapeM * kThreadM) - 1) / (kCtaShapeM * kThreadM), uint32_t(blocks_n));

  kernel::Conv3dWgrad<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ElementAccumulator,
    ConvertOp,
    InnerProductOp,
    kThreadM,
    kThreadN,
    kCtaShapeM,
    kCtaShapeN
  ><<< grid, block, 0, stream >>>(
    problem_size,
    tensor_dy,
    tensor_x,
    tensor_dw_in,
    tensor_dw_out,
    alpha,
    beta
  );

  cudaError_t result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    return Status::kErrorInternal;
  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Generic 2D convolution targeting Conv2dFprop, Conv2dDgrad, and Conv2dWgrad.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
Status Conv2d(
  conv::Operator convolutional_operator,
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_A,
  TensorRef<ElementB, LayoutB> tensor_B,
  TensorRef<ElementC, LayoutC> tensor_C,
  TensorRef<ElementC, LayoutC> tensor_D,
  ElementCompute alpha,
  ElementCompute beta,
  cudaStream_t stream = nullptr) {
  
  switch (convolutional_operator) {
  case conv::Operator::kFprop:
    return Conv2dFprop<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator,
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, stream);
    break;

  case conv::Operator::kDgrad:
    return Conv2dDgrad<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator,
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, stream);
    break;

  case conv::Operator::kWgrad:
    return Conv2dWgrad<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator,
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, stream);
    break;

  default: break;
  }
  
  return Status::kErrorNotSupported;
}

/// Generic 3D convolution targeting Conv3dFprop, Conv3dDgrad, and Conv3dWgrad.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
Status Conv3d(
  conv::Operator convolutional_operator,
  conv::Conv3dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_A,
  TensorRef<ElementB, LayoutB> tensor_B,
  TensorRef<ElementC, LayoutC> tensor_C,
  TensorRef<ElementC, LayoutC> tensor_D,
  ElementCompute alpha,
  ElementCompute beta,
  cudaStream_t stream = nullptr) {
  
  switch (convolutional_operator) {
  case conv::Operator::kFprop:
    return Conv3dFprop<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator, 
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, stream);

  case conv::Operator::kDgrad:
    return Conv3dDgrad<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator, 
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, stream);

  case conv::Operator::kWgrad:
    return Conv3dWgrad<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator, 
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, stream);

  default: break;
  }
  
  return Status::kErrorNotSupported;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace reference
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

