/*
 * Modified by Neural Magic
 * Adapted from https://github.com/Vahe1994/AQLM
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <cstdlib>

void code1x16_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k,
  const int4 codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at most 3 long.
  const int codebook_stride // as int4.
);

void code2x8_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k,
  const int4 codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at most 3 long.
  const int codebook_stride // as int4.
);

void code1x16_dequant_cuda(
  const void* A,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k,
  const int4 codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at most 3 long.
  const int codebook_stride // as int4.
);

void code2x8_dequant_cuda(
  const void* A,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k,
  const int4 codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at most 3 long, corresponds to cols.
  const int codebook_stride // as int4
);


int codebook_stride(const torch::Tensor& codebooks)
{
  return codebooks.stride(0) * codebooks.element_size() / sizeof(int4);
}

void code1x16_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook,
  const int4 codebook_a_sizes  // cumulative sizes of A spanning each codebook, at most 3 long.
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);

  code1x16_matvec_cuda(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    codebook.data_ptr(),
    prob_m,
    prob_k,
    codebook_a_sizes,
    codebook_stride(codebook)
  );
}

torch::Tensor code1x16_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const int4 codebook_a_sizes,
  const std::optional<torch::Tensor>& bias) {
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code1x16_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks,
      codebook_a_sizes
    );
  }
  flat_output *= scales.flatten().unsqueeze(0);

  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  auto output = flat_output.reshape(output_sizes);
  return output;
}

void code2x8_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook,
  const int4 codebook_a_sizes
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);
  code2x8_matvec_cuda(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    codebook.data_ptr(),
    prob_m,
    prob_k,
    codebook_a_sizes,
    2 * codebook_stride(codebook)
  );
}

torch::Tensor code2x8_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const int4 codebook_a_sizes,
  const std::optional<torch::Tensor>& bias
) {
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code2x8_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks,
      codebook_a_sizes
    );
  }
  flat_output *= scales.flatten().unsqueeze(0);
  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  auto output = flat_output.reshape(output_sizes);
  return output;
}

// Accumulate the partition sizes.
int4 accumulate_sizes(const torch::Tensor& codebook_partition_sizes)
{
  int4 cumulative_sizes;
  auto cumulative_size = &cumulative_sizes.x;
  int i = 0;
  int last = 0;
  assert(codebook_partition_sizes.size(0) <= 4);
  for (; i <  codebook_partition_sizes.size(0); ++i, ++cumulative_size)
  {
    *cumulative_size = codebook_partition_sizes[i].item<int>() + last;
    last = *cumulative_size;
  }
  // fill in the rest with unreachable.
  for (; i < 4; ++i, ++cumulative_size)
  {
    *cumulative_size = last*10;
  }
  return cumulative_sizes;
}

torch::Tensor aqlm_gemm(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const torch::Tensor& codebook_partition_sizes,
  const std::optional<torch::Tensor>& bias
)
{
  int4 cumulative_sizes = accumulate_sizes(codebook_partition_sizes);

  int const nbooks = codebooks.size(0) / codebook_partition_sizes.size(0);
  int const entries = codebooks.size(1);

  if (nbooks == 1 && entries == (1 << 16))
  { 
    return code1x16_matmat(input, codes, codebooks, scales, cumulative_sizes, bias);
  }
  if (nbooks == 2 && entries == (1 << 8))
  {
    return code2x8_matmat(input, codes, codebooks, scales, cumulative_sizes, bias);
  }

  TORCH_CHECK(false, "AQLM with ", nbooks, " codebooks and ", entries, " entries is not currently supported.")
  return {};
}

torch::Tensor aqlm_dequant(
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& codebook_partition_sizes
)
{
  int4 cumulative_sizes = accumulate_sizes(codebook_partition_sizes);

  int const nbooks = codebooks.size(0) / codebook_partition_sizes.size(0);
  int const entries = codebooks.size(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(codes));
  int rows = codes.size(1);
  int cols = codes.size(0);

  auto in_features = codes.size(1) * 8;
  auto out_features = codes.size(0);

  assert(out_features = codebook_partition_sizes.sum().item<int>());

  auto weights = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );

  if (nbooks == 1 && entries == (1 << 16))
  {
    code1x16_dequant_cuda(
      codes.data_ptr(),
      weights.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features,
      cumulative_sizes,
      codebook_stride(codebooks));

    // if you wanted to flip to scaling the weights, (though it's 30%-ish slower and not consistent with gemv implementation.)
    // weights *= scales.index({"...", 0, 0});

     return weights;
  }

  if (nbooks == 2 && entries == (1 << 8))
  {
     code2x8_dequant_cuda(
        codes.data_ptr(), 
        weights.data_ptr(), 
        codebooks.data_ptr(), 
        out_features,
        in_features, 
        cumulative_sizes, 
        codebook_stride(codebooks));

    // if you wanted to flip to scaling the weights, (though it's 30%-ish slower and not consistent with gemv implementation)
    // weights *= scales.index({"...", 0, 0});

     return weights;
  }

  TORCH_CHECK(false, "AQLM with ", nbooks, " codebooks and ", entries, " entries is not currently supported.")
  return {};
}
