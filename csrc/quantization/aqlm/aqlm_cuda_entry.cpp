#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void code1x16_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);

void code2x8_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);

void code1x16_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook
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
    prob_k
  );
}

torch::Tensor code1x16_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
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
    code1x16_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks
    );
  }
  flat_output *= scales.flatten().unsqueeze(0);

  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  auto output = flat_output.reshape(output_sizes); // .clone();
  return output;
}

void code2x8_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook
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
    prob_k
  );
}

torch::Tensor code2x8_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
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
      codebooks
    );
  }
  flat_output *= scales.flatten().unsqueeze(0);
  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  auto output = flat_output.reshape(output_sizes).clone();
  return output;
}
