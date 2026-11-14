#include <torch/all.h>

#include "ime2_hp_kernel.inc"

at::Tensor ime2_hp_gemm_cpu(const at::Tensor& a, const at::Tensor& qweight,
                            const at::Tensor& scales) {
    TORCH_CHECK(a.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(a.dim() == 2 && a.is_contiguous(), "A must be contiguous [M, K]");
    TORCH_CHECK(qweight.scalar_type() == at::kByte && qweight.dim() == 3,
                "qweight must be uint8 [N, K/32, 16]");
    TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.dim() == 2,
                "scales must be float32 [N, K/32]");
    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = qweight.size(0);
    const int64_t bck = qweight.size(1);
    TORCH_CHECK(K > 0 && K % 32 == 0, "K must be a multiple of 32");
    TORCH_CHECK(bck == K / 32 && qweight.size(2) == 16, "qweight shape");
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == bck, "scales shape");
    TORCH_CHECK(vllm_ime2::ime_vlenb() == 128,
                "vmadot.hp runs on the A100 cores (vlenb 128)");

    auto qw = qweight.contiguous();
    auto sc = scales.contiguous();
    auto out = at::empty({M, N}, a.options());
    const size_t qa_bytes = vllm_ime2::Ime2HpQuantARowBytes(static_cast<size_t>(K)) *
                            static_cast<size_t>(M);
    const size_t qb_bytes =
        vllm_ime2::Ime2HpPackedBSize(static_cast<size_t>(N), static_cast<size_t>(bck));
    std::vector<uint8_t> qa(qa_bytes), qb(qb_bytes);
    vllm_ime2::Ime2HpQuantizeA(a.data_ptr<float>(), static_cast<size_t>(K),
                               static_cast<size_t>(M), static_cast<size_t>(K),
                               reinterpret_cast<std::byte*>(qa.data()));
    vllm_ime2::Ime2HpPack(static_cast<size_t>(N), static_cast<size_t>(bck),
                          qw.data_ptr<uint8_t>(), sc.data_ptr<float>(), qb.data());
    vllm_ime2::Ime2HpGemm(reinterpret_cast<const std::byte*>(qa.data()),
                          reinterpret_cast<const std::byte*>(qb.data()),
                          out.data_ptr<float>(), static_cast<size_t>(M),
                          static_cast<size_t>(N), static_cast<size_t>(bck),
                          static_cast<size_t>(N), nullptr);
    return out;
}
