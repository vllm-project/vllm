#pragma once

/*
  This runner supports:
  FP4 inputs (A and B)
  float blockwise scaling factor
  float alpha scalings
  T output (D) where T = {float, half, __nv_bfloat16}

  Activations, biases and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
  Block scaling factor are interleaved.
*/

class CutlassFp4GemmRunnerInterface {
 public:
  CutlassFp4GemmRunnerInterface() {}

  virtual ~CutlassFp4GemmRunnerInterface() {}

  virtual void cutlass_scaled_fp4_mm(
      void* D, void const* A, void const* B, void const* input_sf,
      void const* weight_sf, float const* global_sf, int m, int n, int k,
      char* workspace, const size_t workspaceBytes, cudaStream_t stream) = 0;

  // Returns desired workspace size in bytes.
  virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

  virtual std::vector<CutlassGemmConfig> getConfigs() const = 0;
};

template <typename T>
class CutlassFp4GemmRunner : public virtual CutlassFp4GemmRunnerInterface {
 public:
  CutlassFp4GemmRunner();
  ~CutlassFp4GemmRunner();

  void CutlassFp4GemmRunner::cutlass_scaled_fp4_mm(
      torch::Tensor& D, torch::Tensor& A, torch::Tensor& B,
      torch::Tensor& input_sf, torch::Tensor& weight_sf,
      torch::Tensor& global_sf, CutlassGemmConfig gemmConfig,
      torch::Tensor& workspace, const size_t workspaceBytes);

  void cutlass_scaled_fp4_mm(void* D, void const* A, void const* B,
                             void const* input_sf, void const* weight_sf,
                             float const* global_sf, int m, int n, int k,
                             CutlassGemmConfig gemmConfig, char* workspace,
                             const size_t workspaceBytes,
                             cudaStream_t stream) override;

  // Returns desired workspace size in bytes.
  size_t getWorkspaceSize(int const m, int const n, int const k) override;

  std::vector<CutlassGemmConfig> getConfigs() const override;

 private:
  size_t dispatchToArch(T* D, void const* A, void const* B,
                        void const* input_sf, void const* weight_sf,
                        float const* global_sf, int m, int n, int k,
                        CutlassGemmConfig gemmConfig, char* workspace,
                        const size_t workspaceBytes, cudaStream_t stream,
                        int* occupancy = nullptr);

  size_t getWorkspaceSizeImpl(int const m, int const n, int const k);

  int mSm;
  int mMultiProcessorCount;
};
