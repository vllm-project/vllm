#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <functional>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <memory>
#include "primitives.h"
#include "forward_context.h"

#ifndef cudaErrorNotInitialized
#define cudaErrorNotInitialized ((cudaError_t)1000)  // Arbitrary unused error code
#endif

// MOE配置结构体
struct MOEConfig {
    int tp_rank = 0;
    int tp_size = 1;
    int expert_num = 0;
    int num_experts_per_tok = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int max_batch_token = 0;
    int cache_expert_num = 0;
    int block_size = 0;
    int cache_topk = 0;
    int update_expert_num = 0;
    int forward_context_num_threads = 14;  // ForwardContext 线程数
    bool normTopKProb = false;
    int nGroup = 0;
    int topKGroup = 0;

    MOEConfig() = default;
    MOEConfig(int tp_rank, int tp_size, int expert_num, int num_experts_per_tok,
              int hidden_size, int intermediate_size, int max_batch_token,
              int cache_expert_num, int block_size, int cache_topk, int update_expert_num,
              int forward_context_num_threads = 14);
};

struct AsyncState {
    // FIXED: 移除 volatile，使用 memory fence 保证同步
    int32_t gpu_signal;
    int32_t callback_completed;
    int32_t layer_idx;
    int32_t batch_idx;
    int32_t num_tokens;
    int32_t sync_count;
    int32_t submit_count;
    int32_t complete_count;

    AsyncState();
};
// MoE主类
class Moe {
public:
    // 构造函数：接收指针和形状信息
    Moe(float8_e4m3_t* w13_weights, float8_e4m3_t* w2_weights,
        float* w13_scales, float* w2_scales, int layer_id,
        const MOEConfig& config, ForwardContext* ctx = nullptr);

    ~Moe();
    int layer_id() const { return  layer_id_;}
    // 前向接口：接收原始指针
    void forward(bfloat16_t* input, int* topk_ids, float* topk_weights,
                 bfloat16_t* output, int num_tokens);
    
    // 前向接口：接收 Tensor（重载）
    void forward(torch::Tensor input, torch::Tensor topk_ids,
                 torch::Tensor topk_weights, torch::Tensor output, int num_tokens);
    
    // CUDA 方法：更新 expert cache
    void update_expert_cache(torch::Tensor w13_cache, torch::Tensor w2_cache,
                            torch::Tensor w13_scale_cache, torch::Tensor w2_scale_cache,
                            torch::Tensor map, int64_t num_experts);
private:
    // 核心方法
    void topk_sort_inplace(int *topk_ids, float *topk_weights, int n_tokens,
                                   int num_experts_per_tok, int num_experts);

    void packet_input(bfloat16_t *input, bfloat16_t *padding_buf, std::vector<int> ids, int stride);

    void combine_expert_output(float *output, float *expert_output,
                                const std::vector<int>& ids,
                                const std::vector<float>& weights,
                                int stride);

    void forward_single_expert(bfloat16_t *input, float* output, int expert_id, int n_tokens);

    void forward_experts(bfloat16_t *input, int *topk_ids, float *topk_weights,  bfloat16_t *output, int n_tokens, ForwardContext *ctx);

    void forward_sparse(bfloat16_t *input, int *topk_ids, float *topk_weights, bfloat16_t *output, int n_tokens, ForwardContext *ctx);

    // 成员变量
    MOEConfig config_;
    int layer_id_;

    // 权重指针
    float8_e4m3_t* m_w13_weights = nullptr;
    float8_e4m3_t* m_w2_weights = nullptr;
    float* m_w13_scale = nullptr;
    float* m_w2_scale = nullptr;

    // 权重buffer大小（用于边界检查）
    int64_t w13_weight_size_ = 0;
    int64_t w2_weight_size_ = 0;
    int64_t w13_scale_size_ = 0;
    int64_t w2_scale_size_ = 0;

    ForwardContext* ctx = nullptr;
    bool owns_ctx_ = false; 
};


class MoeOffloadEngine {
    public:
        explicit MoeOffloadEngine(const MOEConfig& config);
        ~MoeOffloadEngine();
    
        uintptr_t ptr() { return reinterpret_cast<uintptr_t>(this); }
    
        void create_cpu_moe_layer(torch::Tensor w13_weight, torch::Tensor w2_weight,
                                 torch::Tensor w13_scale, torch::Tensor w2_scale,
                                 int layer_id);
    
        Moe* get_cpu_moe_layer(int layer_id);
        const Moe* get_cpu_moe_layer(int layer_id) const;
    
        void update_expert_cache(torch::Tensor w13_cache, torch::Tensor w2_cache,
                                              torch::Tensor w13_scale_cache, torch::Tensor w2_scale_cache,
                                              torch::Tensor map, int layer_id, int num_experts);
        
        void expert_cache_policy(torch::Tensor cache_map, torch::Tensor miss_map, 
                                torch::Tensor policy_sort, torch::Tensor topk_ids, 
                                torch::Tensor cpu_topk, torch::Tensor copy_map);
    
        void get_output(torch::Tensor gpu_output);
        void set_input(torch::Tensor gpu_hidden_states, torch::Tensor gpu_topk_ids, 
                      torch::Tensor gpu_topk_weights);
    
        void call(int layer_idx, int batch_idx, int num_tokens);
        // FIXED: 移除重复类名限定
        void initialize_async_state();
        void cleanup_async_state();
        void cpu_polling_loop();
    
        // FIXED: 添加参数类型
        cudaError_t submit(int layer_idx, int batch_idx, int num_tokens);
        cudaError_t sync();
    
    private:
        MOEConfig config_;
        // FIXED: 添加成员变量名
        std::unordered_map<int, Moe> moe_layers_;

        std::unique_ptr<ForwardContext> forward_context_;

        // CPU input/output buffers
        torch::Tensor output_;
        torch::Tensor topk_ids_;
        torch::Tensor topk_weights_;
        torch::Tensor hidden_states_;

        // Async state
        AsyncState* cpu_state_;
        AsyncState* gpu_state_;
        std::thread polling_thread_;
        std::atomic<bool> shutdown_;
        bool async_state_initialized_;
    
    };
    