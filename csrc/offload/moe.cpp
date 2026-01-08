#include "moe.h"
#ifndef __NVCC__
#include <immintrin.h>
#endif
#include <cstring>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <nvtx3/nvToolsExt.h>
#include <torch/extension.h>

MOEConfig::MOEConfig(int tp_rank, int tp_size, int expert_num, int num_experts_per_tok,
                     int hidden_size, int intermediate_size, int max_batch_token,
                     int cache_expert_num, int block_size, int cache_topk, int update_expert_num,
                     int forward_context_num_threads)
    : tp_rank(tp_rank), tp_size(tp_size), expert_num(expert_num),
      num_experts_per_tok(num_experts_per_tok), hidden_size(hidden_size),
      intermediate_size(intermediate_size), max_batch_token(max_batch_token),
      cache_expert_num(cache_expert_num), block_size(block_size),
      cache_topk(cache_topk), update_expert_num(update_expert_num),
      forward_context_num_threads(forward_context_num_threads) {}


Moe::Moe(float8_e4m3_t* w13_weights, float8_e4m3_t* w2_weights,
         float* w13_scales, float* w2_scales, int layer_id,
         const MOEConfig& config, ForwardContext* ctx)
    : config_(config), layer_id_(layer_id), m_w13_weights(w13_weights), m_w2_weights(w2_weights),
      m_w13_scale(w13_scales), m_w2_scale(w2_scales), ctx(ctx){

    // 验证指针非空
    if (!w13_weights || !w2_weights || !w13_scales || !w2_scales) {
        throw std::invalid_argument("Weight/scale pointers cannot be null");
    }

    if (!ctx) {
        this->ctx = new ForwardContext(config.forward_context_num_threads);
        std::cout << "ForwardContext created" << std::endl;
        owns_ctx_ = true;
        if (!set_tiledata_use()) {
            throw std::runtime_error("Failed to enable AMX tile data. Ensure CPU supports AMX.");
        }
    } else {
        this->ctx = ctx;
        owns_ctx_ = false;
    }

}

Moe::~Moe() {
    if (owns_ctx_ && ctx) {
        delete ctx;
        ctx = nullptr;
    }
}

AsyncState::AsyncState()
    : gpu_signal(0),
      callback_completed(1),
      layer_idx(0),
      batch_idx(0),
      num_tokens(0),
      sync_count(0),
      submit_count(0),
      complete_count(0) {}


static float act_fn(float x) {
    return x / (1.0f + expf(-x)); // expf,fabsf
}

void Moe::topk_sort_inplace(int *topk_ids, float *topk_weights, int n_tokens,
                                   int num_experts_per_tok, int num_experts) {
    //std::cout<<"topk_Sort"<<std::endl;
    const int top_k = config_.num_experts_per_tok;
    ctx->do_work_stealing_job(n_tokens, [&](int thread_id, int idx){
        int token_id = idx;
        int write_idx = 0;
        for (int j = 0; j < top_k; ++j) {
            int id = topk_ids[token_id * top_k + j];
            if (id != -1) {
                topk_ids[token_id * top_k + write_idx] = id;
                if (j != write_idx) {
                    topk_weights[token_id * top_k + write_idx] =
                        topk_weights[token_id * top_k + j];
                }
                ++write_idx;
                if (write_idx == num_experts) break;
            }
        }
        for (int j = write_idx; j < top_k; ++j) {
            topk_ids[token_id * top_k + j] = -1;
        }
    });
}

void Moe::packet_input(bfloat16_t *input, bfloat16_t *padding_buf, std::vector<int> ids, int stride)
{
    static const int MB = 32;
    static const int KB = 32;
    int hidden_size = config_.hidden_size;
    ctx->do_work_stealing_job(ids.size(), [&](int thread_id, int idx){
        for(int k = 0; k < stride; k+=KB)
        {
            memcpy(padding_buf + idx / MB * MB * hidden_size + idx % MB * KB + k * MB, input + ids[idx] * stride + k, KB * sizeof(bfloat16_t));
        }
    });

}

// 合并专家输出
void Moe::combine_expert_output(float *output, float *expert_output,
                                const std::vector<int>& ids,
                                const std::vector<float>& weights,
                                int stride) {
    constexpr int kStep = 16;
    const int n = ids.size();
    ctx->do_work_stealing_job(n, [&](int thread_id, int idx) {
        int token_id = ids[idx];
        __m512 weight_vec = _mm512_set1_ps(weights[idx]);
        float *out_ptr = output + token_id * stride;
        float *data = expert_output  + idx * stride;

        for (int i = 0; i < stride; i += kStep) {
            __m512 v = _mm512_loadu_ps(data + i);
            __m512 o = _mm512_loadu_ps(out_ptr + i);
            __m512 c = _mm512_fmadd_ps(v, weight_vec, o);
            _mm512_storeu_ps(out_ptr + i, c);
        }
    });

}


void Moe::forward_single_expert(bfloat16_t *input, float* output, int expert_id, int n_tokens)
{
    std::string nvtx_msg = "forwardSingleExpert_expert_" + std::to_string(expert_id) +
                          "_ntoks_" + std::to_string(n_tokens);
    nvtxRangePushA(nvtx_msg.c_str());

    constexpr int BM = 32;
    constexpr int BN = 32;

    int intermediate_size = config_.intermediate_size;
    int hidden_size = config_.hidden_size;
    int block_size = config_.block_size;
    //std::cout << "n_tokens = " << n_tokens << std::endl;
    float* gate_up_output  = static_cast<float*>(ctx->getBuffer("gate_up_output",  n_tokens * intermediate_size * 2 * sizeof(float)));
    bfloat16_t* m_down_input = static_cast<bfloat16_t*>(ctx->getBuffer("down_input", n_tokens * intermediate_size * sizeof(bfloat16_t)));

    memset(gate_up_output, 0, n_tokens * intermediate_size * 2 * sizeof(float));
    memset(output, 0, n_tokens * hidden_size * sizeof(float));

    int nth = intermediate_size * 2 / BN;  // 每个任务计算input BM行,  weight BN列  7168 * 2   gate/up 跨度 7168 * 32
    ctx->do_work_stealing_job((n_tokens / BM) * nth , [&](int thread_id, int idx){
        // idx / nth ==> 下一组数据
        // input  idx / nth * BM * hidden_size          OK!
        // weight/scale                                 OK!
        // out    idx/ nth * BM * intermediate_size * 2
        bfloat16_t*     input_ptr       = input            + idx / nth * BM * hidden_size;
        float8_e4m3_t*  weights         = m_w13_weights    + expert_id * intermediate_size * hidden_size * 2                         + idx % nth * BN * hidden_size;
        float*          scale           = m_w13_scale      + expert_id * intermediate_size/block_size * hidden_size/block_size * 2   + (idx % nth * BN/ block_size) * (hidden_size/block_size);
        float*          out             = gate_up_output   + (idx / nth) * BM * intermediate_size * 2                                  + idx % nth * BM * BN;
        amx_gemm_block_32_K_32(input_ptr, weights, scale, out, hidden_size, 32);
    });

    //dump_martix(gate_up_output, 32, 8, "gate_up_output", BM);
    //dump_martix(gate_up_output + BM * intermediate_size * 2, 2, 8, "gate_up_output", BM);
    // 每个任务处理一行内容
    // src = idx / BN  *  BN * intermediate_size * 2  + idx %BN * intermediate_size
    // dst = idx / BN  *  BN * intermediate_size      + idx %BN * intermediate_size
    // 64行out   0 -> 0 32
    //          1 -> 1 33
    //          2 -> 2 34
    //          32 -> 64  96
    //          33 -> 65

    ctx->do_work_stealing_job(n_tokens, [&](int thread_id, int idx){
        //int start = idx / 32 * 32 * intermediate_size * 2 + idx % 32 * intermediate_size;
        //int end = start + intermediate_size;
        //for (int j = start; j < end; j++)
        //{
            //m_down_input[j] = bfloat16_t::from_float(act_fn(gate_up_output[j]) * gate_up_output[j + intermediate_size * 32]);
        //}
        float*      gate = gate_up_output + idx / BN  *  BN * intermediate_size * 2  + idx % BN * intermediate_size;
        float*        up = gate + intermediate_size * BN;
        bfloat16_t*  out = m_down_input   + idx / BN  *  BN * intermediate_size      + idx % BN * intermediate_size;
        for( int j = 0; j < intermediate_size; j++)
        {
            out[j] = bfloat16_t::from_float(act_fn(gate[j]) * up[j]);
        }
    });

    // 每个任务计算input BM行,  weight BN列
    nth = hidden_size / BN;
    ctx->do_work_stealing_job((n_tokens / BM) * nth, [&](int thread_id, int idx) {
        bfloat16_t*    down_input_ptr = m_down_input  + idx / nth * BM * intermediate_size;
        float*         down_output    = output        + idx / nth * BM * hidden_size                                              + idx % nth * BN;

        float8_e4m3_t* weights        = m_w2_weights  + expert_id * intermediate_size *  hidden_size                              + idx % nth * BN * intermediate_size;
        float*         scale          = m_w2_scale    + expert_id * (intermediate_size/block_size) * (hidden_size / block_size)   + (idx % nth * BN/ block_size) * (intermediate_size / block_size);

        amx_gemm_block_32_K_32(down_input_ptr, weights, scale, down_output, intermediate_size, hidden_size);
    });

    nvtxRangePop();
}


void Moe::forward_experts(bfloat16_t *input, int *topk_ids, float *topk_weights,  bfloat16_t *output, int n_tokens, ForwardContext *ctx)
{
    std::string nvtx_msg = "forwardExperts_ntoks_" + std::to_string(n_tokens);
    nvtxRangePushA(nvtx_msg.c_str());

    const int expert_num = config_.expert_num;
    const int top_k = config_.num_experts_per_tok;
    const int hidden_size = config_.hidden_size;
    constexpr int MB = 32;

    std::vector<int> ids[expert_num]; // index for each expert
    std::vector<float> weights[expert_num]; // weight for each expert
    float *output_fp32_buf = (float *)ctx->getBuffer("output_fp32", n_tokens * hidden_size * sizeof(float));
    memset(output_fp32_buf, 0, n_tokens * hidden_size * sizeof(float));
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < top_k; ++j) {
            int expert_id = topk_ids[i * top_k + j];
            if (expert_id < 0) break;
            ids[expert_id].push_back(i);
            weights[expert_id].push_back(topk_weights[i * top_k + j]);
        }
    }


    for(int expert_id = 0; expert_id < expert_num; expert_id++)
    {
        if(ids[expert_id].size()==0) continue;
        int padding_len = (ids[expert_id].size() + MB - 1) / MB * MB;
        bfloat16_t * input_packet    = static_cast<bfloat16_t*>(ctx->getBuffer("input_packet",  padding_len * config_.hidden_size * sizeof(bfloat16_t)));
        float      * m_down_output   = static_cast<float*>(ctx->getBuffer("m_down_output",      padding_len * config_.hidden_size * sizeof(float)));
        packet_input(input, input_packet, ids[expert_id], config_.hidden_size);

        forward_single_expert(input_packet, m_down_output, expert_id, padding_len);

        combine_expert_output(output_fp32_buf, m_down_output, ids[expert_id], weights[expert_id], config_.hidden_size);
    }

    ctx->do_work_stealing_job(n_tokens, [&](int thread_id, int idx){
        fp32_to_bf16(output_fp32_buf + idx * hidden_size, output + idx * hidden_size, hidden_size);
    });

    nvtxRangePop();
}

void Moe::forward_sparse(bfloat16_t *input, int *topk_ids, float *topk_weights, bfloat16_t *output, int n_tokens, ForwardContext *ctx)
{
    std::string nvtx_msg = "forwardSparse_ntoks_" + std::to_string(n_tokens);
    nvtxRangePushA(nvtx_msg.c_str());

    int task_num = 0;
    int intermediate_size = config_.intermediate_size;
    int hidden_size = config_.hidden_size;
    int top_k = config_.num_experts_per_tok;
    int block_size = config_.block_size;




    float      *m_gate_up_output = (float*)ctx->getBuffer("m_gate_up_output",       n_tokens * top_k * hidden_size * 2 * sizeof(float));
    bfloat16_t *m_down_input     = (bfloat16_t*)ctx->getBuffer("m_down_input", n_tokens * top_k * intermediate_size * sizeof(bfloat16_t));
    float      *m_down_output      = (float *)ctx->getBuffer("down_output",           n_tokens * top_k * hidden_size * sizeof(float));

    int* input_ids  = (int*)ctx->getBuffer("input_ids",  n_tokens * top_k * sizeof(int));
    int* expert_ids = (int*)ctx->getBuffer("expert_ids", n_tokens * top_k * sizeof(int));
    int* task_ids   = (int*)ctx->getBuffer("task_ids",   n_tokens * top_k * sizeof(int));
    int* n_act      = (int*)ctx->getBuffer("n_act",      n_tokens * sizeof(int));

    for(int i = 0; i < n_tokens; i++) {
        int act = 0;
        for(int j = 0; j < top_k; j++)
        {
            if(topk_ids[i * top_k + j] == -1) break;
            input_ids[task_num]     = i;
            expert_ids[task_num]    = topk_ids[i * top_k + j];
            task_ids[i * top_k + j] = task_num;
            act ++;
            task_num++;
        }
        n_act[i] = act;
    }


    if(task_num != 0)
    {
        uint64_t stride = block_size;
        uint64_t nth = 2 * intermediate_size / stride;
        uint64_t weight_size = hidden_size * intermediate_size * 2;
        uint64_t weight_stride_size = stride * hidden_size;
        uint64_t scale_size  = (hidden_size / block_size) * (intermediate_size / block_size) * 2;
        uint64_t scale_stride_size = hidden_size / block_size;


        ctx->do_work_stealing_job(nth * task_num, [&](int thread_id, int idx){
            uint64_t task_id = idx / nth;
            uint64_t ith     = idx % nth;

            bfloat16_t*     input_ptr         = input + input_ids[task_id] * hidden_size;
            float8_e4m3_t*  weights           = m_w13_weights + expert_ids[task_id] * weight_size  + ith * weight_stride_size;
            float*          scale             = m_w13_scale   + expert_ids[task_id] * scale_size   + ith * scale_stride_size;
            float*          gate_up_output    = m_gate_up_output + task_id * intermediate_size * 2 + ith * stride;

            gemv_anni_grouped(input_ptr, (const uint8_t *)weights, scale, gate_up_output, stride, hidden_size, block_size);

        });


        nth = intermediate_size / stride;

        ctx->do_work_stealing_job(nth * task_num, [&](int thread_id, int idx){
            uint64_t task_id = idx / nth;
            uint64_t ith     = idx % nth;

            bfloat16_t*    down_input_ptr = m_down_input     + task_id * intermediate_size     + ith * stride;
            float*         gate_up_output = m_gate_up_output + task_id * intermediate_size * 2 + ith * stride;

            for (uint64_t j = 0; j <  stride; j++)
            {
                down_input_ptr[j] = bfloat16_t::from_float(act_fn(gate_up_output[j]) * gate_up_output[j + intermediate_size]);
            }
        });


        weight_size = hidden_size * intermediate_size;
        weight_stride_size = stride * intermediate_size;
        scale_size  = (hidden_size / block_size) * (intermediate_size / block_size);
        scale_stride_size = intermediate_size / block_size;


        nth = hidden_size / stride;
        ctx->do_work_stealing_job(nth * task_num, [&](int thread_id, int idx) {
            uint32_t task_id = idx / nth;
            uint64_t ith     = idx % nth;

            bfloat16_t*    down_input_ptr = m_down_input  + task_id * intermediate_size;
            float8_e4m3_t* weights        = m_w2_weights  + expert_ids[task_id] * weight_size + ith * weight_stride_size;
            float*         scale          = m_w2_scale    + expert_ids[task_id] * scale_size  + ith * scale_stride_size;
            float*         down_output    = m_down_output + task_id * hidden_size             + ith * stride;

            gemv_anni_grouped(down_input_ptr, (const uint8_t *)weights, scale, down_output, stride, intermediate_size, block_size);

        });

    }




    ctx->do_work_stealing_job(n_tokens, [&](int thread_id, int idx) {
        __m512 vw[8];
        int active_num = n_act[idx];

        for(int t = 0; t < active_num; t++)
        {
            vw[t] = _mm512_set1_ps(topk_weights[idx * top_k + t]);
        }

        for (int m = 0; m < config_.hidden_size; m+=16)
        {
            __m512 vo = _mm512_setzero_ps();
            for (int j = 0; j < active_num; j++) {
                __m512 vi = _mm512_load_ps(m_down_output + m + task_ids[idx * top_k + j] * hidden_size);
                vo = _mm512_fmadd_ps(vi, vw[j], vo);
            }
            _mm256_storeu_si256((__m256i_u*)(output + idx * hidden_size + m) , (__m256i)_mm512_cvtneps_pbh(vo));
        }
    });


    nvtxRangePop();
}

void Moe::forward(bfloat16_t *input, int *topk_ids,
                  float *topk_weights, bfloat16_t *output,
                  int num_tokens) {
    std::string nvtx_msg = "forwardMoE_layer_" + std::to_string(layer_id_) +
                          "_ntoks_" + std::to_string(num_tokens);
    nvtxRangePushA(nvtx_msg.c_str());


    topk_sort_inplace(topk_ids, topk_weights, num_tokens, config_.num_experts_per_tok, config_.expert_num);
    if (num_tokens < 128) {
        forward_sparse(input, topk_ids, topk_weights, output, num_tokens, ctx);
    } else {
        forward_experts(input, topk_ids, topk_weights, output, num_tokens, ctx);
    }

    nvtxRangePop();
}

void Moe::forward(torch::Tensor input, torch::Tensor topk_ids,
                  torch::Tensor topk_weights, torch::Tensor output,
                  int num_tokens) {
    TORCH_CHECK(input.device().is_cpu(), "Input must be CPU tensor");
    TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(num_tokens <= config_.max_batch_token, "num_tokens exceeds max_batch_token");



    auto* input_ptr = reinterpret_cast<bfloat16_t*>(input.data_ptr());
    auto* topk_ids_ptr = topk_ids.data_ptr<int>();
    auto* topk_weights_ptr = topk_weights.data_ptr<float>();
    auto* output_ptr = reinterpret_cast<bfloat16_t*>(output.data_ptr());
    

    forward(input_ptr, topk_ids_ptr, topk_weights_ptr, output_ptr, num_tokens);
}




// ==================== MoeOffloadEngine 实现 ====================
MoeOffloadEngine::MoeOffloadEngine(const MOEConfig& config)
    : config_(config),
      cpu_state_(nullptr),
      gpu_state_(nullptr),
      shutdown_(false),
      async_state_initialized_(false) {

    try {
        if (!set_tiledata_use()) {
            throw std::runtime_error("Failed to enable AMX tile data. Ensure CPU supports AMX.");
        }
        forward_context_ = std::make_unique<ForwardContext>(config.forward_context_num_threads);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] MoeOffloadEngine: Failed to initialize ForwardContext: " << e.what() << std::endl;
        throw;
    }

    try {

        size_t output_size = config_.max_batch_token * config_.hidden_size * 2;  // BFloat16 = 2 bytes
        size_t topk_ids_size = config_.max_batch_token * config_.num_experts_per_tok * sizeof(int32_t);
        size_t topk_weights_size = config_.max_batch_token * config_.num_experts_per_tok * sizeof(float);
        size_t hidden_states_size = config_.max_batch_token * config_.hidden_size * 2;  // BFloat16 = 2 bytes
        size_t total_size = output_size + topk_ids_size + topk_weights_size + hidden_states_size;

        output_ = torch::zeros({config_.max_batch_token, config_.hidden_size}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kBFloat16).pinned_memory(true));

        topk_ids_ = torch::zeros({config_.max_batch_token, config_.num_experts_per_tok}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32).pinned_memory(true));

        topk_weights_ = torch::zeros({config_.max_batch_token, config_.num_experts_per_tok}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(true));

        hidden_states_ = torch::zeros({config_.max_batch_token, config_.hidden_size}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kBFloat16).pinned_memory(true));

        initialize_async_state();
    } catch (const std::bad_alloc& e) {
        std::cerr << "[ERROR] MoeOffloadEngine: Memory allocation failed (std::bad_alloc)" << std::endl;
        std::cerr << "  max_batch_token=" << config_.max_batch_token
                  << ", hidden_size=" << config_.hidden_size
                  << ", num_experts_per_tok=" << config_.num_experts_per_tok << std::endl;
        std::cerr << "  Total pinned memory required: "
                  << (config_.max_batch_token * config_.hidden_size * 2 * 2 +  // 2 BFloat16 tensors, 2 bytes each
                      config_.max_batch_token * config_.num_experts_per_tok * (sizeof(int32_t) + sizeof(float))) / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cerr << "  Suggestion: Reduce max_batch_token or check available system memory" << std::endl;
        throw std::runtime_error("Failed to allocate pinned memory for MoeOffloadEngine: " + std::string(e.what()));
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] MoeOffloadEngine: Exception during initialization: " << e.what() << std::endl;
        throw;
    }
}

MoeOffloadEngine::~MoeOffloadEngine() {
    if (async_state_initialized_) {
        shutdown_.store(true);
        if (polling_thread_.joinable()) {
            polling_thread_.join();
        }
        cleanup_async_state();
    }

}

void MoeOffloadEngine::create_cpu_moe_layer(torch::Tensor w13_weight,
                                           torch::Tensor w2_weight,
                                           torch::Tensor w13_scale,
                                           torch::Tensor w2_scale,
                                           int layer_id) {
    TORCH_CHECK(layer_id >= 0, "layer_id must be >= 0");
    TORCH_CHECK(w13_weight.device().is_cpu(), "w13_weight must be CPU tensor");
    TORCH_CHECK(w2_weight.device().is_cpu(), "w2_weight must be CPU tensor");
    TORCH_CHECK(w13_scale.device().is_cpu(), "w13_scale must be CPU tensor");
    TORCH_CHECK(w2_scale.device().is_cpu(), "w2_scale must be CPU tensor");

    
    auto* w13_ptr = reinterpret_cast<float8_e4m3_t*>(w13_weight.data_ptr());
    auto* w2_ptr = reinterpret_cast<float8_e4m3_t*>(w2_weight.data_ptr());
    auto* w13_scale_ptr = w13_scale.data_ptr<float>();
    auto* w2_scale_ptr = w2_scale.data_ptr<float>();

    try {

        ForwardContext* ctx = forward_context_.get();
        
        auto result = moe_layers_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(layer_id),
            std::forward_as_tuple(w13_ptr, w2_ptr, w13_scale_ptr, w2_scale_ptr, layer_id, config_, ctx)
        );

        TORCH_CHECK(result.second, "Layer with id ", layer_id, " already exists");
    } catch (const std::bad_alloc& e) {
        std::cerr << "[ERROR] MoeOffloadEngine::create_cpu_moe_layer: Memory allocation failed (std::bad_alloc)" << std::endl;
        std::cerr << "  layer_id=" << layer_id << std::endl;
        std::cerr << "  This may be due to ForwardContext thread pool allocation failure" << std::endl;
        throw std::runtime_error("Failed to create CPU MoE layer: " + std::string(e.what()));
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] MoeOffloadEngine::create_cpu_moe_layer: Exception: " << e.what() << std::endl;
        std::cerr << "  layer_id=" << layer_id << std::endl;
        throw;
    }
}

Moe* MoeOffloadEngine::get_cpu_moe_layer(int layer_id) {
    auto it = moe_layers_.find(layer_id);
    return (it != moe_layers_.end()) ? &it->second : nullptr;
}

const Moe* MoeOffloadEngine::get_cpu_moe_layer(int layer_id) const {
    auto it = moe_layers_.find(layer_id);
    return (it != moe_layers_.end()) ? &it->second : nullptr;
}

void MoeOffloadEngine::call(int layer_idx, int batch_idx, int num_tokens) {
    std::string nvtx_msg = "MoeOffloadEngine_call_layer_" + std::to_string(layer_idx) +
                          "_batch_" + std::to_string(batch_idx) +
                          "_ntoks_" + std::to_string(num_tokens);
    nvtxRangePushA(nvtx_msg.c_str());

    auto layer = get_cpu_moe_layer(layer_idx);
    TORCH_CHECK(layer != nullptr, "Layer ", layer_idx, " not found");

    layer->forward(hidden_states_, topk_ids_, topk_weights_, output_, num_tokens);

    nvtxRangePop();
}

void MoeOffloadEngine::update_expert_cache(torch::Tensor w13_cache, torch::Tensor w2_cache,
                                          torch::Tensor w13_scale_cache, torch::Tensor w2_scale_cache,
                                          torch::Tensor map, int layer_id, int num_experts) {
    auto layer = get_cpu_moe_layer(layer_id);
    TORCH_CHECK(layer != nullptr, "Layer ", layer_id, " not found");
    layer->update_expert_cache(w13_cache, w2_cache, w13_scale_cache, w2_scale_cache, map, num_experts);
}

void MoeOffloadEngine::cpu_polling_loop() {
    std::cout << "cpu_polling_loop thread Start!" << std::endl;
    while (!shutdown_.load(std::memory_order_acquire)) {
        if (cpu_state_->gpu_signal == 1) {
            nvtxRangePushA("cpu_callback_func");
            call(cpu_state_->layer_idx, cpu_state_->batch_idx, cpu_state_->num_tokens);
            nvtxRangePop();

            std::atomic_thread_fence(std::memory_order_seq_cst);
            cpu_state_->callback_completed = 1;
            cpu_state_->complete_count += 1;
            cpu_state_->gpu_signal = 0;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}
