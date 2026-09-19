// OPT: min-heap of (score, physical_idx) for CTA-local online top-K.
// Same contract as indexer/cute_decode/cuda/topk_heap.cuh, but K is a
// template so 512/1024/2048 share one implementation.
#pragma once

#include <cstdint>

namespace dsa_opt {

template <int kTopK>
__device__ __forceinline__ void HeapSiftUp(float* scores, int32_t* indices, int i) {
    while (i > 0) {
        const int parent = (i - 1) >> 1;
        if (scores[parent] <= scores[i]) {
            break;
        }
        const float ts = scores[parent];
        const int32_t ti = indices[parent];
        scores[parent] = scores[i];
        indices[parent] = indices[i];
        scores[i] = ts;
        indices[i] = ti;
        i = parent;
    }
}

template <int kTopK>
__device__ __forceinline__ void HeapSiftDown(
    float* scores, int32_t* indices, int n, int i) {
    while (true) {
        const int left = (i << 1) + 1;
        const int right = left + 1;
        int smallest = i;
        if (left < n && scores[left] < scores[smallest]) {
            smallest = left;
        }
        if (right < n && scores[right] < scores[smallest]) {
            smallest = right;
        }
        if (smallest == i) {
            break;
        }
        const float ts = scores[i];
        const int32_t ti = indices[i];
        scores[i] = scores[smallest];
        indices[i] = indices[smallest];
        scores[smallest] = ts;
        indices[smallest] = ti;
        i = smallest;
    }
}

// Keep the K largest scores. Root is the current minimum of the window.
template <int kTopK>
__device__ __forceinline__ void HeapInsert(
    float* scores, int32_t* indices, int& n, float score, int32_t idx) {
    if (idx < 0) {
        return;
    }
    if (n < kTopK) {
        scores[n] = score;
        indices[n] = idx;
        HeapSiftUp<kTopK>(scores, indices, n);
        ++n;
        return;
    }
    if (score > scores[0]) {
        scores[0] = score;
        indices[0] = idx;
        HeapSiftDown<kTopK>(scores, indices, kTopK, 0);
    }
}

}  // namespace dsa_opt
