#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

// Forward declarations for Objective-C types
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLLibrary;
@protocol MTLComputePipelineState;
@protocol MTLCommandQueue;
@protocol MTLCommandBuffer;
@protocol MTLComputeCommandEncoder;
@protocol MTLBuffer;
#else
typedef void MTLDevice;
typedef void MTLLibrary;
typedef void MTLComputePipelineState;
typedef void MTLCommandQueue;
typedef void MTLCommandBuffer;
typedef void MTLComputeCommandEncoder;
typedef void MTLBuffer;
#endif

namespace vllm {
namespace metal {

// Pipeline identifier for caching compiled kernels
struct PipelineKey {
    std::string kernel_name;

    bool operator==(const PipelineKey& other) const {
        return kernel_name == other.kernel_name;
    }
};

} // namespace metal
} // namespace vllm

// Hash function for PipelineKey
namespace std {
template <>
struct hash<vllm::metal::PipelineKey> {
    size_t operator()(const vllm::metal::PipelineKey& key) const {
        return hash<string>()(key.kernel_name);
    }
};
} // namespace std

namespace vllm {
namespace metal {

// Metal context manages device, library, and pipeline states
class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    // Initialize Metal device and load kernels
    bool initialize();

    // Get Metal device
    void* get_device() const { return device_; }

    // Get command queue
    void* get_command_queue() const { return command_queue_; }

    // Create command buffer
    void* create_command_buffer();

    // Create compute command encoder
    void* create_compute_encoder(void* command_buffer);

    // Get or create pipeline state for a kernel
    void* get_pipeline_state(const std::string& kernel_name);

    // Set compute pipeline state
    void set_pipeline_state(void* encoder, void* pipeline_state);

    // Set buffer argument
    void set_buffer(void* encoder, const void* buffer, size_t offset, uint32_t index);

    // Set bytes argument (for small constant data)
    void set_bytes(void* encoder, const void* data, size_t length, uint32_t index);

    // Set threadgroup memory length
    void set_threadgroup_memory_length(void* encoder, size_t length, uint32_t index);

    // Dispatch threadgroups
    void dispatch_threadgroups(void* encoder,
                              uint32_t threadgroups_x,
                              uint32_t threadgroups_y,
                              uint32_t threadgroups_z,
                              uint32_t threads_per_group_x,
                              uint32_t threads_per_group_y,
                              uint32_t threads_per_group_z);

    // End encoding
    void end_encoding(void* encoder);

    // Commit command buffer
    void commit(void* command_buffer);

    // Wait for command buffer completion
    void wait_until_completed(void* command_buffer);

    // Synchronize (commit and wait)
    void synchronize();

    // Get device name
    std::string get_device_name() const;

    // Get max threads per threadgroup
    uint32_t get_max_threads_per_threadgroup() const;

    // Check if device supports certain features
    bool supports_family(int family) const;

private:
    // Load Metal library from embedded or file
    bool load_library();

    // Create pipeline state for a kernel
    void* create_pipeline_state(const std::string& kernel_name);

    // Metal objects (using void* to avoid Objective-C in header)
    void* device_;           // id<MTLDevice>
    void* library_;          // id<MTLLibrary>
    void* command_queue_;    // id<MTLCommandQueue>

    // Pipeline cache
    std::unordered_map<PipelineKey, void*> pipeline_cache_;

    // Current command buffer for synchronous operations
    void* current_command_buffer_;

    // Initialization flag
    bool initialized_;
};

// Global Metal context singleton
MetalContext* get_metal_context();

} // namespace metal
} // namespace vllm
