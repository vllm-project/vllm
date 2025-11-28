#import "metal_context.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <mutex>
#include <dlfcn.h>

namespace vllm {
namespace metal {

MetalContext::MetalContext()
    : device_(nullptr),
      library_(nullptr),
      command_queue_(nullptr),
      current_command_buffer_(nullptr),
      initialized_(false) {
}

MetalContext::~MetalContext() {
    // Release Metal objects
    if (command_queue_) {
        CFRelease(command_queue_);
    }
    if (library_) {
        CFRelease(library_);
    }
    if (device_) {
        CFRelease(device_);
    }

    // Release pipeline states
    for (auto& pair : pipeline_cache_) {
        if (pair.second) {
            CFRelease(pair.second);
        }
    }
}

bool MetalContext::initialize() {
    if (initialized_) {
        return true;
    }

    @autoreleasepool {
        // Get default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to create Metal device" << std::endl;
            return false;
        }
        device_ = (__bridge_retained void*)device;

        std::cout << "Metal device initialized: " << [device.name UTF8String] << std::endl;

        // Create command queue
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "Failed to create Metal command queue" << std::endl;
            return false;
        }
        command_queue_ = (__bridge_retained void*)commandQueue;

        // Load Metal library
        if (!load_library()) {
            std::cerr << "Failed to load Metal library" << std::endl;
            return false;
        }

        initialized_ = true;
        return true;
    }
}

// Static function to get library address
static void* get_library_marker() { return nullptr; }

bool MetalContext::load_library() {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

        NSError* error = nil;
        id<MTLLibrary> library = nil;

        // First try to find the library next to the Python extension
        // Get the path to the metallib file (should be in same directory as _metal_C.so)
        Dl_info info;
        if (dladdr((void*)get_library_marker, &info)) {
            std::string lib_path = info.dli_fname;
            size_t last_slash = lib_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                lib_path = lib_path.substr(0, last_slash + 1) + "vllm_metal_kernels.metallib";

                NSString* pathString = [NSString stringWithUTF8String:lib_path.c_str()];
                NSURL* libraryURL = [NSURL fileURLWithPath:pathString];

                library = [device newLibraryWithURL:libraryURL error:&error];
                if (library) {
                    std::cout << "Loaded Metal library from: " << lib_path << std::endl;
                } else if (error) {
                    std::cerr << "Failed to load Metal library from " << lib_path << ": "
                             << [[error localizedDescription] UTF8String] << std::endl;
                }
            }
        }

        // If that didn't work, try the bundle approach (for app bundles)
        if (!library) {
            NSBundle* mainBundle = [NSBundle mainBundle];
            NSURL* libraryURL = [mainBundle URLForResource:@"vllm_metal_kernels" withExtension:@"metallib"];

            if (libraryURL) {
                library = [device newLibraryWithURL:libraryURL error:&error];
                if (error) {
                    std::cerr << "Failed to load Metal library from bundle: "
                             << [[error localizedDescription] UTF8String] << std::endl;
                }
            }
        }

        // Last resort: try default library
        if (!library) {
            library = [device newDefaultLibrary];
            if (!library) {
                std::cerr << "Failed to load Metal library from all sources" << std::endl;
                return false;
            }
        }

        library_ = (__bridge_retained void*)library;

        std::cout << "Metal library loaded successfully" << std::endl;
        std::cout << "Available function names:" << std::endl;
        NSArray<NSString*>* functionNames = [library functionNames];
        for (NSString* name in functionNames) {
            if ([name containsString:@"paged_attention"] ||
                [name containsString:@"reshape_and_cache"] ||
                [name containsString:@"copy_blocks"] ||
                [name containsString:@"swap_blocks"]) {
                std::cout << "  - " << [name UTF8String] << std::endl;
            }
        }

        return true;
    }
}

void* MetalContext::create_pipeline_state(const std::string& kernel_name) {
    @autoreleasepool {
        id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_;
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

        NSString* nsKernelName = [NSString stringWithUTF8String:kernel_name.c_str()];
        id<MTLFunction> function = [library newFunctionWithName:nsKernelName];

        if (!function) {
            std::cerr << "Failed to find Metal function: " << kernel_name << std::endl;
            return nullptr;
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pipelineState =
            [device newComputePipelineStateWithFunction:function error:&error];

        if (error || !pipelineState) {
            std::cerr << "Failed to create pipeline state for: " << kernel_name << std::endl;
            if (error) {
                std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
            }
            return nullptr;
        }

        return (__bridge_retained void*)pipelineState;
    }
}

void* MetalContext::get_pipeline_state(const std::string& kernel_name) {
    PipelineKey key{kernel_name};

    auto it = pipeline_cache_.find(key);
    if (it != pipeline_cache_.end()) {
        return it->second;
    }

    // Create new pipeline state
    void* pipeline_state = create_pipeline_state(kernel_name);
    if (pipeline_state) {
        pipeline_cache_[key] = pipeline_state;
    }

    return pipeline_state;
}

void* MetalContext::create_command_buffer() {
    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)command_queue_;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        return (__bridge_retained void*)commandBuffer;
    }
}

void* MetalContext::create_compute_encoder(void* command_buffer) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)command_buffer;
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        return (__bridge_retained void*)encoder;
    }
}

void MetalContext::set_pipeline_state(void* encoder, void* pipeline_state) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> cmdEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
        id<MTLComputePipelineState> pipelineState = (__bridge id<MTLComputePipelineState>)pipeline_state;
        [cmdEncoder setComputePipelineState:pipelineState];
    }
}

void MetalContext::set_buffer(void* encoder, const void* buffer, size_t offset, uint32_t index) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> cmdEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        [cmdEncoder setBuffer:mtlBuffer offset:offset atIndex:index];
    }
}

void MetalContext::set_bytes(void* encoder, const void* data, size_t length, uint32_t index) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> cmdEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
        [cmdEncoder setBytes:data length:length atIndex:index];
    }
}

void MetalContext::set_threadgroup_memory_length(void* encoder, size_t length, uint32_t index) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> cmdEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
        [cmdEncoder setThreadgroupMemoryLength:length atIndex:index];
    }
}

void MetalContext::dispatch_threadgroups(void* encoder,
                                        uint32_t threadgroups_x,
                                        uint32_t threadgroups_y,
                                        uint32_t threadgroups_z,
                                        uint32_t threads_per_group_x,
                                        uint32_t threads_per_group_y,
                                        uint32_t threads_per_group_z) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> cmdEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;

        MTLSize gridSize = MTLSizeMake(threadgroups_x, threadgroups_y, threadgroups_z);
        MTLSize threadgroupSize = MTLSizeMake(threads_per_group_x,
                                             threads_per_group_y,
                                             threads_per_group_z);

        [cmdEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    }
}

void MetalContext::end_encoding(void* encoder) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> cmdEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
        [cmdEncoder endEncoding];
        CFRelease(encoder);
    }
}

void MetalContext::commit(void* command_buffer) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)command_buffer;
        [cmdBuffer commit];
    }
}

void MetalContext::wait_until_completed(void* command_buffer) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)command_buffer;
        [cmdBuffer waitUntilCompleted];

        // Check for errors
        if (cmdBuffer.error) {
            std::cerr << "Metal command buffer error: "
                     << [[cmdBuffer.error localizedDescription] UTF8String] << std::endl;
        }

        CFRelease(command_buffer);
    }
}

void MetalContext::synchronize() {
    // Create a temporary command buffer and wait for completion
    void* cmd_buffer = create_command_buffer();
    commit(cmd_buffer);
    wait_until_completed(cmd_buffer);
}

std::string MetalContext::get_device_name() const {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
        return std::string([device.name UTF8String]);
    }
}

uint32_t MetalContext::get_max_threads_per_threadgroup() const {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
        return static_cast<uint32_t>(device.maxThreadsPerThreadgroup.width);
    }
}

bool MetalContext::supports_family(int family) const {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
        return [device supportsFamily:(MTLGPUFamily)family];
    }
}

// Global singleton
static MetalContext* g_metal_context = nullptr;
static std::mutex g_context_mutex;

MetalContext* get_metal_context() {
    std::lock_guard<std::mutex> lock(g_context_mutex);

    if (!g_metal_context) {
        g_metal_context = new MetalContext();
        if (!g_metal_context->initialize()) {
            delete g_metal_context;
            g_metal_context = nullptr;
            throw std::runtime_error("Failed to initialize Metal context");
        }
    }

    return g_metal_context;
}

} // namespace metal
} // namespace vllm
