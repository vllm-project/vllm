/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <NvInferRuntime.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

class BufferManagerTest;

namespace tensorrt_llm::runtime
{
//! \brief A helper class for managing memory on host and device.
class BufferManager
{
public:
    using IBufferPtr = IBuffer::UniquePtr;

    using ITensorPtr = ITensor::UniquePtr;

    using CudaStreamPtr = std::shared_ptr<CudaStream>;

    //! \brief Construct a BufferManager.
    //!
    //! \param[in] cudaStream The cuda stream to use for all operations on GPU (allocation, de-allocation, copying,
    //! etc.).
    explicit BufferManager(CudaStreamPtr stream, bool trimPool = false);

    //! \brief Destructor.
    ~BufferManager()
    {
        if (mTrimPool)
        {
            memoryPoolTrimTo(0);
        }
    }

    static auto constexpr kBYTE_TYPE = nvinfer1::DataType::kUINT8;

    //! \brief Allocates an `IBuffer` of the given size on the GPU, using cudaMallocAsync.
    [[nodiscard]] IBufferPtr gpu(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE) const;

    //! \brief Allocates an `ITensor` of the given dimensions on the GPU, using cudaMallocAsync.
    [[nodiscard]] ITensorPtr gpu(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE) const;

    //! \brief Allocates an `IBuffer` of the given size on the GPU, using cudaMalloc.
    [[nodiscard]] static IBufferPtr gpuSync(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `ITensor` of the given dimensions on the GPU, using cudaMalloc.
    [[nodiscard]] static ITensorPtr gpuSync(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `IBuffer` of the given size on the CPU.
    [[nodiscard]] static IBufferPtr cpu(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `ITensor` of the given dimensions on the CPU.
    [[nodiscard]] static ITensorPtr cpu(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates a pinned `IBuffer` of the given size on the CPU.
    [[nodiscard]] static IBufferPtr pinned(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates a pinned `ITensor` of the given dimensions on the CPU.
    [[nodiscard]] static ITensorPtr pinned(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates a pinned `IBuffer` of the given size on the CPU in the default memory pool.
    [[nodiscard]] static IBufferPtr pinnedPool(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates a pinned `ITensor` of the given dimensions on the CPU in the default memory pool.
    [[nodiscard]] static ITensorPtr pinnedPool(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `IBuffer` of the given size in UVM.
    [[nodiscard]] static IBufferPtr managed(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `ITensor` of the given dimensions in UVM.
    [[nodiscard]] static ITensorPtr managed(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `IBuffer` of the given size and memory type.
    [[nodiscard]] IBufferPtr allocate(
        MemoryType memoryType, std::size_t size, nvinfer1::DataType type = kBYTE_TYPE) const;

    //! \brief Allocates an `ITensor` of the given dimensions and memory type.
    [[nodiscard]] ITensorPtr allocate(
        MemoryType memoryType, nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE) const;

    //! \brief Create an empty `IBuffer` of the given memory type. It may be resized later.
    [[nodiscard]] IBufferPtr emptyBuffer(MemoryType memoryType, nvinfer1::DataType type = kBYTE_TYPE) const
    {
        return allocate(memoryType, 0, type);
    }

    //! \brief Create an empty `ITensor` of the given memory type. It may be reshaped later.
    [[nodiscard]] ITensorPtr emptyTensor(MemoryType memoryType, nvinfer1::DataType type = kBYTE_TYPE) const
    {
        return allocate(memoryType, ITensor::makeShape({}), type);
    }

    //! \brief Set the contents of the given `buffer` to value.
    void setMem(IBuffer& buffer, int32_t value) const;

    //! \brief Set the contents of the given `buffer` to zero.
    void setZero(IBuffer& buffer) const;

    //! \brief Copy `src` to `dst`.
    void copy(void const* src, IBuffer& dst, MemoryType srcType) const;

    //! \brief Copy `src` to `dst`.
    void copy(IBuffer const& src, void* dst, MemoryType dstType) const;

    //! \brief Copy `src` to `dst`.
    void copy(void const* src, IBuffer& dst) const
    {
        return copy(src, dst, IBuffer::memoryType(src));
    }

    //! \brief Copy `src` to `dst`.
    void copy(IBuffer const& src, void* dst) const
    {
        return copy(src, dst, IBuffer::memoryType(dst));
    }

    //! \brief Copy `src` to `dst`.
    void copy(IBuffer const& src, IBuffer& dst) const;

    //! \brief Copy `src` into a new `IBuffer` with a potentially different memory type.
    [[nodiscard]] IBufferPtr copyFrom(IBuffer const& src, MemoryType memoryType) const;

    //! \brief Copy `src` into a new `ITensor` with a potentially different memory type.
    [[nodiscard]] ITensorPtr copyFrom(ITensor const& src, MemoryType memoryType) const;

    //! \brief Copy `src` into a new `IBuffer` with a potentially different memory type.
    template <typename T>
    [[nodiscard]] IBufferPtr copyFrom(std::vector<T> const& src, MemoryType memoryType) const
    {
        auto buffer = allocate(memoryType, src.size(), TRTDataType<std::remove_cv_t<T>>::value);
        copy(src.data(), *buffer);
        return buffer;
    }

    //! \brief Copy `src` into a new `ITensor` with a potentially different memory type.
    template <typename T>
    [[nodiscard]] ITensorPtr copyFrom(T* src, nvinfer1::Dims dims, MemoryType memoryType) const
    {
        auto buffer = allocate(memoryType, dims, TRTDataType<std::remove_cv_t<T>>::value);
        copy(src, *buffer);
        return buffer;
    }

    //! \brief Copy `src` into a new `ITensor` with a potentially different memory type.
    template <typename T>
    [[nodiscard]] ITensorPtr copyFrom(std::vector<T> const& src, nvinfer1::Dims dims, MemoryType memoryType) const
    {
        TLLM_CHECK_WITH_INFO(src.size() == ITensor::volumeNonNegative(dims),
            common::fmtstr("[TensorRT-LLM][ERROR] Incompatible size %lu and dims %s", src.size(),
                ITensor::toString(dims).c_str()));
        return copyFrom(src.data(), dims, memoryType);
    }

    //! \brief Get the underlying cuda stream.
    [[nodiscard]] CudaStream const& getStream() const;

    //! \brief The current size of the memory reserved by the memory pool.
    [[nodiscard]] std::size_t memoryPoolReserved() const;

    //! \brief The current size of the memory used by the memory pool.
    [[nodiscard]] std::size_t memoryPoolUsed() const;

    //! \brief The current size of the memory free in the memory pool.
    [[nodiscard]] std::size_t memoryPoolFree() const;

    //! \brief Try to trim the memory reserved by the pool to `size` bytes. This synchronizes implicitly with the
    //! stream.
    void memoryPoolTrimTo(std::size_t size);

private:
    friend class ::BufferManagerTest;

    void static initMemoryPool(int device);

    std::size_t static memoryPoolReserved(int device);

    std::size_t static memoryPoolUsed(int device);

    std::size_t static memoryPoolFree(int device)
    {
        return memoryPoolReserved(device) - memoryPoolUsed(device);
    }

    void static memoryPoolTrimTo(int device, std::size_t size);

    CudaStreamPtr mStream;
    bool const mTrimPool;
};

} // namespace tensorrt_llm::runtime
