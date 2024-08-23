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

#include "tensorrt_llm/executor/types.h"

#include "tensorrt_llm/common/arrayView.h"
#include "tensorrt_llm/common/assert.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace tensorrt_llm::runtime
{
class ITensor;
class CudaStream;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::executor
{

class Tensor;

namespace detail
{
std::shared_ptr<runtime::ITensor> const& toITensor(Tensor const& tensor);
Tensor ofITensor(std::shared_ptr<runtime::ITensor> tensor);
using DimType64 = int64_t;

} // namespace detail

// A thin wrapper around span that supports constructions with an initializer list.
class Shape : public tensorrt_llm::common::ArrayView<detail::DimType64 const>
{
public:
    using Base = tensorrt_llm::common::ArrayView<detail::DimType64 const>;
    using DimType64 = typename std::remove_cv_t<Base::value_type>;

    Shape()
        : Base{nullptr, 0} {};

    Shape(DimType64 const* data, Base::size_type size)
        : Base{data, size}
    {
    }

    Shape(std::initializer_list<DimType64> dims) // NOLINT(*-explicit-constructor)
        : Base{dims.begin(), dims.size()}
    {
    }
};

class Tensor
{
public:
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;

    //! Allocate a cpu tensor with the given shape and data type.
    //!
    //! \param shape The shape of the tensor.
    //! \param dataType The data type of the tensor.
    static Tensor cpu(DataType dataType, Shape shape = {});

    template <typename T>
    static Tensor cpu(Shape shape = {})
    {
        return Tensor::cpu(getRuntimeType<T>(), shape);
    }

    [[nodiscard]] Tensor copyToCpu(Tensor::CudaStreamPtr stream = nullptr) const;

    //! Allocate a cpu tensor in pinned memory with the given shape and data type.
    //!
    //! \param shape The shape of the tensor.
    //! \param dataType The data type of the tensor.
    static Tensor pinned(DataType dataType, Shape shape = {});

    template <typename T>
    static Tensor pinned(Shape shape = {})
    {
        return Tensor::pinned(getRuntimeType<T>(), shape);
    }

    [[nodiscard]] Tensor copyToPinned(Tensor::CudaStreamPtr stream = nullptr) const;

    //! Allocate a cpu tensor in pooled pinned memory with the given shape and data type.
    //!
    //! \param shape The shape of the tensor.
    //! \param dataType The data type of the tensor.
    static Tensor pooledPinned(DataType dataType, Shape shape = {});

    template <typename T>
    static Tensor pooledPinned(Shape shape = {})
    {
        return Tensor::pooledPinned(getRuntimeType<T>(), shape);
    }

    [[nodiscard]] Tensor copyToPooledPinned(Tensor::CudaStreamPtr stream = nullptr) const;

    //! Allocate a tensor in managed memory (UVM) with the given shape and data type.
    //!
    //! \param shape The shape of the tensor.
    //! \param dataType The data type of the tensor.
    static Tensor managed(DataType dataType, Shape shape = {});

    template <typename T>
    static Tensor managed(Shape shape = {})
    {
        return Tensor::managed(getRuntimeType<T>(), shape);
    }

    [[nodiscard]] Tensor copyToManaged(Tensor::CudaStreamPtr stream = nullptr) const;

    //! Allocate a gpu tensor with the given shape and data type on a particular cuda stream.
    //!
    //! \param shape The shape of the tensor.
    //! \param stream Specifies the CUDA stream on which to allocate the tensor for GPU memory.
    //! \param dataType The data type of the tensor.
    static Tensor gpu(DataType dataType, CudaStreamPtr stream, Shape shape = {});

    template <typename T>
    static Tensor gpu(CudaStreamPtr stream, Shape shape = {})
    {
        return Tensor::gpu(getRuntimeType<T>(), std::move(stream), shape);
    }

    [[nodiscard]] Tensor copyToGpu(Tensor::CudaStreamPtr stream) const;

    //! Wrap a data pointer into a tensor without taking ownership.
    //!
    //! \param shape The shape of the tensor.
    //! \param dataType The data type of the tensor.
    //! \param stream Specifies the CUDA stream on which to allocate the tensor for GPU memory.
    static Tensor of(DataType dataType, void* data, Shape shape);

    //! Wrap a data pointer into a tensor without taking ownership.
    //!
    //! \param shape The shape of the tensor.
    //! \param dataType The data type of the tensor.
    //! \param stream Specifies the CUDA stream on which to allocate the tensor for GPU memory.
    template <typename T>
    static Tensor of(T* data, Shape shape)
    {
        return of(getRuntimeType<T>(), static_cast<void*>(data), shape);
    }

    //! Wrap any container into a tensor without taking ownership.
    //!
    //! \param shape The shape of the tensor.
    //! \param dataType The data type of the tensor.
    //! \param stream Specifies the CUDA stream on which to allocate the tensor for GPU memory.
    template <typename T>
    static Tensor of(T& data)
    {
        using DimType64 = Shape::DimType64;
        if constexpr (!std::is_same_v<DimType64, decltype(data.size())>)
        {
            TLLM_CHECK(data.size() <= std::numeric_limits<DimType64>::max());
        }
        return of(data.data(), {static_cast<Shape::DimType64 const>(data.size())});
    }

    Tensor() noexcept = default;

    ~Tensor() = default;

    Tensor(Tensor const& other) noexcept = default;

    Tensor(Tensor&& other) noexcept = default;

    Tensor& operator=(Tensor const& other) noexcept = default;

    Tensor& operator=(Tensor&& other) noexcept = default;

    //!
    //! \brief Returns a pointer to underlying array.
    //!
    [[nodiscard]] void* getData();

    //!
    //! \brief Returns a pointer to underlying array.
    //!
    [[nodiscard]] void const* getData() const;

    //!
    //! \brief Returns the data type of the buffer.
    //!
    [[nodiscard]] DataType getDataType() const;

    //!
    //! \brief Returns the memory type of the buffer.
    //!
    [[nodiscard]] MemoryType getMemoryType() const;

    //!
    //! \brief Returns the tensor dimensions.
    //!
    [[nodiscard]] Shape getShape() const;

    //!
    //! \brief Returns the number of elements in the tensor.
    //!
    [[nodiscard]] std::size_t getSize() const;

    //!
    //! \brief Returns the size of the tensor in bytes.
    //!
    [[nodiscard]] std::size_t getSizeInBytes() const;

    //!
    //! \brief Set the entire memory to zero.
    //!
    //! \param stream Must be a valid CUDA stream if the memory type is GPU.
    void setZero(CudaStreamPtr stream = nullptr);

    //!
    //! \brief Copy the data and shape from another tensor.
    //!
    //! \param other A tensor to copy from.
    //! \param stream Must be a valid CUDA stream if the memory type is GPU.
    void setFrom(Tensor const& other, CudaStreamPtr stream = nullptr);

    explicit operator bool() const
    {
        return static_cast<bool>(mTensor);
    }

    bool operator==(Tensor const& rhs) const
    {
        return mTensor == rhs.mTensor;
    }

    bool operator!=(Tensor const& rhs) const
    {
        return !(rhs == *this);
    }

private:
    using Impl = runtime::ITensor;
    explicit Tensor(std::shared_ptr<runtime::ITensor> tensor);

    template <typename T>
    static DataType getRuntimeType()
    {
        return TypeTraits<std::remove_cv_t<T>>::value;
    }

    [[nodiscard]] Tensor copyTo(std::shared_ptr<Impl> tensor, CudaStreamPtr stream) const;

    std::shared_ptr<Impl> mTensor;

    friend std::shared_ptr<runtime::ITensor> const& detail::toITensor(Tensor const& tensor);
    friend Tensor detail::ofITensor(std::shared_ptr<runtime::ITensor> tensor);
    friend class Serialization;
};

} // namespace tensorrt_llm::executor
