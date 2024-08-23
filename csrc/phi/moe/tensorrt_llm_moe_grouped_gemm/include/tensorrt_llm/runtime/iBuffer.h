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

#include "tensorrt_llm/common/arrayView.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"

#include <NvInferRuntime.h>

#include <cstdint>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <memory>
#include <ostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace tensorrt_llm::runtime
{

enum class MemoryType : std::int32_t
{
    kGPU = 0,
    kCPU = 1,
    kPINNED = 2,
    kUVM = 3
};

template <MemoryType T>
struct MemoryTypeString
{
};

template <>
struct MemoryTypeString<MemoryType::kGPU>
{
    static auto constexpr value = "GPU";
};

template <>
struct MemoryTypeString<MemoryType::kCPU>
{
    static auto constexpr value = "CPU";
};

template <>
struct MemoryTypeString<MemoryType::kPINNED>
{
    static auto constexpr value = "PINNED";
};

template <>
struct MemoryTypeString<MemoryType::kUVM>
{
    static auto constexpr value = "UVM";
};

//! \brief For converting a TensorRT data type to a C++ data type.
template <nvinfer1::DataType kDataType, bool kIsUnsigned = false, bool kIsPointer = false>
struct DataTypeTraits
{
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kFLOAT>
{
    using type = float;
    static char constexpr name[] = "float";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kHALF>
{
    using type = half;
    static char constexpr name[] = "half";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT8>
{
    using type = std::int8_t;
    static char constexpr name[] = "int8";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT32>
{
    using type = std::int32_t;
    static char constexpr name[] = "int32";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT64>
{
    using type = std::int64_t;
    static char constexpr name[] = "int64";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT32, true>
{
    using type = std::uint32_t;
    static char constexpr name[] = "uint32";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT64, true>
{
    using type = std::uint64_t;
    static char constexpr name[] = "uint64";
    static auto constexpr size = sizeof(type);
};

template <bool kUnsigned>
struct DataTypeTraits<nvinfer1::DataType::kBOOL, kUnsigned>
{
    using type = bool;
    static char constexpr name[] = "bool";
    static auto constexpr size = sizeof(type);
};

template <bool kUnsigned>
struct DataTypeTraits<nvinfer1::DataType::kUINT8, kUnsigned>
{
    using type = std::uint8_t;
    static char constexpr name[] = "uint8";
    static auto constexpr size = sizeof(type);
};

#ifdef ENABLE_BF16
template <>
struct DataTypeTraits<nvinfer1::DataType::kBF16>
{
    using type = __nv_bfloat16;
    static char constexpr name[] = "bfloat16";
    static auto constexpr size = sizeof(type);
};
#endif

#ifdef ENABLE_FP8
template <>
struct DataTypeTraits<nvinfer1::DataType::kFP8>
{
    using type = __nv_fp8_e4m3;
    static char constexpr name[] = "fp8";
    static auto constexpr size = sizeof(type);
};
#endif

template <nvinfer1::DataType kDataType, bool kUnsigned>
struct DataTypeTraits<kDataType, kUnsigned, true>
{
    using type = typename DataTypeTraits<kDataType, kUnsigned, false>::type*;
    static char constexpr name[] = "*";
    static auto constexpr size = sizeof(type);
};

//! \brief A wrapper around `nvinfer1::DataType` that provides a support for pointer types.
class BufferDataType
{
public:
    constexpr BufferDataType( // NOLINT(*-explicit-constructor)
        nvinfer1::DataType dataType, bool _unsigned = false, bool pointer = false)
        : mDataType{dataType}
        , mUnsigned{_unsigned}
        , mPointer{pointer}
    {
    }

    static auto constexpr kTrtPointerType = nvinfer1::DataType::kINT64;

    constexpr operator nvinfer1::DataType() const noexcept // NOLINT(*-explicit-constructor)
    {
        return mPointer ? kTrtPointerType : mDataType;
    }

    [[nodiscard]] constexpr nvinfer1::DataType getDataType() const noexcept
    {
        return mDataType;
    }

    [[nodiscard]] constexpr bool isPointer() const noexcept
    {
        return mPointer;
    }

    [[nodiscard]] constexpr bool isUnsigned() const
    {
        switch (mDataType)
        {
        case nvinfer1::DataType::kBOOL: [[fallthrough]];
        case nvinfer1::DataType::kUINT8: return true;
        default: return mUnsigned;
        }
    }

    [[nodiscard]] constexpr std::size_t getSize() const noexcept
    {
        return tensorrt_llm::common::getDTypeSize(static_cast<nvinfer1::DataType>(*this));
    }

private:
    nvinfer1::DataType mDataType;
    bool mUnsigned;
    bool mPointer;
};

//! \brief For converting a C++ data type to a TensorRT data type.
template <typename T, bool = false>
struct TRTDataType
{
};

template <>
struct TRTDataType<float>
{
    static constexpr auto value = nvinfer1::DataType::kFLOAT;
};

template <>
struct TRTDataType<half>
{
    static constexpr auto value = nvinfer1::DataType::kHALF;
};

template <>
struct TRTDataType<std::int8_t>
{
    static constexpr auto value = nvinfer1::DataType::kINT8;
};

template <>
struct TRTDataType<std::int32_t>
{
    static constexpr auto value = nvinfer1::DataType::kINT32;
};

template <>
struct TRTDataType<std::uint32_t>
{
    static constexpr auto value = BufferDataType{nvinfer1::DataType::kINT32, true};
};

template <>
struct TRTDataType<std::int64_t>
{
    static constexpr auto value = nvinfer1::DataType::kINT64;
};

template <>
struct TRTDataType<std::uint64_t>
{
    static constexpr auto value = BufferDataType{nvinfer1::DataType::kINT64, true};
};

template <>
struct TRTDataType<bool>
{
    static constexpr auto value = nvinfer1::DataType::kBOOL;
};

template <>
struct TRTDataType<std::uint8_t>
{
    static constexpr auto value = nvinfer1::DataType::kUINT8;
};

#ifdef ENABLE_BF16
template <>
struct TRTDataType<__nv_bfloat16>
{
    static constexpr auto value = nvinfer1::DataType::kBF16;
};
#endif

#ifdef ENABLE_FP8
template <>
struct TRTDataType<__nv_fp8_e4m3>
{
    static constexpr auto value = nvinfer1::DataType::kFP8;
};
#endif

template <>
struct TRTDataType<kernels::KVCacheIndex>
{
    static constexpr auto value = TRTDataType<kernels::KVCacheIndex::UnderlyingType>::value;
};

template <>
struct TRTDataType<void*>
{
    static constexpr auto value = BufferDataType::kTrtPointerType;
};

template <typename T>
struct TRTDataType<T*>
{
private:
    static auto constexpr kUnderlyingType = BufferDataType{TRTDataType<T, false>::value};

public:
    static auto constexpr value = BufferDataType{kUnderlyingType.getDataType(), kUnderlyingType.isUnsigned(), true};
};

template <typename T>
using PointerElementType = typename std::remove_reference_t<T>::element_type;

template <typename T>
std::shared_ptr<std::remove_const_t<T>> constPointerCast(std::shared_ptr<T> const& ptr) noexcept
{
    return std::const_pointer_cast<std::remove_const_t<T>>(ptr);
}

template <typename T, typename D>
std::shared_ptr<std::remove_const_t<T>> constPointerCast(std::unique_ptr<T, D>&& ptr) noexcept
{
    return std::const_pointer_cast<std::remove_const_t<T>>(std::shared_ptr(std::move(ptr)));
}

class IBuffer
{
public:
    using UniquePtr = std::unique_ptr<IBuffer>;
    using SharedPtr = std::shared_ptr<IBuffer>;
    using UniqueConstPtr = std::unique_ptr<IBuffer const>;
    using SharedConstPtr = std::shared_ptr<IBuffer const>;
    using DataType = nvinfer1::DataType;

    //!
    //! \brief Returns a pointer to underlying array.
    //!
    [[nodiscard]] virtual void* data() = 0;

    //!
    //! \brief Returns a pointer to underlying array.
    //!
    [[nodiscard]] virtual void const* data() const = 0;

    //!
    //! \brief Returns a pointer to the underlying array at a given element index.
    //!
    [[nodiscard]] virtual void* data(std::size_t index)
    {
        auto* const dataPtr = this->data();
        return dataPtr ? static_cast<std::uint8_t*>(dataPtr) + toBytes(index) : nullptr;
    }

    //!
    //! \brief Returns a pointer to the underlying array at a given element index.
    //!
    [[nodiscard]] virtual void const* data(std::size_t index) const
    {
        auto const* const dataPtr = this->data();
        return dataPtr ? static_cast<std::uint8_t const*>(dataPtr) + toBytes(index) : nullptr;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    [[nodiscard]] virtual std::size_t getSize() const = 0;

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    [[nodiscard]] virtual std::size_t getSizeInBytes() const
    {
        return toBytes(getSize());
    }

    //!
    //! \brief Returns the capacity of the buffer.
    //!
    [[nodiscard]] virtual std::size_t getCapacity() const = 0;

    //!
    //! \brief Returns the data type of the buffer.
    //!
    [[nodiscard]] virtual DataType getDataType() const = 0;

    virtual char const* getDataTypeName() const;

    //!
    //! \brief Returns the memory type of the buffer.
    //!
    [[nodiscard]] virtual MemoryType getMemoryType() const = 0;

    virtual char const* getMemoryTypeName() const;

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    virtual void resize(std::size_t newSize) = 0;

    //!
    //! \brief Releases the buffer. It will be reset to nullptr.
    //!
    virtual void release() = 0;

    virtual ~IBuffer() = default;

    //!
    //! \brief Not allowed to copy.
    //!
    IBuffer(IBuffer const&) = delete;

    //!
    //! \brief Not allowed to copy.
    //!
    IBuffer& operator=(IBuffer const&) = delete;

    //!
    //! \brief Creates a sliced view on the underlying `buffer`. The view will have the same data type as `buffer`.
    //!
    //! \param buffer The buffer to view.
    //! \param offset The offset of the view.
    //! \param size The size of the view.
    //! \return A view on the `buffer`.
    //!
    static UniquePtr slice(SharedPtr buffer, std::size_t offset, std::size_t size);

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset, std::size_t size)
    {
        return IBuffer::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset, size);
    }

    static UniquePtr slice(SharedPtr buffer, std::size_t offset)
    {
        auto const size = buffer->getSize() - offset;
        return IBuffer::slice(std::move(buffer), offset, size);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset)
    {
        return IBuffer::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset);
    }

    //!
    //! \brief Returns a view on the underlying `tensor` which can be independently resized.
    //!
    //! \param tensor The tensor to view.
    //! \return A view on the `tensor`.
    //!
    static UniquePtr view(SharedPtr tensor)
    {
        auto constexpr offset = 0;
        return IBuffer::slice(std::move(tensor), offset);
    }

    //!
    //! \brief Returns a view on the underlying `tensor` with a different size.
    //!
    //! \param tensor The tensor to view.
    //! \param size The size of the view.
    //! \return A view on the `tensor`.
    //!
    static UniquePtr view(SharedPtr tensor, std::size_t size)
    {
        auto v = IBuffer::view(std::move(tensor));
        v->resize(size);
        return v;
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr view(TConstPtr&& tensor, std::size_t size)
    {
        return IBuffer::view(constPointerCast(std::forward<TConstPtr>(tensor)), size);
    }

    //!
    //! \brief Wraps the given `data` in an `IBuffer`. The `IBuffer` will not own the underlying `data` and cannot
    //! be resized beyond `capacity`.
    //!
    //! \param data The data to wrap.
    //! \param type The data type of the `data`.
    //! \param size The size of the buffer.
    //! \param capacity The capacity of the buffer.
    //! \return An `IBuffer`.
    static UniquePtr wrap(void* data, DataType type, std::size_t size, std::size_t capacity);

    static UniquePtr wrap(void* data, DataType type, std::size_t size)
    {
        return wrap(data, type, size, size);
    }

    template <typename T>
    static UniquePtr wrap(T* data, std::size_t size, std::size_t capacity)
    {
        return wrap(data, TRTDataType<T>::value, size, capacity);
    }

    template <typename T>
    static UniquePtr wrap(T* data, std::size_t size)
    {
        return wrap<T>(data, size, size);
    }

    template <typename T>
    static UniquePtr wrap(std::vector<T>& v)
    {
        return wrap<T>(v.data(), v.size(), v.capacity());
    }

    //!
    //! \brief Determine the memory type of a pointer.
    //!
    static MemoryType memoryType(void const* data);

protected:
    IBuffer() = default;

    //!
    //! \brief Returns an array index or size in bytes.
    //!
    [[nodiscard]] std::size_t toBytes(std::size_t size) const
    {
        return size * BufferDataType(getDataType()).getSize();
    }
};

template <typename T>
T const* bufferCast(IBuffer const& buffer)
{
    if (TRTDataType<typename std::remove_cv<T>::type>::value != buffer.getDataType())
    {
        throw std::bad_cast();
    }
    return static_cast<T const*>(buffer.data());
}

template <typename T>
T* bufferCast(IBuffer& buffer)
{
    if (TRTDataType<typename std::remove_cv<T>::type>::value != buffer.getDataType())
    {
        throw std::bad_cast();
    }
    return static_cast<T*>(buffer.data());
}

template <typename T>
class BufferRange : public tensorrt_llm::common::ArrayView<T>
{
public:
    using Base = tensorrt_llm::common::ArrayView<T>;
    using typename Base::size_type;

    BufferRange(T* data, size_type size)
        : Base{data, size}
    {
    }

    template <typename U = T, std::enable_if_t<!std::is_const_v<U>, bool> = true>
    explicit BufferRange(IBuffer& buffer)
        : BufferRange(bufferCast<U>(buffer), buffer.getSize())
    {
    }

    template <typename U = T, std::enable_if_t<std::is_const_v<U>, bool> = true>
    explicit BufferRange(IBuffer const& buffer)
        : BufferRange(bufferCast<U>(buffer), buffer.getSize())
    {
    }
};

//! \brief Utility function to print a buffer.
std::ostream& operator<<(std::ostream& output, IBuffer const& buffer);

} // namespace tensorrt_llm::runtime
