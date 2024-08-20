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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/cudaEvent.h"

#include <cuda_runtime_api.h>

#include <memory>

namespace tensorrt_llm::runtime
{

class CudaStream
{
public:
    //! Creates a new cuda stream on the current device. The stream will be destroyed in the destructor.
    //!
    //! \param flags Flags for stream creation. See ::cudaStreamCreateWithFlags for a list of valid flags that can be
    //! passed.
    //! \param priority Priority of the stream. Lower numbers represent higher priorities. See
    //! ::cudaDeviceGetStreamPriorityRange for more information about the meaningful stream priorities that can be
    //! passed.
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking, int priority = 0)
        : mDevice{tensorrt_llm::common::getDevice()}
    {
        cudaStream_t stream;
        TLLM_CUDA_CHECK(::cudaStreamCreateWithPriority(&stream, flags, priority));
        TLLM_LOG_TRACE("Created stream %p", stream);
        bool constexpr ownsStream{true};
        mStream = StreamPtr{stream, Deleter{ownsStream}};
    }

    //! Pass an existing cuda stream to this object.
    //!
    //! \param stream The stream to pass to this object.
    //! \param device The device on which the stream was created.
    //! \param ownsStream Whether this object owns the stream and destroys it in the destructor.
    explicit CudaStream(cudaStream_t stream, int device, bool ownsStream = true)
        : mDevice{device}
    {
        TLLM_CHECK_WITH_INFO(stream != nullptr, "stream is nullptr");
        mStream = StreamPtr{stream, Deleter{ownsStream}};
    }

    //! Construct with an existing cuda stream or the default stream by passing nullptr.
    explicit CudaStream(cudaStream_t stream)
        : CudaStream{stream, tensorrt_llm::common::getDevice(), false}
    {
    }

    //! Returns the device on which the stream was created.
    [[nodiscard]] int getDevice() const
    {
        return mDevice;
    }

    //! Returns the stream associated with this object.
    [[nodiscard]] cudaStream_t get() const
    {
        return mStream.get();
    }

    //! \brief Synchronizes the stream.
    void synchronize() const
    {
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(get()));
    }

    //! \brief Record an event on the stream.
    void record(CudaEvent::pointer event) const
    {
        TLLM_CUDA_CHECK(::cudaEventRecord(event, get()));
    }

    //! \brief Record an event on the stream.
    void record(CudaEvent const& event) const
    {
        record(event.get());
    }

    //! \brief Wait for an event.
    void wait(CudaEvent::pointer event) const
    {
        TLLM_CUDA_CHECK(::cudaStreamWaitEvent(get(), event));
    }

    //! \brief Wait for an event.
    void wait(CudaEvent const& event) const
    {
        wait(event.get());
    }

private:
    class Deleter
    {
    public:
        explicit Deleter(bool ownsStream)
            : mOwnsStream{ownsStream}
        {
        }

        explicit Deleter()
            : Deleter{true}
        {
        }

        constexpr void operator()(cudaStream_t stream) const
        {
            if (mOwnsStream && stream != nullptr)
            {
                TLLM_CUDA_CHECK(::cudaStreamDestroy(stream));
                TLLM_LOG_TRACE("Destroyed stream %p", stream);
            }
        }

    private:
        bool mOwnsStream;
    };

    using StreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, Deleter>;

    StreamPtr mStream;
    int mDevice{-1};
};

} // namespace tensorrt_llm::runtime
