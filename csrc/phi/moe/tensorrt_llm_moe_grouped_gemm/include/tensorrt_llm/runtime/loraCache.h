/*loraCac
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraCachePageManagerConfig.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <deque>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace tensorrt_llm::runtime
{

class LoraExpectedException : public std::runtime_error
{
public:
    explicit LoraExpectedException(std::string const& msg);
    ~LoraExpectedException() noexcept override;
};

class LoraCacheFullException : public LoraExpectedException
{
public:
    explicit LoraCacheFullException(std::string const& msg);
    ~LoraCacheFullException() noexcept override;
};

/**
 * Holds memory of lora cache pages, and manages allocation and freeing of whole pages.
 * Memory is pre-allocated either on the host or device
 *
 * Note that this class is not thread safe
 */
class LoraCachePageManager
{
public:
    using TensorPtr = ITensor::SharedPtr;

    /**
     * \param[in] config: a LoraCachePageManagerConfig
     * \param[in] bufferManager: a Buffermanager used to allocate page blocks
     */
    LoraCachePageManager(LoraCachePageManagerConfig const& config, BufferManager const& bufferManager);

    /**
     * \brief claim pages
     *
     * \param[in] numPages number of pages to claim
     * \returns a tuple, where the first values is a boolean indicating whether pages were claimed.  If the first value
     * is true the second value will have a list of pageIds
     */
    [[nodiscard]] std::optional<std::vector<std::size_t>> claimPages(SizeType32 numPages);

    /**
     * \brief get number of available (free) pages in manager
     *
     * \returns number of free pages in manager
     */
    [[nodiscard]] SizeType32 numAvailablePages() const;

    /**
     * \brief release given pages
     *
     * \param[in] pages: list of pages to release (free)
     */
    void releasePages(std::vector<std::size_t> const& pages);

    /**
     * \brief return pointer to given page block
     *
     * \param[in] blockIdx;
     * \returns -- pointer to page block
     */
    [[nodiscard]] ITensor::SharedConstPtr blockPtr(SizeType32 blockIdx) const;

    /**
     * \brief return pointer to given page
     *
     * \param[in] pageIdx:
     * \returns -- const pointer to page
     */
    [[nodiscard]] ITensor::SharedConstPtr pagePtr(std::size_t pageIdx) const;

    /**
     * \brief return pointer to given page
     *
     * \param[in] pageIdx:
     * \returns -- mutable pointer to page
     */
    [[nodiscard]] ITensor::SharedPtr mutablePagePtr(std::size_t pageIdx);

private:
    std::vector<TensorPtr> mPageBlocks;
    std::deque<std::size_t> mFreePageIds;
    std::vector<std::uint8_t> mIsPageFree;
    LoraCachePageManagerConfig const mConfig;

    void initialize(BufferManager const& bufferManager);
};

/**
 * LoraCache
 *
 * Caches LoRA weights with LRU eviction policy.
 *
 * Tasks put in the cache are marked in progress and can not be evicted, until they are marked done.
 *
 * A cache page holds a optimally sized LoRA. A page is of size [numSlots x pageWidth]
 * An optimally size LoRA is on that has the configured optimalAdapterSize.
 *
 * Conceptually a slot corresponds to a r=1, 1-layer, 1-module set of in/out weights.
 * Page width is set to the number of weights in smallest module.
 *
 * The number of slots per page is then ceilDiv(num weights in optimally sized LoRA, num weights in smallest module)
 *
 * Cache pages are allocated on one or more blocks
 */
class LoraCache
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TaskIdType = std::uint64_t;

    /**
     * Contains information on a single layer / module.
     * A list of these configs is associated with each task and can be used to populate runtime tensors.
     */
    struct TaskLayerModuleConfig
    {
        std::size_t pageId;
        SizeType32 slotIdx;
        SizeType32 inSize;  // adapterSize * inDim
        SizeType32 outSize; // outDim * adapterSize
        SizeType32 moduleId;
        SizeType32 layerId;
        SizeType32 adapterSize;
        SizeType32 numSlots; // number of slots used by this layer / module. Used to avoid copying extra data from page.

        // pointer to inWeights cast to an int64_t
        std::int64_t weightsInPointer;
        // pointer to out weights cast to an int64_t
        std::int64_t weightsOutPointer;

        std::string toString() const;

        bool operator==(LoraCache::TaskLayerModuleConfig const& o) const;
    };

    using TaskLayerModuleConfigListPtr = std::shared_ptr<std::vector<TaskLayerModuleConfig>>;

    /**
     * param[in] pageManagerConfig: a LoraCachePageManagerConfig
     * param[in] modelConfig: a ModelConfig
     * param[in] worldConfig: a WorldConfig
     * param[in] bufferManager: a BufferManager only used to allocate page blocks
     */
    LoraCache(LoraCachePageManagerConfig const& pageManagerConfig, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig, BufferManager const& bufferManager);

    /**
     * \brief put a task in the cache, and claim pages for it, and optionally load task weights.
     *
     * \param[in] taskId: the task id
     * \param[in] weights: lora weights tensor
     * \param[in] config: lora config tensor
     * \param[in] load: if true load weights before returning, otherwise do not
     */
    void put(TaskIdType taskId, TensorPtr weights, TensorPtr config, bool load = true);

    /**
     * \brief load task weights.  This method must be called after put.  It is designed to be called asynchronously
     * after put returns with load = false
     *
     * \param[in] taslId: the task id
     * \param[in] weights: lora weights tensor
     * \param[in] config: lora config tensor
     */
    void loadWeights(TaskIdType taskId, TensorPtr weights, TensorPtr config);

    /**
     * \param[in] taskId: the task id
     * \returns -- true if task is loaded (weights are in place) and false otherwise
     */
    [[nodiscard]] inline bool isLoaded(TaskIdType taskId) const
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        return kVALUE_STATUS_LOADED == getStatus(taskId);
    }

    /**
     * \param[in] taskId: the task id
     * \returns -- true if task is marked done and can be evicted
     */
    [[nodiscard]] bool isDone(TaskIdType taskId) const;

    /**
     * \param[in] taskId: the task id
     * \returns -- true if task is in the cache (not necessarily loaded) and false otherwise
     */
    [[nodiscard]] inline bool has(TaskIdType taskId) const
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        return kVALUE_STATUS_MISSING != getStatus(taskId);
    }

    /**
     * \param[in] taskId: the task id
     * \returns -- list of Value objects with pointers to task weights
     */
    [[nodiscard]] std::shared_ptr<std::vector<TaskLayerModuleConfig>> get(TaskIdType taskId);

    /**
     * \brief bump task and make it the most recently used
     *
     * \param[in] taskId: the task id
     */
    void bump(TaskIdType taskId);

    /**
     * \brief mark task done meaning it can be evicted
     * \param[in] taskId: the task id
     */
    void markTaskDone(TaskIdType taskId);

    /**
     * \brief mark all tasks in cache done
     */
    void markAllDone();

    /**
     * \param[in] taskId: the taskid
     * \returns -- number of pages needed to store the given task
     */
    [[nodiscard]] SizeType32 determineNumPages(TaskIdType taskId) const;

    /**
     * \param[in] config: lora config tensor
     * \returns -- number of pages needed to store the task configured with config tensor
     */
    [[nodiscard]] SizeType32 determineNumPages(TensorPtr config) const;

    /**
     * \param[in] config: a lora config tensor
     * \returns -- true in task fits in cache false otherwise
     */
    [[nodiscard]] bool fits(TensorPtr config) const;

    /**
     * \brief copy task to another cache. Caches must have the same page size.
     * \param[in] taskId: the task id to copy
     * \param[in] otherCache: the LoraCache to move the task to
     * \param[in] markDone: mark the copied task done as it's copied
     */
    void copyTask(TaskIdType taskId, LoraCache& deviceCache, bool markDone = false);

    /**
     * \returns -- total number of pages allocated to cache (used or not)
     */
    [[nodiscard]] SizeType32 getNumPages() const;

    /**
     * \param[in] pageId: the page id
     * \returns -- const pointer to page
     */
    [[nodiscard]] ITensor::SharedConstPtr getPagePtr(size_t pageId) const;

    /**
     * \brief Copy task weights to cache pages.
     * \param[in] weights: task weights
     * \param[in] config: task config tensor
     * \param[in] modelConfig: a ModelConfig
     * \param[in] worldConfig: a WorldConfig
     * \param[in] modelIdToModel: map from lora module id to LoraModule
     * \param[in] manager: a BufferManager the manager to use to perform the copies
     * \param[out] pages: list of page tensors to copy weights to
     * \param[in] pageIds: page ids for the pages
     * \returns -- list of cache Values objects
     */
    static std::vector<LoraCache::TaskLayerModuleConfig> copyToPages(TensorPtr weights, TensorPtr config,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::unordered_map<SizeType32, LoraModule> moduleIdToModel, BufferManager const& manager,
        std::vector<TensorPtr> const& pages, std::vector<std::size_t> const& pageIds);

    /**
     * \brief splits second dim of input into tpSize parts and writes the tpRank split to output
     * \param[out] output: output tensor
     * \param[in] input: input tensor
     * \param[in] tpSize: number of splits
     * \param[in] tpRank: the split to write to output
     */
    static void splitTransposeCpu(ITensor& output, ITensor const& input, SizeType32 tpSize, SizeType32 tpRank);

private:
    /**
     * \brief Holds configuration and state for a single task
     */
    struct TaskValue
    {
        // pageIds holding this tasks weights
        std::vector<std::size_t> pageIds;
        // locations of weights in pages
        TaskLayerModuleConfigListPtr configs;
        // ordered location of this value in either mDoneTasks or mInProgressTasks
        std::list<TaskIdType>::iterator it;

        /* indicates if the task is inProgress (in mInProgress list, not evictable)
         * if inProgress=false the task is in mDoneTasks list.
         */
        bool inProgress;
        /*
         * indicates the weights have been copied into the cache.
         * If inProgress=true and loaded=false we are in the middle of adding the task to the cache.
         * We cannot evict or copyTask tasks in this state.
         */
        bool loaded;
        /**
         * Marks a task a done.  This is used to mark a task as done during loading.
         * if done=true at the end of loading (end of put, loadweights, or copyTask) the task will be marked as done
         */
        bool done;
        /**
         * Indicates weights are loading either in put or loadWeights
         * This is used to block concurrent loadWeights calls for the same task.
         */
        bool loadInProgress;

        TaskValue() = delete;
        ~TaskValue() = default;

        TaskValue(std::vector<std::size_t> const& pageIds, TaskLayerModuleConfigListPtr const& configs,
            std::list<TaskIdType>::iterator it, bool inProgress, bool loaded, bool done, bool loadInProgress = false)
            : pageIds(pageIds)
            , configs(configs)
            , it(it)
            , inProgress(inProgress)
            , loaded(loaded)
            , done(done)
            , loadInProgress(loadInProgress)
        {
        }

        TaskValue(TaskValue&& o) noexcept
        {
            std::swap(pageIds, o.pageIds);
            std::swap(configs, o.configs);
            std::swap(it, o.it);
            std::swap(inProgress, o.inProgress);
            std::swap(loaded, o.loaded);
            std::swap(done, o.done);
            std::swap(loadInProgress, o.loadInProgress);
        }

        TaskValue& operator=(TaskValue&& o)
        {
            std::swap(pageIds, o.pageIds);
            std::swap(configs, o.configs);
            std::swap(it, o.it);
            std::swap(inProgress, o.inProgress);
            std::swap(loaded, o.loaded);
            std::swap(done, o.done);
            std::swap(loadInProgress, o.loadInProgress);
            return *this;
        }
    };

    using TaskValuePtr = std::shared_ptr<TaskValue>;

    enum ValueStatus
    {
        // task is not in the cache (inProgress or Done)
        kVALUE_STATUS_MISSING = 0,
        // task is in cache, but weights are not
        kVALUE_STATUS_PROCESSING = 1,
        // task and weights are in the cache
        kVALUE_STATUS_LOADED = 2,
    };

    LoraCachePageManagerConfig mPageManagerConfig;
    ModelConfig mModelConfig;
    WorldConfig mWorldConfig;

    // Protects mCachePageManager
    mutable std::mutex mPagesMutex;
    std::unique_ptr<LoraCachePageManager> mCachePageManager;

    /*
     * Protects mutations of mCacheMap, mInProgressTasks and mDoneTasks
     * And the state booleans in TaskValue (ie inProgress, loaded, done, loadInProgress)
     * mCacheMutex does not protect other values within a TaskValue (ie weights, pageIds, etc)
     */
    mutable std::mutex mCacheMutex;
    std::unordered_map<TaskIdType, TaskValuePtr> mCacheMap;
    std::list<TaskIdType> mInProgressTasks;
    std::list<TaskIdType> mDoneTasks;

    std::vector<std::unique_ptr<BufferManager>> mDeviceBufferManagers;
    std::unique_ptr<BufferManager> mBufferManager;

    std::unordered_map<SizeType32, LoraModule> mModuleIdToModule;

    template <typename T>
    static void splitTransposeCpuInner(ITensor& output, ITensor const& input, SizeType32 tpSize, SizeType32 tpRank);

    void loadWeights(TaskValue& cacheValue, TensorPtr weights, TensorPtr config);
    void bumpTaskInProgress(TaskIdType taskId);
    [[nodiscard]] ValueStatus getStatus(TaskIdType taskId) const;

    /**
     * \brief claim numPages, evicting tasks if needed
     * \param[in] numPages: number of pages to claim
     * \returns -- list of page ids
     * \throws std::runtime_error if all pages cannot be claimed
     */
    [[nodiscard]] std::vector<std::size_t> claimPagesWithEvict(SizeType32 numPages);

    /**
     * Internal helper method used inside copyTask.  Not thread safe on its own
     */
    std::map<size_t, std::pair<size_t, SizeType32>> copyTaskMapPages(TaskValue& targetTaskValue,
        TaskValue const& sourceTaskValue, std::vector<size_t> const& targetPageIds, LoraCache const& targetCache);
};

std::string to_string(LoraCache::TaskLayerModuleConfig const& v);

std::ostream& operator<<(std::ostream& os, LoraCache::TaskLayerModuleConfig const& v);

} // namespace tensorrt_llm::runtime
