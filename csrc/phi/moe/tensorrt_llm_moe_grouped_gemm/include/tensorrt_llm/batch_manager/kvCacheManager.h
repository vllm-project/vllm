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

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/llmRequest.h" // TODO forward declare
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace std
{

// Implement std::hash function object for vector<TokenIdType>.
// This allows us to use unordered_map with vector<TokenIdType> as key.
// Based on https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933

template <>
struct hash<vector<int32_t>>
{
    size_t operator()(vector<int32_t> const& vec) const noexcept
    {
        size_t seed = vec.size();
        for (auto x : vec)
        {
            uint32_t y = static_cast<uint32_t>(x);
            y = ((y >> 16) ^ y) * 0x45d9f3b;
            y = ((y >> 16) ^ y) * 0x45d9f3b;
            y = (y >> 16) ^ y;
            seed ^= y + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

} // namespace std

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class KVCacheBlock;

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using BlockPtr = std::shared_ptr<KVCacheBlock>;
using FreeBlocksQueue = std::list<BlockPtr>;
using NextBlockMap = std::unordered_map<VecTokens, BlockPtr>;

struct KvCacheStats
{
    SizeType32 maxNumBlocks;
    SizeType32 freeNumBlocks;
    SizeType32 usedNumBlocks;
    SizeType32 toksPerBlock;
};

// Basic building block of a paged KV cache - a single
// cache block. This class just holds metadata, no pointers
// since it is reused across all layers.
class KVCacheBlock
{
public:
    using IdType = std::int32_t;

    explicit KVCacheBlock(IdType blockId, kernels::KVCacheIndex blockIdx);

    void startScheduling();

    [[nodiscard]] IdType getBlockId() const;

    [[nodiscard]] kernels::KVCacheIndex::UnderlyingType getMemoryPoolBlockIndex() const;

    [[nodiscard]] bool isPrimary() const;

    void swapMemoryPoolBlockOffset(std::shared_ptr<KVCacheBlock> otherBlock);

    void incRefCount();

    void decRefCount();

    void decSchedulingRefCount();

    [[nodiscard]] bool hasRefs() const;

    [[nodiscard]] bool hasSchedulingRefs() const;

    void setTokens(VecTokens& tokens, bool isFull);

    [[nodiscard]] VecTokens const& getTokens() const;

    void setFreeBlockIterator(FreeBlocksQueue::iterator freeBlockIterator);

    void resetFreeBlockIterator();

    [[nodiscard]] std::optional<FreeBlocksQueue::iterator> const& getFreeBlockIterator() const;

    void setPrevBlock(BlockPtr prevBlock);

    void addNextBlock(VecTokens const& tokens, BlockPtr block);

    void removeNextBlock(VecTokens const& tokens);

    static std::shared_ptr<KVCacheBlock> findBestGPUBlockToFree(std::shared_ptr<KVCacheBlock> searchStart);

    static std::shared_ptr<KVCacheBlock> findLeafBlock(std::shared_ptr<KVCacheBlock> searchStart);

    [[nodiscard]] BlockPtr findMatchingBlock(VecTokens const& tokens) const;

    //! \brief Free block from previous block if present.
    void freeLeafBlock();

    [[nodiscard]] bool isFull() const;

    [[nodiscard]] bool isShared() const;

private:
    // Linear ID of block independent of pool
    IdType mBlockId;

    // Index of block in memory pool backing this block
    // Choice of pool is encoded into the type
    kernels::KVCacheIndex mMemoryPoolBlockIndex;

    // Number of references to the block
    SizeType32 mRefCount;

    // Number of references to the block
    SizeType32 mSchedulingRefCount;

    // Key of this block in mNextBlocks map in block pointed to by mPrevBlock
    VecTokens mTokens;

    // Previous block in sequence
    BlockPtr mPrevBlock;

    // Next block(s) in sequence(s)
    NextBlockMap mNextBlocks;

    // Iterator pointing to this block in mFreeBlocks.
    std::optional<FreeBlocksQueue::iterator> mFreeBlockIterator;

    // Flag indicating if block is full
    bool mIsFull;
};

class GenerationRequest
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using SharedPtr = std::shared_ptr<GenerationRequest>;

    explicit GenerationRequest(SizeType32 seqSlotIdx, SizeType32 numTokens, SizeType32 beamWidth)
        : mSeqSlotIdx(seqSlotIdx)
        , mNumTokens(numTokens)
        , mBeamWidth(beamWidth)
        , mCacheBlockIds(beamWidth)
    {
    }

    void addNewTokens(SizeType32 n)
    {
        mNumTokens += n;
    }

    void removeTokens(SizeType32 n)
    {
        TLLM_CHECK(n <= mNumTokens);
        TLLM_CHECK(mNumTokens - n >= 0);
        mNumTokens -= n;
    }

    [[nodiscard]] SizeType32 getSequenceSlotIdx() const
    {
        return mSeqSlotIdx;
    }

    [[nodiscard]] SizeType32 getNumTokens() const
    {
        return mNumTokens;
    }

    [[nodiscard]] SizeType32 getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] std::vector<std::vector<SizeType32>> const& getCacheBlockIds() const
    {
        return mCacheBlockIds;
    }

    void addCacheBlock(SizeType32 beamIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(beamIdx).push_back(blockId);
    }

    void changeCacheBlock(SizeType32 beamIdx, SizeType32 pagedBlockIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(beamIdx).at(pagedBlockIdx) = blockId;
    }

    void clearCacheBlocks()
    {
        for (auto& beamBlockIds : mCacheBlockIds)
        {
            beamBlockIds.clear();
        }
    }

    void removeLastBlock()
    {
        for (auto& beamBlockIds : mCacheBlockIds)
        {
            beamBlockIds.pop_back();
        }
    }

    void setNumPrepopulatedTokens(std::vector<int> numPrepopulatedTokens)
    {
        mNumPrepopulatedTokens = std::move(numPrepopulatedTokens);
    }

    [[nodiscard]] std::vector<int> const& getNumPrepopulatedTokens() const
    {
        return mNumPrepopulatedTokens;
    }

private:
    // Slot id of the sequence
    SizeType32 mSeqSlotIdx;
    // Current number of generated tokens
    SizeType32 mNumTokens;
    // Number of beams
    SizeType32 mBeamWidth;
    // List of blocks allocated for each beam of the sequence
    std::vector<std::vector<KVCacheBlock::IdType>> mCacheBlockIds;
    // Number of tokens already in kv cache before context phase.
    // A value > 0 indicates cached kv cache blocks were reused.
    // One value per beam.
    std::vector<int> mNumPrepopulatedTokens;
};

// BlockManager manages overall metadata of KVCacheBlocks in a layer of the
// network. Layers are expected to be symmetric, so the metadata can be
// reused for all layers of the network.
// The array of cache blocks for a layer is called a pool.
// Each pool has shape [max_blocks, 2, num_heads, tokens_per_block, head_size].
// Size per block and number of blocks per pool are pre-determined and set in
// constructor. These should not be changed after.
// Block shape is [2, num_heads, tokens_per_block, head_size].
// BlockManager maintains a list of free blocks at any time.
// Alloc pops off the block at the front, and Free pushes it back to the vector.
// BlockManager maintains a vector of lists of seqSlotIdx to allocated blocks
// per sequence. This can be used to Free all blocks belonging to a sequence.
class BlockManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    explicit BlockManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
        std::shared_ptr<runtime::CudaStream> stream, bool onboardBlocks, CacheType cacheType = CacheType::kSELF);

    ~BlockManager();

    void allocatePools(nvinfer1::DataType dtype, bool useUvm);

    void startScheduling();

    //! \brief Assign blocks for new sequence. Try to reuse blocks.
    void addSequence(
        GenerationRequest& sequence, SizeType32 inputLength, std::shared_ptr<LlmRequest> const& llmRequest);

    //! \brief Assign blocks for new sequence. Does not try to reuse blocks.
    void addSequence(GenerationRequest& sequence, SizeType32 numBlocks, SizeType32 unsharedBlockIdx);

    //! \brief Release block, which puts it back onto free blocks queue.
    //! \details Block appended by default, will be put at front if toFront is true.
    void releaseBlock(std::shared_ptr<KVCacheBlock> block, bool toFront = false);

    //! \brief Allocate new block for each beam of the sequence.
    //! \details Might free cached blocks if no free blocks are available.
    void allocateBlock(GenerationRequest& sequence, bool shareAmongBeams = false);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx);

    //! \brief Release blocks of the sequence. Store blocks for reuse if llmReqeust is provided.
    void releaseBlocks(GenerationRequest& sequence, std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

    //! \brief Simulate freeing all blocks for that sequence to check impact on number of free blocks
    void schedulingReleaseBlocks(GenerationRequest& sequence);

    //! \brief Release last block in the sequence
    void releaseLastBlock(GenerationRequest& sequence);

    [[nodiscard]] SizeType32 getNumFreeBlocks() const noexcept
    {
        return mFreePrimaryBlocks.size();
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept
    {
        return mReusedBlocks;
    }

    [[nodiscard]] SizeType32 getNumAllocatedBlocks() const noexcept
    {
        return getMaxNumBlocks() - getNumFreeBlocks();
    }

    [[nodiscard]] bool hasFreeBlocks(SizeType32 numRequired = 1) const noexcept
    {
        return getNumFreeBlocks() >= numRequired;
    }

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const noexcept
    {
        return mSchedulingNumFreeBlocks >= numRequired;
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const noexcept
    {
        return static_cast<SizeType32>(mAllBlocksById.size());
    }

    [[nodiscard]] SizeType32 getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    //! \brief Get size of one K/V cache block in one layer.
    //! @details Volume of [numKvHeads, tokensPerBlock, sizePerHead]
    [[nodiscard]] SizeType32 getBlockSize() const
    {
        return mBlockSize;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getPrimaryPool() const noexcept
    {
        return mPrimaryPool;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getSecondaryPool() const noexcept
    {
        return mSecondaryPool;
    }

    //! \brief Get index in pool to K or V block.
    //! \param blockId the blockId as returned by getBlockId()
    //! \param fieldIdx either 0 (K) or 1 (V),
    [[nodiscard]] kernels::KVCacheIndex getKOrVBlockIndex(KVCacheBlock::IdType blockId, SizeType32 fieldIdx) const;

    //! \brief Bring offloaded block from secondary to primary memory.
    //! \details Does nothing of block is already in primary memory.
    void onboardBlock(BlockPtr offloadBlock);

private:
    //! \brief Add single block to beam of sequence and mAllocatedBlocksPerSeq.
    void addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType32 beamIdx, SizeType32 seqSlotIdx);

    //! \brief Store blocks in cached blocks.
    //! \param blockedTokens Tokens of each block.
    //! \param blockIds Id of each block.
    void storeBlocks(std::list<VecTokens> blockedTokens, std::vector<KVCacheBlock::IdType> const& blockIds);

    //! \brief Try to load blocks from cache. Allocate new blocks if necessary.
    //! \param blockedTokens Tokens of each block.
    //! \param sequence Sequence to which blocks are assigned.
    //! \param beamIdx Beam of sequence to which blocks are assigned.
    //! \param seqSlotIdx Batch slot of sequence to which blocks are assigned.
    //! \return Number of matched tokens from loaded blocks.
    SizeType32 loadOrAllocateBlocks(std::list<VecTokens> const& blockedTokens, GenerationRequest& sequence,
        SizeType32 beamIdx, SizeType32 seqSlotIdx);

    //! \brief Find best primary block to free.
    //! \details The best primary block to free is the primary block that appears first in the queue and have no primary
    //! block descendants
    [[nodiscard]] std::shared_ptr<KVCacheBlock> findBestGPUBlockToFree();

    //! \brief Find block least likely to be reused, free it if necessary and return.
    [[nodiscard]] BlockPtr getFreeBlock();

    //! \brief Claim block if it is in free blocks list.
    void claimBlock(KVCacheBlock& block);

    //! \brief Free block from previous block and claim it from free blocks list.
    void claimLeafBlock(KVCacheBlock& block);

    //! \brief Compute pointer to raw KV block (K & V, all layers).
    [[nodiscard]] runtime::ITensor::SharedPtr computeBlockPointer(std::shared_ptr<KVCacheBlock> block) const;

    //! \brief Copy content of src block to dst.
    void copyBlock(BlockPtr src, BlockPtr dst);

private:
    // Number of blocks in pools
    SizeType32 mNumPrimaryBlocks;
    SizeType32 mNumSecondaryBlocks;
    // List of free blocks. Blocks are either backed by fast primary memory or slow secondary memory,
    // we maintain separate queues for these.
    FreeBlocksQueue mFreePrimaryBlocks;
    FreeBlocksQueue mFreeSecondaryBlocks;
    // List of allocated blocks for each sequences
    std::vector<std::vector<BlockPtr>> mAllocatedBlocksPerSeq;
    // Memory pools. Primary is fast memory, secondary is slower memory used for offloading.
    runtime::ITensor::SharedPtr mPrimaryPool;
    runtime::ITensor::SharedPtr mSecondaryPool;
    // Whether offloaded blocks should be onboarded before reuse.
    bool mOnboardBlocks;
    // Buffer manager
    runtime::BufferManager mBufferManager;
    // Number of layers
    SizeType32 mNumLayers;
    // Volume of [numKvHeads, tokensPerBlock, sizePerHead]
    SizeType32 mBlockSize;
    // Used to keep track of number of free blocks during scheduling
    SizeType32 mSchedulingNumFreeBlocks;
    // Number of tokens per one block
    SizeType32 mTokensPerBlock;
    // List of all blocks by idx
    std::vector<BlockPtr> mAllBlocksById;
    // Dummy block acting as root for BlockToken searches
    BlockPtr mCachedBlocksRoot;
    // Statistics for block allocations/reuse
    std::size_t mAllocTotalBlocks, mAllocNewBlocks, mReusedBlocks;
    // KV cache type (self or cross)
    CacheType mCacheType;
};

class KVCacheManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using SequencesPtr = GenerationRequest::SharedPtr;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, bool useOneMoreBlock,
        CudaStreamPtr stream, bool enableBlockReuse = false, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF);

    void allocatePools(nvinfer1::DataType dtype, bool useUvm = false);

    void startScheduling();

    [[nodiscard]] SizeType32 getTokensPerBlock() const
    {
        return mBlockManager.getTokensPerBlock();
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const
    {
        return mBlockManager.getMaxNumBlocks();
    }

    [[nodiscard]] SizeType32 getUsedNumBlocks() const
    {
        return mBlockManager.getNumAllocatedBlocks();
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const
    {
        return mBlockManager.getNumFreeBlocks();
    }

    [[nodiscard]] KvCacheStats getKvCacheStats() const
    {
        KvCacheStats kvCacheStats;
        kvCacheStats.maxNumBlocks = getMaxNumBlocks();
        kvCacheStats.freeNumBlocks = getNumFreeBlocks();
        kvCacheStats.usedNumBlocks = getUsedNumBlocks();
        kvCacheStats.toksPerBlock = getTokensPerBlock();

        return kvCacheStats;
    }

    [[nodiscard]] SizeType32 getMaxBlocksPerSeq() const
    {
        return mMaxBlocksPerSeq;
    }

    [[nodiscard]] BlockManager const& getBlockManager() const
    {
        return mBlockManager;
    }

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request to completion (i.e. for
    /// maxNewTokens)
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getNeededBlocksToCompletion(LlmRequest const& req) const;

    void addContextTokens(SizeType32 seqSlotIdx, SizeType32 numTokens);

    void addToken(SizeType32 seqSlotIdx);

    void addSequence(SizeType32 seqSlotIdx, SizeType32 inputLength, SizeType32 beamWidth,
        std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

    void removeSequence(SizeType32 seqSlotIdx, std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

    void schedulingRemoveSequence(SizeType32 seqSlotIdx);

    [[nodiscard]] runtime::ITensor::UniquePtr getBlockPoolPointers() const;

    void getBlockOffsetsOfBatch(
        runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const;

    //! @return maxBlockCount of all beams
    SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const;

    // Volume of [2, numKvHeads, tokensPerBlock, sizePerHead]
    [[nodiscard]] static SizeType32 constexpr calculatePageSize(tensorrt_llm::runtime::ModelConfig const& modelConfig)
    {
        return 2 * modelConfig.getNbKvHeads() * modelConfig.getTokensPerBlock() * modelConfig.getSizePerHead();
    }

    // numLayers * 2 * numKvHeads * sizePerHead
    [[nodiscard]] static SizeType32 constexpr calculateCacheSizePerToken(
        tensorrt_llm::runtime::ModelConfig const& modelConfig, tensorrt_llm::runtime::WorldConfig const& worldConfig)
    {
        return modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism()) * 2 * modelConfig.getNbKvHeads()
            * modelConfig.getSizePerHead();
    }

    [[nodiscard]] static std::tuple<SizeType32, SizeType32> const calculateMaxNumBlocks(KvCacheConfig const& config,
        nvinfer1::DataType dtype, tensorrt_llm::runtime::ModelConfig const& modelConfig,
        tensorrt_llm::runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager);

    [[nodiscard]] SizeType32 getNumPrepopulatedTokens(SizeType32 batchSlotIdx, SizeType32 beamIdx) const
    {
        auto const& prepopulatedTokens = mSequences.at(batchSlotIdx)->getNumPrepopulatedTokens();
        return prepopulatedTokens.size() > 0 ? prepopulatedTokens.at(beamIdx) : 0;
    }

    [[nodiscard]] bool isEnableBlockReuse() const
    {
        return mEnableBlockReuse;
    }

    void removeToken(SizeType32 seqSlotIdx);
    void rewindKVCache(SizeType32 seqSlotIdx, SizeType32 rewindLengths);

    [[nodiscard]] bool isCrossKv() const
    {
        return mCacheType == CacheType::kCROSS;
    }

private:
    void setOffsets(kernels::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 seqSlotIdx,
        SizeType32 beamIdx, SizeType32 blockIdx, KVCacheBlock::IdType blockId) const;

    void resetBlockOffsets(SizeType32 seqSlotIdx, SizeType32 beamWidth);
    void cacheBlockOffsets(GenerationRequest const& seq, SizeType32 seqSlotIdx);
    void cacheNewBlockOffsets(GenerationRequest const& seq, SizeType32 seqSlotIdx);
    void updateNewBlockPointer(GenerationRequest const& seq, SizeType32 seqSlotIdx, SizeType32 blockIdx);
    void updateToken(SizeType32 seqSlotIdx, bool addToken);

private:
    // Maximum number of sequences
    SizeType32 mMaxNumSequences;
    // Maximum beam width
    SizeType32 mMaxBeamWidth;
    // Maximum number of blocks per sequence
    SizeType32 mMaxBlocksPerSeq;
    // Maximum kv cache length per sequence
    // Enable cyclic kv cache when it exceeds
    SizeType32 mMaxAttentionWindow;
    // Number of tokens to fill up the sink tokens to a full block size
    SizeType32 mSinkBubbleLength;
    // Maximum token length (including bubble)
    SizeType32 mMaxTokenNum;
    // Number of tokens in the sink blocks
    SizeType32 mSinkBlockTokenLength;
    // Block manager
    BlockManager mBlockManager;
    // List of all sequences
    std::vector<SequencesPtr> mSequences;
    // buffer for block indices for all managed sequences
    runtime::ITensor::SharedPtr mSequenceBlockIndices;
    // Whether to cache KV pages for reuse
    bool mEnableBlockReuse;
    // KV cache type (self or cross)
    CacheType mCacheType;
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
