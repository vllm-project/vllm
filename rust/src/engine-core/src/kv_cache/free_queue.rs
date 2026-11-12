// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `FreeKVCacheBlockQueue` of `kv_cache_utils.py`.

use super::block::BlockId;

const NONE: u32 = u32::MAX;

/// This class organizes a list of KVCacheBlock objects to a doubly linked
/// list of free blocks. We implement this class instead of using Python
/// builtin deque to support removing a block in the middle of the queue
/// in O(1) time.
///
/// The queue is ordered by block ID in the beginning. When a block is allocated
/// and then freed, it will be appended back with the eviction order:
/// 1. The least recent used block is at the front (LRU).
/// 2. If two blocks have the same last accessed time (allocated by the
///    same sequence), the one with more hash tokens (the tail of a block
///    chain) is at the front.
///
/// Note that we maintain this order by reversing the block order when free
/// blocks of a request. This operation is outside of this class.
///
/// The links are index based: entries `0..num_blocks` belong to the blocks of
/// the same id, and two extra entries are the fake head and tail. A block is
/// in the queue iff both of its links are set.
#[derive(Clone, Debug)]
pub struct FreeKvCacheBlockQueue {
    pub num_free_blocks: usize,
    prev: Vec<u32>,
    next: Vec<u32>,
    head: u32,
    tail: u32,
}

impl FreeKvCacheBlockQueue {
    /// A queue holding blocks `0..num_blocks` in id order.
    pub fn new(num_blocks: usize) -> Self {
        let n = u32::try_from(num_blocks).expect("too many blocks");
        assert!(n < NONE - 2, "too many blocks");
        // Create a fake head and a tail block for the doubly linked list to
        // reduce branching in the code.
        //
        // The implementation guaranteed that the fake head and tail
        // are NEVER got popped, so we could safely assume each real blocks
        // in the queue has prev and next blocks.
        let head = n;
        let tail = n + 1;
        let mut prev = vec![NONE; num_blocks + 2];
        let mut next = vec![NONE; num_blocks + 2];
        // Initialize doubly links of consecutive blocks, with the fake head
        // before the first and the fake tail after the last.
        let mut last = head;
        for i in 0..n {
            next[last as usize] = i;
            prev[i as usize] = last;
            last = i;
        }
        next[last as usize] = tail;
        prev[tail as usize] = last;
        Self {
            num_free_blocks: num_blocks,
            prev,
            next,
            head,
            tail,
        }
    }

    /// Whether `block` is linked into the queue.
    pub fn contains(&self, block: BlockId) -> bool {
        self.prev[block as usize] != NONE && self.next[block as usize] != NONE
    }

    /// Pop the first free block and reduce num_free_blocks by 1.
    pub fn popleft(&mut self) -> BlockId {
        let first = self.next[self.head as usize];
        if first == self.tail || first == NONE {
            assert!(
                self.num_free_blocks == 0,
                "num_free_blocks ({}) is out of sync with the free list.",
                self.num_free_blocks
            );
            panic!("No free blocks available");
        }
        let next = self.next[first as usize];
        if next == NONE {
            // This should not happen if the block is from the free list.
            // It indicates a bug in the caller's logic.
            panic!("Invalid block found in popleft() which doesn't have a valid next_free_block");
        }
        // Connect fake_head and the next block of first_block (i.e. second block
        // or fake tail).
        self.next[self.head as usize] = next;
        self.prev[next as usize] = self.head;
        // Remove the block from the linked list.
        self.prev[first as usize] = NONE;
        self.next[first as usize] = NONE;
        self.num_free_blocks -= 1;
        first
    }

    /// Pop the first n free blocks and reduce num_free_blocks by n.
    pub fn popleft_n(&mut self, n: usize) -> Vec<BlockId> {
        if n == 0 {
            return Vec::new();
        }
        assert!(self.num_free_blocks >= n);
        self.num_free_blocks -= n;

        let mut ret = Vec::with_capacity(n);
        let mut curr = self.next[self.head as usize];
        // Pop n blocks from the head of the list
        for _ in 0..n {
            assert!(curr != NONE && curr != self.tail);
            ret.push(curr);
            let last = curr;
            curr = self.next[curr as usize];
            // Reset prev_free_block and next_free_block of all popped blocks
            self.prev[last as usize] = NONE;
            self.next[last as usize] = NONE;
        }
        if curr != NONE {
            // The queue is not empty, connect the fake head to
            // the new first block.
            self.next[self.head as usize] = curr;
            self.prev[curr as usize] = self.head;
        }
        ret
    }

    /// Remove a block in the free list and reduce num_free_blocks by 1.
    pub fn remove(&mut self, block: BlockId) {
        let prev = self.prev[block as usize];
        let next = self.next[block as usize];
        if prev == NONE || next == NONE {
            // This should not happen if the block is from the free list.
            // It indicates a bug in the caller's logic.
            panic!("remove() called on an invalid block: {block}");
        }
        // Link the previous block to the next block.
        self.next[prev as usize] = next;
        // Link the next block to the previous block.
        self.prev[next as usize] = prev;
        // Remove the block from the linked list.
        self.prev[block as usize] = NONE;
        self.next[block as usize] = NONE;
        self.num_free_blocks -= 1;
    }

    /// Put a block back into the free list and increase
    /// num_free_blocks by 1.
    pub fn append(&mut self, block: BlockId) {
        let last = self.prev[self.tail as usize];
        assert!(
            last != NONE,
            "prev_free_block of fake_free_list_tail should always exist"
        );
        // Connect the new block after the last block.
        self.next[last as usize] = block;
        self.prev[block as usize] = last;
        // Connect the fake tail after the new block.
        self.next[block as usize] = self.tail;
        self.prev[self.tail as usize] = block;
        self.num_free_blocks += 1;
    }

    /// Put a list of blocks at the front of the free list.
    pub fn prepend_n(&mut self, blocks: &[BlockId]) {
        if blocks.is_empty() {
            return;
        }
        let first = self.next[self.head as usize];
        assert!(
            first != NONE,
            "next_free_block of fake_free_list_head should always exist"
        );
        let mut prev = self.head;
        for &block in blocks {
            self.prev[block as usize] = prev;
            self.next[prev as usize] = block;
            prev = block;
        }
        self.next[prev as usize] = first;
        self.prev[first as usize] = prev;
        self.num_free_blocks += blocks.len();
    }

    /// Put a list of blocks back into the free list.
    pub fn append_n(&mut self, blocks: &[BlockId]) {
        if blocks.is_empty() {
            return;
        }
        let mut last = self.prev[self.tail as usize];
        assert!(
            last != NONE,
            "prev_free_block of fake_free_list_tail should always exist"
        );
        // Add inter-connections between consecutive blocks
        for &block in blocks {
            self.prev[block as usize] = last;
            self.next[last as usize] = block;
            last = block;
        }
        // Connect the last block of <blocks> to the fake tail
        self.next[last as usize] = self.tail;
        self.prev[self.tail as usize] = last;
        self.num_free_blocks += blocks.len();
    }

    /// Get all free blocks in the free list. Mainly used for testing.
    pub fn get_all_free_blocks(&self) -> Vec<BlockId> {
        self.iter_blocks_after(None).collect()
    }

    /// Iterate free blocks in eviction order after the cursor.
    pub fn iter_blocks_after(&self, cursor: Option<BlockId>) -> impl Iterator<Item = BlockId> + '_ {
        let mut curr = match cursor {
            None => self.next[self.head as usize],
            Some(cursor) => self.next[cursor as usize],
        };
        std::iter::from_fn(move || {
            if curr == NONE || curr == self.tail {
                return None;
            }
            let block = curr;
            curr = self.next[curr as usize];
            Some(block)
        })
    }
}
