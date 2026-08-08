use pyo3::exceptions::{PyAssertionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

// Sentinel block_ids for the doubly-linked free list.
// Real block_ids are >= 0. We use negative values for the sentinels.
const SENTINEL_HEAD: i32 = -2;
const SENTINEL_TAIL: i32 = -3;
const NO_LINK: i32 = i32::MIN;

#[pyclass(module = "vllm_rs", str)]
pub struct KVCacheBlock {
    #[pyo3(get)]
    pub block_id: i32,
    #[pyo3(get, set)]
    pub ref_cnt: i32,
    // block_hash is `BlockHashWithGroupId` which is `NewType(bytes)` in vLLM.
    // We store it as owned bytes so Rust can hash/compare without going through
    // the Python object.
    block_hash_bytes: Option<Vec<u8>>,
    #[pyo3(get, set)]
    pub is_null: bool,
}

// Mirrors the hand-written `__repr__` of the Python dataclass
// (kv_cache_utils.py also writes it by hand, to avoid recursing through the
// free-list pointers). The hash is summarized as a byte count instead of
// dumped raw, and prev/next are always None because Rust owns the list state
// — neither is expressible with derive(Debug) or the #[pyclass(str = "...")]
// format-string shorthand. `__repr__` below delegates here; #[pyclass(str)]
// derives `__str__` from this impl, matching Python's str→repr fallback.
impl std::fmt::Display for KVCacheBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KVCacheBlock(block_id={}, ref_cnt={}, _block_hash={}, prev_free_block=None, next_free_block=None)",
            self.block_id,
            self.ref_cnt,
            match &self.block_hash_bytes {
                Some(b) => format!("{} bytes", b.len()),
                None => "None".to_string(),
            }
        )
    }
}

impl KVCacheBlock {
    pub fn block_hash_ref(&self) -> Option<&[u8]> {
        self.block_hash_bytes.as_deref()
    }

    pub fn take_hash(&mut self) -> Option<Vec<u8>> {
        self.block_hash_bytes.take()
    }

    pub fn set_hash_internal(&mut self, v: Vec<u8>) {
        self.block_hash_bytes = Some(v);
    }
}

#[pymethods]
impl KVCacheBlock {
    #[new]
    #[pyo3(signature = (block_id, ref_cnt=0, _block_hash=None, is_null=false))]
    pub fn new(
        block_id: i32,
        ref_cnt: i32,
        _block_hash: Option<Vec<u8>>,
        is_null: bool,
    ) -> Self {
        Self {
            block_id,
            ref_cnt,
            block_hash_bytes: _block_hash,
            is_null,
        }
    }

    #[getter]
    fn block_hash<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.block_hash_bytes
            .as_ref()
            .map(|b| PyBytes::new(py, b))
    }

    #[setter]
    fn set_block_hash(&mut self, v: Bound<'_, PyAny>) -> PyResult<()> {
        if self.block_hash_bytes.is_some() {
            return Err(PyAssertionError::new_err(
                "The block already has a hash. This should not happen.",
            ));
        }
        let bytes: Vec<u8> = v.extract()?;
        self.block_hash_bytes = Some(bytes);
        Ok(())
    }

    pub fn reset_hash(&mut self) {
        self.block_hash_bytes = None;
    }

    // Compatibility stubs for code that reads the linked-list pointers
    // externally (e.g. simple_kv_offload). Always return None in the Rust
    // implementation — Rust owns the list state internally. Callers that
    // need external iteration should use FreeKVCacheBlockQueue.iter_free_blocks.
    #[getter]
    fn prev_free_block(&self) -> Option<()> {
        None
    }

    #[getter]
    fn next_free_block(&self) -> Option<()> {
        None
    }

    #[setter]
    fn set_prev_free_block(&self, _v: Bound<'_, PyAny>) {}

    #[setter]
    fn set_next_free_block(&self, _v: Bound<'_, PyAny>) {}

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Rust-internal linked list of free blocks. Indexed by block_id (i32).
/// Mirrors `FreeKVCacheBlockQueue` but avoids Python attribute access for
/// prev/next pointers — stored as two Vec<i32> using SENTINEL values.
#[pyclass(module = "vllm_rs")]
pub struct FreeKVCacheBlockQueue {
    /// `Py<KVCacheBlock>` objects owned by this queue (held for returning to Python).
    /// Indexed by block_id. Null blocks and the passed-in `blocks` list share
    /// the same Py references.
    blocks: Vec<Py<KVCacheBlock>>,
    /// prev[i] = block_id of the previous block in the list, or SENTINEL_HEAD.
    /// NO_LINK means "not in list".
    prev: Vec<i32>,
    next: Vec<i32>,
    /// Head/tail sentinels — point to the first/last real block_id in the list,
    /// or to each other (= SENTINEL_TAIL / SENTINEL_HEAD) when empty.
    head_next: i32,
    tail_prev: i32,
    /// Fake head/tail to match the Python API exposing them for traversal.
    /// They are Py<KVCacheBlock> with block_id = -1, not linked into `blocks`.
    fake_head: Py<KVCacheBlock>,
    fake_tail: Py<KVCacheBlock>,
    num_free: usize,
}

impl FreeKVCacheBlockQueue {
    pub fn rust_num_free(&self) -> usize {
        self.num_free
    }

    fn verify_block_id(&self, bid: i32) -> PyResult<usize> {
        if bid < 0 || (bid as usize) >= self.blocks.len() {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid block_id {} for queue of size {}",
                bid,
                self.blocks.len()
            )));
        }
        Ok(bid as usize)
    }

    fn unlink(&mut self, idx: usize) {
        let bid = idx as i32;
        let p = self.prev[idx];
        let n = self.next[idx];
        if p == SENTINEL_HEAD {
            self.head_next = n;
        } else if p != NO_LINK {
            self.next[p as usize] = n;
        }
        if n == SENTINEL_TAIL {
            self.tail_prev = p;
        } else if n != NO_LINK {
            self.prev[n as usize] = p;
        }
        self.prev[idx] = NO_LINK;
        self.next[idx] = NO_LINK;
        // defensive: if unlinking wiped both sentinels, keep them consistent
        if self.head_next == bid || self.tail_prev == bid {
            // shouldn't happen given the branches above, but guard anyway
            self.head_next = SENTINEL_TAIL;
            self.tail_prev = SENTINEL_HEAD;
        }
    }

    fn link_at_tail(&mut self, idx: usize) {
        let bid = idx as i32;
        let old_tail = self.tail_prev;
        self.prev[idx] = old_tail;
        self.next[idx] = SENTINEL_TAIL;
        if old_tail == SENTINEL_HEAD {
            self.head_next = bid;
        } else {
            self.next[old_tail as usize] = bid;
        }
        self.tail_prev = bid;
    }
}

#[pymethods]
impl FreeKVCacheBlockQueue {
    #[new]
    pub fn new(py: Python<'_>, blocks: Vec<Py<KVCacheBlock>>) -> PyResult<Self> {
        let n = blocks.len();
        let mut prev = vec![NO_LINK; n];
        let mut next = vec![NO_LINK; n];
        // Sanity-check contiguous block_ids [0..n).
        for (i, b) in blocks.iter().enumerate() {
            let bid = b.borrow(py).block_id;
            if bid != i as i32 {
                return Err(PyValueError::new_err(format!(
                    "FreeKVCacheBlockQueue: block[{}].block_id = {} (expected {})",
                    i, bid, i
                )));
            }
        }
        // Link them all front-to-back in block_id order.
        let (head_next, tail_prev) = if n == 0 {
            (SENTINEL_TAIL, SENTINEL_HEAD)
        } else {
            for i in 0..n {
                prev[i] = if i == 0 { SENTINEL_HEAD } else { (i - 1) as i32 };
                next[i] = if i == n - 1 {
                    SENTINEL_TAIL
                } else {
                    (i + 1) as i32
                };
            }
            (0i32, (n - 1) as i32)
        };
        let fake_head = Py::new(py, KVCacheBlock::new(-1, 0, None, false))?;
        let fake_tail = Py::new(py, KVCacheBlock::new(-1, 0, None, false))?;
        Ok(Self {
            blocks,
            prev,
            next,
            head_next,
            tail_prev,
            fake_head,
            fake_tail,
            num_free: n,
        })
    }

    #[getter]
    fn num_free_blocks(&self) -> usize {
        self.num_free
    }

    // Compatibility stubs — fake_free_list_head/tail are surfaced so any
    // Python caller that *only* checks `x.next_free_block is tail` won't
    // crash. Traversal via the fake head will not return the real list
    // (prev/next are None on real KVCacheBlock); callers needing traversal
    // should use get_all_free_blocks().
    #[getter]
    fn fake_free_list_head<'py>(&self, py: Python<'py>) -> Bound<'py, KVCacheBlock> {
        self.fake_head.bind(py).clone()
    }

    #[getter]
    fn fake_free_list_tail<'py>(&self, py: Python<'py>) -> Bound<'py, KVCacheBlock> {
        self.fake_tail.bind(py).clone()
    }

    pub fn popleft<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, KVCacheBlock>> {
        if self.head_next == SENTINEL_TAIL {
            if self.num_free != 0 {
                return Err(PyAssertionError::new_err(
                    "num_free is out of sync with the free list.",
                ));
            }
            return Err(PyValueError::new_err("No free blocks available"));
        }
        let idx = self.head_next as usize;
        self.unlink(idx);
        self.num_free -= 1;
        Ok(self.blocks[idx].bind(py).clone())
    }

    pub fn popleft_n<'py>(
        &mut self,
        py: Python<'py>,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        if n == 0 {
            return Ok(PyList::empty(py));
        }
        if n > self.num_free {
            return Err(PyAssertionError::new_err(format!(
                "popleft_n({}) > num_free({})",
                n, self.num_free
            )));
        }
        self.num_free -= n;

        let out = PyList::empty(py);
        let mut cur = self.head_next;
        for _ in 0..n {
            if cur < 0 {
                return Err(PyAssertionError::new_err(
                    "popleft_n: ran off the free list before n",
                ));
            }
            let idx = cur as usize;
            let nx = self.next[idx];
            self.prev[idx] = NO_LINK;
            self.next[idx] = NO_LINK;
            out.append(self.blocks[idx].bind(py).clone())?;
            cur = nx;
        }
        // `cur` is the new head (either a real idx or SENTINEL_TAIL).
        if cur == SENTINEL_TAIL {
            self.head_next = SENTINEL_TAIL;
            self.tail_prev = SENTINEL_HEAD;
        } else {
            self.head_next = cur;
            self.prev[cur as usize] = SENTINEL_HEAD;
        }
        Ok(out)
    }

    pub fn remove(&mut self, py: Python<'_>, block: Bound<'_, KVCacheBlock>) -> PyResult<()> {
        let bid = block.borrow().block_id;
        let idx = self.verify_block_id(bid)?;
        // Check block is actually in the free list (prev/next not NO_LINK).
        if self.prev[idx] == NO_LINK || self.next[idx] == NO_LINK {
            // Match Python's error message style.
            let _ = py;
            return Err(PyRuntimeError::new_err(format!(
                "remove() called on an invalid block: block_id={}",
                bid
            )));
        }
        self.unlink(idx);
        self.num_free -= 1;
        Ok(())
    }

    pub fn append(&mut self, py: Python<'_>, block: Bound<'_, KVCacheBlock>) -> PyResult<()> {
        let _ = py;
        let bid = block.borrow().block_id;
        let idx = self.verify_block_id(bid)?;
        self.link_at_tail(idx);
        self.num_free += 1;
        Ok(())
    }

    pub fn append_n(
        &mut self,
        py: Python<'_>,
        blocks: Vec<Py<KVCacheBlock>>,
    ) -> PyResult<()> {
        if blocks.is_empty() {
            return Ok(());
        }
        let mut last = self.tail_prev;
        for b in &blocks {
            let bid = b.borrow(py).block_id;
            let idx = self.verify_block_id(bid)?;
            self.prev[idx] = last;
            if last == SENTINEL_HEAD {
                self.head_next = bid;
            } else {
                self.next[last as usize] = bid;
            }
            last = bid;
        }
        // Close up to the tail sentinel.
        let last_idx = last as usize;
        self.next[last_idx] = SENTINEL_TAIL;
        self.tail_prev = last;
        self.num_free += blocks.len();
        Ok(())
    }

    pub fn get_all_free_blocks<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        let mut cur = self.head_next;
        while cur >= 0 {
            let idx = cur as usize;
            out.append(self.blocks[idx].bind(py).clone())?;
            cur = self.next[idx];
        }
        Ok(out)
    }
}
