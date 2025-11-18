# Scenario 03: Discussion Points - KV Cache Management

## Key Discussion Topics

### 1. PagedAttention Fundamentals (15 min)

**Question:** "Why is paged memory better than contiguous allocation for KV cache?"

**Expected Answer:**
- Contiguous: Pre-allocate max_seq_len for each sequence → wastes memory for short sequences
- Paged: Allocate in fixed blocks → only use what's needed
- Reduces fragmentation from ~30% to <5%
- Enables memory sharing across sequences

**Follow-ups:**
- "How do you choose block size?" (Trade-off: 16 tokens balances granularity and overhead)
- "What's the memory overhead of block tables?" (Minimal: ~4 bytes per block per sequence)

### 2. Prefix Caching (12 min)

**Question:** "How would you implement prefix caching for system prompts?"

**Key Points:**
- Hash token sequences to detect duplicates
- Store block IDs for common prefixes
- Reference counting for shared blocks
- Copy-on-write when modifying shared cache

**Follow-up:** "What if two sequences share 80% of their prefix but differ in the middle?"
- Expected: Can only share complete block-aligned prefixes
- Partial sharing not supported in basic design
- Could use tree-based structure for finer sharing

### 3. Eviction Policies (10 min)

**Question:** "The cache is full. How do you decide which sequences to evict?"

**Strategies:**
1. **LRU:** Simple, but doesn't consider priority
2. **Priority-based:** Considers user priority, progress, memory usage
3. **Hybrid:** LRU within priority tiers

**Trade-offs:**
- LRU: Fair but may evict important sequences
- Priority: Better for SLAs but can starve low-priority
- Hybrid: Balanced but more complex

## Common Mistakes

1. **Not accounting for reference counting** - Leads to use-after-free bugs
2. **Ignoring copy-on-write overhead** - Can be expensive if done frequently
3. **Wrong block size** - Too small: overhead, too large: fragmentation
4. **Missing layer dimension** - KV cache exists for every layer

## Red/Green Flags

**Red Flags:**
- Proposes contiguous allocation only
- Doesn't understand fragmentation
- Can't explain copy-on-write
- No eviction strategy

**Green Flags:**
- Mentions PagedAttention or similar
- Accurate memory calculations
- Discusses reference counting
- Proposes prefix caching unprompted
