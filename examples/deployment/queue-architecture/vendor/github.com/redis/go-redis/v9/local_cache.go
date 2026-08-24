package redis

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// cacheEntryState tracks the lifecycle of a local cache entry.
type cacheEntryState uint8

const (
	// cacheEntryInProgress marks a placeholder entry while a value is being fetched.
	cacheEntryInProgress cacheEntryState = iota
	// cacheEntryValid marks an entry that contains a value that can be returned.
	cacheEntryValid
)

// cacheEntry represents a cached command reply and its Redis-key associations.
type cacheEntry struct {
	cacheKey  string
	redisKeys []string
	value     []byte
	state     cacheEntryState

	token      uint64
	sizeBytes  int64
	reservedAt time.Time
	waitCh     chan struct{}
	waitClosed bool

	// lastAccessNs is a recency token for LRU eviction: a global atomic counter
	// bumped on every access, stored atomically so the read path can mark a
	// touch under the shard's RLock without upgrading to a write lock.
	lastAccessNs atomic.Int64

	// validAt retains time.Now's monotonic component for the MaxStaleness
	// backstop, so wall-clock corrections cannot extend an entry's lifetime.
	// Written under Lock (Set/Fulfill), read under RLock (get).
	validAt time.Time

	// ownerConnID is the conn that fetched this entry (set by FulfillOwned; 0 =
	// none). Default CLIENT TRACKING sends a key's invalidation only to that
	// conn, so the entry must be evicted when it goes away (see EvictByConn).
	ownerConnID uint64
}

// lruSequence is the global monotonic counter feeding lastAccessNs. It totally
// orders recency across all entries in all shards for approximate-LRU eviction.
var lruSequence atomic.Int64

// nextLRUToken returns the next strictly-greater LRU token.
func nextLRUToken() int64 {
	return lruSequence.Add(1)
}

// CacheSizer calculates estimated memory usage in bytes for a cache entry.
//
// Experimental: this API may change in a minor release.
type CacheSizer func(cacheKey string, redisKeys []string, value []byte) int64

// CacheConfig configures a local cache instance.
//
// Experimental: this API may change in a minor release.
type CacheConfig struct {
	// MaxEntries limits the number of entries. Zero or negative means unlimited.
	MaxEntries int
	// MaxMemoryBytes limits estimated memory usage in bytes. Zero or negative means unlimited.
	//
	// If both MaxEntries and MaxMemoryBytes are unlimited, MaxEntries defaults to
	// defaultCacheMaxEntries so the cache cannot grow without bound. The cache is
	// sharded 16 ways (above small thresholds) and each shard enforces its 1/16
	// share, so an entry larger than MaxMemoryBytes/16 is never admitted —
	// size it to at least 16× your largest reply.
	MaxMemoryBytes int64
	// Sizer estimates memory usage per entry. If nil, a built-in approximation is used.
	//
	// Sizer may be invoked concurrently from multiple goroutines and must be
	// thread-safe. It must return quickly and must not call back into the
	// cache (Get, Set, Delete*, Flush, etc.): some call sites hold an internal
	// shard lock, so re-entry can deadlock.
	Sizer CacheSizer
	// StaleTimeout is the duration after which an IN_PROGRESS placeholder is
	// considered stale and eligible for takeover by a new Reserve call.
	// If zero, defaults to defaultStaleTimeout (5s).
	StaleTimeout time.Duration

	// DrainInterval is the background-drainer period (default 5ms; zero uses the
	// default): how often idle pool conns are swept for buffered "invalidate"
	// frames, roughly bounding cache-hit staleness. Values below 1ms are clamped
	// to 1ms.
	DrainInterval time.Duration

	// MaxStaleness caps how long a cached entry is served after it became valid,
	// regardless of invalidation. It is a correctness
	// BACKSTOP for lost invalidations or connection-lifecycle gaps ("Window 2"), not
	// the primary freshness mechanism. Keep it well above the invalidation round-trip
	// (e.g. seconds); per-entry refetch overhead scales ~1/MaxStaleness.
	//
	// Default: 0 (disabled).
	MaxStaleness time.Duration
}

// Cache is the thread-safe storage contract used by client-side caching.
//
// All methods may be called concurrently. Cache keys and Redis keys are opaque
// strings and must be preserved exactly. Removing a reservation must wake any
// Get calls waiting for it.
//
// Reserve must allow only one caller to fetch a missing key and return a token
// that is valid until FulfillOwned, Cancel, or an eviction removes that
// reservation. FulfillOwned and Cancel must modify only a reservation with the
// matching token. Get may wait for an in-progress reservation and must stop
// waiting when ctx is done.
//
// Experimental: this API may change in a minor release.
type Cache interface {
	Get(ctx context.Context, cacheKey string) ([]byte, bool)
	Reserve(cacheKey string, redisKeys []string) (token uint64, shouldFetch bool)
	// FulfillOwned publishes a reserved value and records the connection that
	// fetched it so the entry can be evicted if that connection loses tracking.
	FulfillOwned(cacheKey string, token, ownerConnID uint64, value []byte) bool
	Cancel(cacheKey string, token uint64) bool
	DeleteByRedisKey(redisKey string) int
	DeleteByCacheKey(cacheKey string) bool
	// EvictByConn removes every entry fetched by connID.
	EvictByConn(connID uint64) int
	Flush() int
}

const (
	defaultStaleTimeout    = 5 * time.Second
	defaultCacheShardCount = 16

	// defaultCacheMaxEntries bounds the cache when the config leaves both
	// MaxEntries and MaxMemoryBytes unlimited (matches the 10k-entry default
	// other Redis clients use, e.g. redis-py).
	defaultCacheMaxEntries = 10000

	// shardingThresholdEntries / shardingThresholdBytes: caches with capacity
	// below these thresholds fall back to a single shard so global LRU /
	// memory-cap semantics behave exactly as a non-sharded cache would.
	shardingThresholdEntries = 64
	shardingThresholdBytes   = 64 * 1024
)

// NewLocalCache creates a thread-safe local cache with approximate-LRU
// eviction. The cache is internally sharded by cache-key hash to reduce
// mutex contention under high concurrent access.
//
// Experimental: this API may change in a minor release.
func NewLocalCache(cfg CacheConfig) *LocalCache {
	sizer := cfg.Sizer
	if sizer == nil {
		sizer = defaultCacheSizer
	}

	staleTimeout := cfg.StaleTimeout
	if staleTimeout <= 0 {
		staleTimeout = defaultStaleTimeout
	}

	maxEntries := cfg.MaxEntries
	maxMemoryBytes := cfg.MaxMemoryBytes
	// An unbounded cache can grow until the process OOMs; require at least
	// one limit.
	if maxEntries <= 0 && maxMemoryBytes <= 0 {
		maxEntries = defaultCacheMaxEntries
	}

	shardCount := defaultCacheShardCount
	if maxEntries > 0 && maxEntries < shardingThresholdEntries {
		shardCount = 1
	}
	if maxMemoryBytes > 0 && maxMemoryBytes < int64(shardingThresholdBytes) {
		shardCount = 1
	}

	c := &LocalCache{
		shards:     make([]cacheShard, shardCount),
		shardCount: uint32(shardCount),
		shardMask:  uint32(shardCount - 1),
		sizer:      sizer,
	}
	for i := range c.shards {
		s := &c.shards[i]
		s.entries = make(map[string]*cacheEntry)
		s.byRedisKey = make(map[string]map[string]struct{})
		s.byConnID = make(map[uint64]map[string]struct{})
		// Distribute capacity so the per-shard caps sum to exactly the
		// configured limits; a ceil-per-shard split would let total residency
		// exceed MaxEntries/MaxMemoryBytes.
		if maxEntries > 0 {
			s.maxEntries = maxEntries / shardCount
			if i < maxEntries%shardCount {
				s.maxEntries++
			}
		}
		if maxMemoryBytes > 0 {
			s.maxMemoryBytes = maxMemoryBytes / int64(shardCount)
			if int64(i) < maxMemoryBytes%int64(shardCount) {
				s.maxMemoryBytes++
			}
		}
		s.maxStaleness = cfg.MaxStaleness
		s.sizer = sizer
		s.staleTimeout = staleTimeout
	}
	return c
}

// LocalCache is the built-in sharded approximate-LRU cache.
//
// Experimental: this API may change in a minor release.
type LocalCache struct {
	shards     []cacheShard
	shardCount uint32
	shardMask  uint32
	sizer      CacheSizer

	nextToken atomic.Uint64
	hits      atomic.Uint64
	misses    atomic.Uint64
}

var _ Cache = (*LocalCache)(nil)

// cacheShard holds the state for one shard of LocalCache. The mutex
// protects entries, byRedisKey, byConnID, and usedBytes.
type cacheShard struct {
	mu         sync.RWMutex
	entries    map[string]*cacheEntry
	byRedisKey map[string]map[string]struct{}
	// byConnID is the owning-conn reverse index (twin of byRedisKey): conn id ->
	// its cache keys. Populated by FulfillOwned, cleaned in removeEntryLocked,
	// consumed by EvictByConn.
	byConnID  map[uint64]map[string]struct{}
	usedBytes int64

	maxEntries     int
	maxMemoryBytes int64
	maxStaleness   time.Duration
	sizer          CacheSizer
	staleTimeout   time.Duration
}

// shardFor returns the shard responsible for cacheKey.
func (c *LocalCache) shardFor(cacheKey string) *cacheShard {
	if c.shardCount == 1 {
		return &c.shards[0]
	}
	return &c.shards[fnv1a32(cacheKey)&c.shardMask]
}

// fnv1a32 returns the FNV-1a 32-bit hash of s. Allocation-free.
func fnv1a32(s string) uint32 {
	const (
		offset uint32 = 2166136261
		prime  uint32 = 16777619
	)
	h := offset
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= prime
	}
	return h
}

const defaultCacheEntryOverhead int64 = 96

func defaultCacheSizer(cacheKey string, redisKeys []string, value []byte) int64 {
	size := defaultCacheEntryOverhead + int64(len(cacheKey)+len(value))
	for _, key := range redisKeys {
		size += int64(len(key)) + 16
	}
	if size < 0 {
		return 0
	}
	return size
}

// Get returns a copy of a cached value, waiting for an in-progress fetch when
// necessary.
func (c *LocalCache) Get(ctx context.Context, cacheKey string) ([]byte, bool) {
	if ctx == nil {
		ctx = context.Background()
	}
	value, ok := c.shardFor(cacheKey).get(ctx, cacheKey)
	if ok {
		c.hits.Add(1)
	} else {
		c.misses.Add(1)
	}
	return value, ok
}

// get is the read-side hot path. Holds only the shard's read lock; updates
// the LRU recency timestamp via atomic store on the entry — no write-lock
// upgrade is needed.
func (s *cacheShard) get(ctx context.Context, cacheKey string) ([]byte, bool) {
	for {
		s.mu.RLock()
		entry, ok := s.entries[cacheKey]
		if !ok {
			s.mu.RUnlock()
			return nil, false
		}

		if entry.state == cacheEntryInProgress {
			waitCh := entry.waitCh
			// Bound the wait by the placeholder's remaining stale window so an
			// abandoned reservation cannot block waiters indefinitely.
			remaining := s.staleTimeout - time.Since(entry.reservedAt)
			s.mu.RUnlock()
			if waitCh == nil {
				// Defensive: treat a missing waitCh as a miss to avoid busy-looping.
				return nil, false
			}
			if remaining <= 0 {
				// Placeholder already stale; miss so the caller refetches.
				return nil, false
			}
			// Wait for the in-flight fetch to either publish (Fulfill) or abort (Cancel/Delete/Flush).
			timer := time.NewTimer(remaining)
			select {
			case <-waitCh:
				timer.Stop()
			case <-ctx.Done():
				timer.Stop()
				return nil, false
			case <-timer.C:
				return nil, false
			}
			continue
		}

		if entry.state != cacheEntryValid {
			s.mu.RUnlock()
			return nil, false
		}

		// Max-staleness backstop: a Valid entry older than maxStaleness is treated
		// as a miss and evicted, so a lost invalidation or connection-lifecycle
		// staleness (Window 2) cannot keep a stale value resident past MaxStaleness.
		// Evict under the write lock so the next access re-fetches — a stale-but-present
		// entry would otherwise suppress the re-fetch via Reserve.
		if s.maxStaleness > 0 && time.Since(entry.validAt) > s.maxStaleness {
			s.mu.RUnlock()
			s.mu.Lock()
			if cur, ok := s.entries[cacheKey]; ok && cur == entry {
				s.removeEntryLocked(cacheKey)
			}
			s.mu.Unlock()
			return nil, false
		}

		value := cloneBytes(entry.value)
		// Record access timestamp without upgrading the lock. Last writer
		// wins; cross-goroutine ordering of timestamps is fine for
		// approximate-LRU semantics.
		entry.lastAccessNs.Store(nextLRUToken())
		s.mu.RUnlock()
		return value, true
	}
}

// Stats returns cumulative activity and current residency.
func (c *LocalCache) Stats() CSCStats {
	return CSCStats{
		Hits:             c.hits.Load(),
		Misses:           c.misses.Load(),
		Entries:          c.Len(),
		MemoryUsageBytes: c.MemoryUsage(),
	}
}

// Reserve claims a missing cache key for fetching.
func (c *LocalCache) Reserve(cacheKey string, redisKeys []string) (token uint64, shouldFetch bool) {
	keysCopy := cloneStrings(redisKeys)
	waitCh := make(chan struct{})
	reservedAt := time.Now()
	sizeBytes := c.sizer(cacheKey, keysCopy, nil)
	if sizeBytes < 0 {
		sizeBytes = 0
	}
	newToken := c.nextToken.Add(1)

	s := c.shardFor(cacheKey)
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry, ok := s.entries[cacheKey]; ok {
		switch entry.state {
		case cacheEntryValid:
			// Existing-VALID hit: record access; caller will re-Get to
			// retrieve.
			entry.lastAccessNs.Store(nextLRUToken())
			return 0, false
		case cacheEntryInProgress:
			if time.Since(entry.reservedAt) < s.staleTimeout {
				return 0, false
			}
			s.removeEntryLocked(cacheKey)
		default:
			return 0, false
		}
	}

	if s.maxMemoryBytes > 0 && sizeBytes > s.maxMemoryBytes {
		return 0, true
	}

	entry := &cacheEntry{
		cacheKey:   cacheKey,
		redisKeys:  keysCopy,
		state:      cacheEntryInProgress,
		token:      newToken,
		reservedAt: reservedAt,
		waitCh:     waitCh,
		sizeBytes:  sizeBytes,
	}
	entry.lastAccessNs.Store(nextLRUToken())

	s.setEntryLocked(entry)
	// Evict only Valid victims. If still over capacity the shard holds only
	// in-flight placeholders: rather than abort a peer's fetch, drop this
	// reservation (the caller fetches uncached). The hard cap holds either way.
	s.evictValidLocked()
	if s.overCapacityLocked() {
		s.removeEntryLocked(cacheKey)
		return 0, true
	}
	if s.entries[cacheKey] != entry {
		return 0, true
	}
	return newToken, true
}

// FulfillOwned publishes a reserved value and records ownerConnID so
// EvictByConn can drop it when that connection is removed. ownerConnID == 0
// leaves the value unowned.
func (c *LocalCache) FulfillOwned(cacheKey string, token, ownerConnID uint64, value []byte) bool {
	return c.fulfill(cacheKey, token, ownerConnID, value)
}

func (c *LocalCache) fulfill(cacheKey string, token, ownerConnID uint64, value []byte) bool {
	valueCopy := cloneBytes(value)

	s := c.shardFor(cacheKey)
	s.mu.Lock()
	defer s.mu.Unlock()

	entry, ok := s.entries[cacheKey]
	if !ok || entry.state != cacheEntryInProgress || entry.token != token {
		return false
	}

	valueSize := s.sizer(cacheKey, entry.redisKeys, valueCopy)
	if valueSize < 0 {
		valueSize = 0
	}
	if s.maxMemoryBytes > 0 && valueSize > s.maxMemoryBytes {
		s.removeEntryLocked(cacheKey)
		return false
	}

	s.usedBytes += valueSize - entry.sizeBytes
	entry.value = valueCopy
	entry.sizeBytes = valueSize
	entry.state = cacheEntryValid
	entry.validAt = time.Now()
	entry.token = 0
	entry.lastAccessNs.Store(nextLRUToken())
	if ownerConnID != 0 {
		entry.ownerConnID = ownerConnID
		s.indexConnLocked(ownerConnID, cacheKey)
	}
	s.closeWaitersLocked(entry)

	s.evictIfNeededLocked()
	current, stillExists := s.entries[cacheKey]
	return stillExists && current == entry && entry.state == cacheEntryValid
}

// EvictByConn removes every entry fetched by connID and returns the count.
// Called when a conn is removed/swapped: the server stops delivering those
// keys' invalidations, so keeping them risks stale serves. Errs toward a miss.
func (c *LocalCache) EvictByConn(connID uint64) int {
	if connID == 0 {
		return 0
	}
	removed := 0
	for i := range c.shards {
		removed += c.shards[i].evictByConn(connID)
	}
	return removed
}

func (s *cacheShard) evictByConn(connID uint64) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	cacheKeys, ok := s.byConnID[connID]
	if !ok {
		return 0
	}
	toRemove := make([]string, 0, len(cacheKeys))
	for cacheKey := range cacheKeys {
		toRemove = append(toRemove, cacheKey)
	}
	removed := 0
	for _, cacheKey := range toRemove {
		if s.removeEntryLocked(cacheKey) {
			removed++
		}
	}
	return removed
}

// indexConnLocked records cacheKey under connID in the owning-connection index.
func (s *cacheShard) indexConnLocked(connID uint64, cacheKey string) {
	cacheKeys := s.byConnID[connID]
	if cacheKeys == nil {
		cacheKeys = make(map[string]struct{})
		s.byConnID[connID] = cacheKeys
	}
	cacheKeys[cacheKey] = struct{}{}
}

// Cancel removes the reservation matching token.
func (c *LocalCache) Cancel(cacheKey string, token uint64) bool {
	s := c.shardFor(cacheKey)
	s.mu.Lock()
	defer s.mu.Unlock()

	entry, ok := s.entries[cacheKey]
	if !ok || entry.state != cacheEntryInProgress || entry.token != token {
		return false
	}

	s.removeEntryLocked(cacheKey)
	return true
}

// DeleteByRedisKey removes entries associated with redisKey.
func (c *LocalCache) DeleteByRedisKey(redisKey string) int {
	removed := 0
	for i := range c.shards {
		removed += c.shards[i].deleteByRedisKey(redisKey)
	}
	return removed
}

func (s *cacheShard) deleteByRedisKey(redisKey string) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	cacheKeys, ok := s.byRedisKey[redisKey]
	if !ok {
		return 0
	}

	// Remove IN_PROGRESS placeholders too: an invalidation can arrive on a
	// different stream than the in-flight reply (the background drainer), so the
	// fetch may predate the write. Removing makes the racing Fulfill fail and
	// waiters refetch, so a raced-invalidation value is never published.
	toRemove := make([]string, 0, len(cacheKeys))
	for cacheKey := range cacheKeys {
		toRemove = append(toRemove, cacheKey)
	}

	removed := 0
	for _, cacheKey := range toRemove {
		if s.removeEntryLocked(cacheKey) {
			removed++
		}
	}
	return removed
}

// DeleteByCacheKey removes one entry by its internal cache key.
func (c *LocalCache) DeleteByCacheKey(cacheKey string) bool {
	s := c.shardFor(cacheKey)
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.removeEntryLocked(cacheKey)
}

// Flush removes all entries.
func (c *LocalCache) Flush() int {
	removed := 0
	for i := range c.shards {
		removed += c.shards[i].flush()
	}
	return removed
}

func (s *cacheShard) flush() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Flush placeholders too (see deleteByRedisKey): a flush (FLUSHDB, or the
	// owned-cache flush on Close) means everything, including in-flight fetches,
	// may be stale.
	removed := 0
	for cacheKey := range s.entries {
		if s.removeEntryLocked(cacheKey) {
			removed++
		}
	}
	return removed
}

// Len returns the current number of entries and reservations.
func (c *LocalCache) Len() int {
	n := 0
	for i := range c.shards {
		s := &c.shards[i]
		s.mu.RLock()
		n += len(s.entries)
		s.mu.RUnlock()
	}
	return n
}

// MemoryUsage returns the cache's estimated memory usage in bytes.
func (c *LocalCache) MemoryUsage() int64 {
	var total int64
	for i := range c.shards {
		s := &c.shards[i]
		s.mu.RLock()
		total += s.usedBytes
		s.mu.RUnlock()
	}
	return total
}

func (s *cacheShard) setEntryLocked(entry *cacheEntry) {
	if old, exists := s.entries[entry.cacheKey]; exists {
		s.removeEntryLocked(old.cacheKey)
	}

	s.entries[entry.cacheKey] = entry
	s.usedBytes += entry.sizeBytes

	for _, redisKey := range entry.redisKeys {
		cacheKeys := s.byRedisKey[redisKey]
		if cacheKeys == nil {
			cacheKeys = make(map[string]struct{})
			s.byRedisKey[redisKey] = cacheKeys
		}
		cacheKeys[entry.cacheKey] = struct{}{}
	}
}

func (s *cacheShard) removeEntryLocked(cacheKey string) bool {
	entry, exists := s.entries[cacheKey]
	if !exists {
		return false
	}

	delete(s.entries, cacheKey)
	s.usedBytes -= entry.sizeBytes
	if s.usedBytes < 0 {
		s.usedBytes = 0
	}

	for _, redisKey := range entry.redisKeys {
		cacheKeys := s.byRedisKey[redisKey]
		if cacheKeys == nil {
			continue
		}
		delete(cacheKeys, cacheKey)
		if len(cacheKeys) == 0 {
			delete(s.byRedisKey, redisKey)
		}
	}

	if entry.ownerConnID != 0 {
		if cacheKeys := s.byConnID[entry.ownerConnID]; cacheKeys != nil {
			delete(cacheKeys, cacheKey)
			if len(cacheKeys) == 0 {
				delete(s.byConnID, entry.ownerConnID)
			}
		}
	}

	s.closeWaitersLocked(entry)
	return true
}

func (s *cacheShard) closeWaitersLocked(entry *cacheEntry) {
	if entry.waitCh != nil && !entry.waitClosed {
		close(entry.waitCh)
		entry.waitClosed = true
	}
}

func (s *cacheShard) overCapacityLocked() bool {
	if s.maxEntries > 0 && len(s.entries) > s.maxEntries {
		return true
	}
	if s.maxMemoryBytes > 0 && s.usedBytes > s.maxMemoryBytes {
		return true
	}
	return false
}

// evictIfNeededLocked evicts by approximate LRU (O(N) scan; rare in
// well-sized caches) until under capacity. Used by Set/Fulfill: it prefers a
// Valid victim but falls back to the oldest IN_PROGRESS placeholder to keep the
// hard cap (that placeholder's Fulfill then fails and its waiters refetch).
func (s *cacheShard) evictIfNeededLocked() {
	for s.overCapacityLocked() {
		victim := s.oldestLocked(cacheEntryValid)
		if victim == nil {
			victim = s.oldestLocked(cacheEntryInProgress)
		}
		if victim == nil {
			return
		}
		s.removeEntryLocked(victim.cacheKey)
	}
}

// evictValidLocked evicts only Valid entries until under capacity. Unlike
// evictIfNeededLocked it never evicts a placeholder, so Reserve can't abort a
// peer's in-flight fetch.
func (s *cacheShard) evictValidLocked() {
	for s.overCapacityLocked() {
		victim := s.oldestLocked(cacheEntryValid)
		if victim == nil {
			return
		}
		s.removeEntryLocked(victim.cacheKey)
	}
}

// oldestLocked returns the entry in the given state with the smallest
// lastAccessNs (the least-recently-used), or nil when none exists.
func (s *cacheShard) oldestLocked(state cacheEntryState) *cacheEntry {
	var victim *cacheEntry
	var oldestNs int64 = math.MaxInt64
	for _, e := range s.entries {
		if e.state != state {
			continue
		}
		if ns := e.lastAccessNs.Load(); ns < oldestNs {
			oldestNs = ns
			victim = e
		}
	}
	return victim
}

func cloneBytes(src []byte) []byte {
	if src == nil {
		return nil
	}
	dst := make([]byte, len(src))
	copy(dst, src)
	return dst
}

func cloneStrings(src []string) []string {
	if len(src) == 0 {
		return nil
	}
	dst := make([]string, len(src))
	copy(dst, src)
	return dst
}
