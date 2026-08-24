package internal

import (
	"context"
	"sync"
	"time"
)

var semTimers = sync.Pool{
	New: func() interface{} {
		t := time.NewTimer(time.Hour)
		t.Stop()
		return t
	},
}

// putSemTimer stops a pooled timer and drains a stale fire before reuse,
// portably across both timer-channel semantics (the drain must never block):
//
//   - main module on go >= 1.23 (synchronous channels): Stop returns TRUE for
//     an expired-but-undelivered fire — the delivery is aborted — and false
//     only once the value was actually received. With the sole receiver
//     being Acquire's own select, the drain branch is unreachable; the
//     select-with-default is a safety net so a future semantics shift cannot
//     turn it into a blocking receive (reviewed on #3942).
//   - GODEBUG=asynctimerchan=1 (consumer main module on go < 1.23, old
//     buffered channels): Stop returns false and the fired value sits in the
//     buffer; the drain consumes it so the timer is clean for Reset-reuse.
func putSemTimer(t *time.Timer) {
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
	semTimers.Put(t)
}

// FastSemaphore is a channel-based semaphore optimized for performance.
// It uses a fast path that avoids timer allocation when tokens are available.
// The channel is pre-filled with tokens: Acquire = receive, Release = send.
// Closing the semaphore unblocks all waiting goroutines.
//
// Performance: ~30 ns/op with zero allocations on fast path.
// Fairness: Eventual fairness (no starvation) but not strict FIFO.
type FastSemaphore struct {
	tokens chan struct{}
	max    int32
}

// NewFastSemaphore creates a new fast semaphore with the given capacity.
func NewFastSemaphore(capacity int32) *FastSemaphore {
	ch := make(chan struct{}, capacity)
	// Pre-fill with tokens
	for i := int32(0); i < capacity; i++ {
		ch <- struct{}{}
	}
	return &FastSemaphore{
		tokens: ch,
		max:    capacity,
	}
}

// TryAcquire attempts to acquire a token without blocking.
// Returns true if successful, false if no tokens available.
func (s *FastSemaphore) TryAcquire() bool {
	select {
	case <-s.tokens:
		return true
	default:
		return false
	}
}

// Acquire acquires a token, blocking if necessary until one is available.
// Returns an error if the context is cancelled or the timeout expires.
// Uses a fast path to avoid timer allocation when tokens are immediately available.
func (s *FastSemaphore) Acquire(ctx context.Context, timeout time.Duration, timeoutErr error) error {
	// Check context first
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Try fast path first (no timer needed)
	select {
	case <-s.tokens:
		return nil
	default:
	}

	// Slow path: need to wait with timeout
	timer := semTimers.Get().(*time.Timer)
	defer putSemTimer(timer)
	timer.Reset(timeout)

	select {
	case <-s.tokens:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return timeoutErr
	}
}

// AcquireBlocking acquires a token, blocking indefinitely until one is available.
func (s *FastSemaphore) AcquireBlocking() {
	<-s.tokens
}

// Release releases a token back to the semaphore.
func (s *FastSemaphore) Release() {
	s.tokens <- struct{}{}
}

// Close closes the semaphore, unblocking all waiting goroutines.
// After close, all Acquire calls will receive a closed channel signal.
func (s *FastSemaphore) Close() {
	close(s.tokens)
}

// Len returns the current number of acquired tokens.
func (s *FastSemaphore) Len() int32 {
	return s.max - int32(len(s.tokens))
}

// FIFOSemaphore is a channel-based semaphore with strict FIFO ordering.
// Unlike FastSemaphore, this guarantees that threads are served in the exact order they call Acquire().
// The channel is pre-filled with tokens: Acquire = receive, Release = send.
// Closing the semaphore unblocks all waiting goroutines.
//
// Performance: ~115 ns/op with zero allocations (slower than FastSemaphore due to timer allocation).
// Fairness: Strict FIFO ordering guaranteed by Go runtime.
type FIFOSemaphore struct {
	tokens chan struct{}
	max    int32
}

// NewFIFOSemaphore creates a new FIFO semaphore with the given capacity.
func NewFIFOSemaphore(capacity int32) *FIFOSemaphore {
	ch := make(chan struct{}, capacity)
	// Pre-fill with tokens
	for i := int32(0); i < capacity; i++ {
		ch <- struct{}{}
	}
	return &FIFOSemaphore{
		tokens: ch,
		max:    capacity,
	}
}

// TryAcquire attempts to acquire a token without blocking.
// Returns true if successful, false if no tokens available.
func (s *FIFOSemaphore) TryAcquire() bool {
	select {
	case <-s.tokens:
		return true
	default:
		return false
	}
}

// Acquire acquires a token, blocking if necessary until one is available.
// Returns an error if the context is cancelled or the timeout expires.
// Always uses timer to guarantee FIFO ordering (no fast path).
func (s *FIFOSemaphore) Acquire(ctx context.Context, timeout time.Duration, timeoutErr error) error {
	// No fast path - always use timer to guarantee FIFO
	timer := semTimers.Get().(*time.Timer)
	defer putSemTimer(timer)
	timer.Reset(timeout)

	select {
	case <-s.tokens:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return timeoutErr
	}
}

// Release releases a token back to the semaphore.
func (s *FIFOSemaphore) Release() {
	s.tokens <- struct{}{}
}
