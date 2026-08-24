package pool

import (
	"container/list"
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
)

// ConnState represents the connection state in the state machine.
// States are designed to be lightweight and fast to check.
//
// State Transitions:
//
//	CREATED → INITIALIZING → IDLE ⇄ IN_USE
//	                           ↓
//	                       UNUSABLE (handoff/reauth)
//	                           ↓
//	                        IDLE/CLOSED
type ConnState uint32

const (
	// StateCreated - Connection just created, not yet initialized
	StateCreated ConnState = iota

	// StateInitializing - Connection initialization in progress
	StateInitializing

	// StateIdle - Connection initialized and idle in pool, ready to be acquired
	StateIdle

	// StateInUse - Connection actively processing a command (retrieved from pool)
	StateInUse

	// StateUnusable - Connection temporarily unusable due to background operation
	// (handoff, reauth, etc.). Cannot be acquired from pool.
	StateUnusable

	// StateClosed - Connection closed
	StateClosed
)

// Predefined state slices to avoid allocations in hot paths
var (
	validFromInUse              = []ConnState{StateInUse}
	validFromCreatedOrIdle      = []ConnState{StateCreated, StateIdle}
	validFromCreatedInUseOrIdle = []ConnState{StateCreated, StateInUse, StateIdle}
	// For AwaitAndTransition calls
	validFromCreatedIdleOrUnusable = []ConnState{StateCreated, StateIdle, StateUnusable}
	validFromIdle                  = []ConnState{StateIdle}
	// For CompareAndSwapUsable
	validFromInitializingOrUnusable = []ConnState{StateInitializing, StateUnusable}
)

// Accessor functions for predefined slices to avoid allocations in external packages
// These return the same slice instance, so they're zero-allocation

// ValidFromIdle returns a predefined slice containing only StateIdle.
// Use this to avoid allocations when calling AwaitAndTransition or TryTransition.
func ValidFromIdle() []ConnState {
	return validFromIdle
}

// ValidFromCreatedIdleOrUnusable returns a predefined slice for initialization transitions.
// Use this to avoid allocations when calling AwaitAndTransition or TryTransition.
func ValidFromCreatedIdleOrUnusable() []ConnState {
	return validFromCreatedIdleOrUnusable
}

// String returns a human-readable string representation of the state.
func (s ConnState) String() string {
	switch s {
	case StateCreated:
		return "CREATED"
	case StateInitializing:
		return "INITIALIZING"
	case StateIdle:
		return "IDLE"
	case StateInUse:
		return "IN_USE"
	case StateUnusable:
		return "UNUSABLE"
	case StateClosed:
		return "CLOSED"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", s)
	}
}

var (
	// ErrInvalidStateTransition is returned when a state transition is not allowed
	ErrInvalidStateTransition = errors.New("invalid state transition")

	// ErrStateMachineClosed is returned when operating on a closed state machine
	ErrStateMachineClosed = errors.New("state machine is closed")

	// ErrTimeout is returned when a state transition times out
	ErrTimeout = errors.New("state transition timeout")
)

// waiter represents a goroutine waiting for a state transition.
// Designed for minimal allocations and fast processing.
type waiter struct {
	validStates map[ConnState]struct{} // States we're waiting for
	targetState ConnState              // State to transition to
	done        chan error             // Signaled when transition completes or times out
}

// ConnStateMachine manages connection state transitions with FIFO waiting queue.
// Optimized for:
// - Lock-free reads (hot path)
// - Minimal allocations
// - Fast state transitions
// - FIFO fairness for waiters
// Note: Handoff metadata (endpoint, seqID, retries) is managed separately in the Conn struct.
type ConnStateMachine struct {
	// Current state - atomic for lock-free reads
	state atomic.Uint32

	// FIFO queue for waiters - only locked during waiter add/remove/notify
	mu          sync.Mutex
	waiters     *list.List   // List of *waiter
	waiterCount atomic.Int32 // Fast lock-free check for waiters (avoids mutex in hot path)
}

// NewConnStateMachine creates a new connection state machine.
// Initial state is StateCreated.
func NewConnStateMachine() *ConnStateMachine {
	sm := &ConnStateMachine{
		waiters: list.New(),
	}
	sm.state.Store(uint32(StateCreated))
	return sm
}

// GetState returns the current state (lock-free read).
// This is the hot path - optimized for zero allocations and minimal overhead.
// Note: Zero allocations applies to state reads; converting the returned state to a string
// (via String()) may allocate if the state is unknown.
func (sm *ConnStateMachine) GetState() ConnState {
	return ConnState(sm.state.Load())
}

// TryTransitionFast is an optimized version for the hot path (Get/Put operations).
// It only handles simple state transitions without waiter notification.
// This is safe because:
// 1. Get/Put don't need to wait for state changes
// 2. Background operations (handoff/reauth) use UNUSABLE state, which this won't match
// 3. If a background operation is in progress (state is UNUSABLE), this fails fast
//
// Returns true if transition succeeded, false otherwise.
// Use this for performance-critical paths where you don't need error details.
//
// Performance: Single CAS operation - as fast as the old atomic bool!
// For multiple from states, use: sm.TryTransitionFast(State1, Target) || sm.TryTransitionFast(State2, Target)
// The || operator short-circuits, so only 1 CAS is executed in the common case.
func (sm *ConnStateMachine) TryTransitionFast(fromState, targetState ConnState) bool {
	return sm.state.CompareAndSwap(uint32(fromState), uint32(targetState))
}

// TryTransition attempts an immediate state transition without waiting.
// Returns the current state after the transition attempt and an error if the transition failed.
// The returned state is the CURRENT state (after the attempt), not the previous state.
// This is faster than AwaitAndTransition when you don't need to wait.
// Uses compare-and-swap to atomically transition, preventing concurrent transitions.
// This method does NOT wait - it fails immediately if the transition cannot be performed.
//
// Performance: Zero allocations on success path (hot path).
func (sm *ConnStateMachine) TryTransition(validFromStates []ConnState, targetState ConnState) (ConnState, error) {
	// Try each valid from state with CAS
	// This ensures only ONE goroutine can successfully transition at a time
	for _, fromState := range validFromStates {
		// Try to atomically swap from fromState to targetState
		// If successful, we won the race and can proceed
		if sm.state.CompareAndSwap(uint32(fromState), uint32(targetState)) {
			// Success! We transitioned atomically
			// Hot path optimization: only check for waiters if transition succeeded
			// This avoids atomic load on every Get/Put when no waiters exist
			if sm.waiterCount.Load() > 0 {
				sm.notifyWaiters()
			}
			return targetState, nil
		}
	}

	// All CAS attempts failed - state is not valid for this transition
	// Return the current state so caller can decide what to do
	// Note: This error path allocates, but it's the exceptional case
	currentState := sm.GetState()
	return currentState, fmt.Errorf("%w: cannot transition from %s to %s (valid from: %v)",
		ErrInvalidStateTransition, currentState, targetState, validFromStates)
}

// Transition unconditionally transitions to the target state.
// Use with caution - prefer AwaitAndTransition or TryTransition for safety.
// This is useful for error paths or when you know the transition is valid.
func (sm *ConnStateMachine) Transition(targetState ConnState) {
	sm.state.Store(uint32(targetState))
	sm.notifyWaiters()
}

// AwaitAndTransition waits for the connection to reach one of the valid states,
// then atomically transitions to the target state.
// Returns the current state after the transition attempt and an error if the operation failed.
// The returned state is the CURRENT state (after the attempt), not the previous state.
// Returns error if timeout expires or context is cancelled.
//
// This method implements FIFO fairness - the first caller to wait gets priority
// when the state becomes available.
//
// Performance notes:
// - If already in a valid state, this is very fast (no allocation, no waiting)
// - If waiting is required, allocates one waiter struct and one channel
func (sm *ConnStateMachine) AwaitAndTransition(
	ctx context.Context,
	validFromStates []ConnState,
	targetState ConnState,
) (ConnState, error) {
	// Fast path: try immediate transition with CAS to prevent race conditions
	// BUT: only if there are no waiters in the queue (to maintain FIFO ordering)
	if sm.waiterCount.Load() == 0 {
		for _, fromState := range validFromStates {
			// Check if we're already in target state
			if fromState == targetState && sm.GetState() == targetState {
				return targetState, nil
			}

			// Try to atomically swap from fromState to targetState
			if sm.state.CompareAndSwap(uint32(fromState), uint32(targetState)) {
				// Success! We transitioned atomically
				sm.notifyWaiters()
				return targetState, nil
			}
		}
	}

	// Fast path failed - check if we should wait or fail
	currentState := sm.GetState()

	// Check if closed
	if currentState == StateClosed {
		return currentState, ErrStateMachineClosed
	}

	// Slow path: need to wait for state change
	// Create waiter with valid states map for fast lookup
	validStatesMap := make(map[ConnState]struct{}, len(validFromStates))
	for _, s := range validFromStates {
		validStatesMap[s] = struct{}{}
	}

	w := &waiter{
		validStates: validStatesMap,
		targetState: targetState,
		done:        make(chan error, 1), // Buffered to avoid goroutine leak
	}

	// Add to FIFO queue
	sm.mu.Lock()
	elem := sm.waiters.PushBack(w)
	sm.waiterCount.Add(1)
	sm.mu.Unlock()

	// Wait for state change or timeout
	select {
	case <-ctx.Done():
		// Timeout or cancellation - remove from queue
		sm.mu.Lock()
		sm.waiters.Remove(elem)
		sm.waiterCount.Add(-1)
		sm.mu.Unlock()
		return sm.GetState(), ctx.Err()
	case err := <-w.done:
		// Transition completed (or failed)
		// Note: waiterCount is decremented either in notifyWaiters (when the waiter is notified and removed)
		// or here (on timeout/cancellation).
		return sm.GetState(), err
	}
}

// notifyWaiters checks if any waiters can proceed and notifies them in FIFO order.
// This is called after every state transition.
func (sm *ConnStateMachine) notifyWaiters() {
	// Fast path: check atomic counter without acquiring lock
	// This eliminates mutex overhead in the common case (no waiters)
	if sm.waiterCount.Load() == 0 {
		return
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Double-check after acquiring lock (waiters might have been processed)
	if sm.waiters.Len() == 0 {
		return
	}

	// Track state locally so we only consider transitions made within this
	// call, not concurrent transitions from woken goroutines. Re-reading the
	// atomic would let a fast goroutine's Transition(StateIdle) leak into our
	// view, causing us to wake multiple waiters at once and breaking FIFO
	// execution ordering.
	currentState := sm.GetState()

	for {
		processed := false

		for elem := sm.waiters.Front(); elem != nil; elem = elem.Next() {
			w := elem.Value.(*waiter)

			if _, valid := w.validStates[currentState]; valid {
				sm.waiters.Remove(elem)
				sm.waiterCount.Add(-1)

				if sm.state.CompareAndSwap(uint32(currentState), uint32(w.targetState)) {
					w.done <- nil
					currentState = w.targetState
					processed = true
					break
				} else {
					sm.waiters.PushFront(w)
					sm.waiterCount.Add(1)
					currentState = sm.GetState()
					processed = true
					break
				}
			}
		}

		if !processed {
			break
		}
	}
}
