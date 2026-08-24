package maintnotifications

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/maintnotifications/logs"
)

// CircuitBreakerState represents the state of a circuit breaker
type CircuitBreakerState int32

const (
	// CircuitBreakerClosed - normal operation, requests allowed
	CircuitBreakerClosed CircuitBreakerState = iota
	// CircuitBreakerOpen - failing fast, requests rejected
	CircuitBreakerOpen
	// CircuitBreakerHalfOpen - testing if service recovered
	CircuitBreakerHalfOpen
)

func (s CircuitBreakerState) String() string {
	switch s {
	case CircuitBreakerClosed:
		return "closed"
	case CircuitBreakerOpen:
		return "open"
	case CircuitBreakerHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// CircuitBreaker implements the circuit breaker pattern for endpoint-specific failure handling
type CircuitBreaker struct {
	// Configuration
	failureThreshold int           // Number of failures before opening
	resetTimeout     time.Duration // How long to stay open before testing
	maxRequests      int           // Max requests allowed in half-open state

	// State tracking (atomic for lock-free access)
	state           atomic.Int32 // CircuitBreakerState
	failures        atomic.Int64 // Current failure count
	successes       atomic.Int64 // Success count in half-open state
	requests        atomic.Int64 // Request count in half-open state
	lastFailureTime atomic.Int64 // Unix timestamp of last failure
	lastSuccessTime atomic.Int64 // Unix timestamp of last success

	// Endpoint identification
	endpoint string
	config   *Config
}

// newCircuitBreaker creates a new circuit breaker for an endpoint
func newCircuitBreaker(endpoint string, config *Config) *CircuitBreaker {
	// Use configuration values with sensible defaults
	failureThreshold := 5
	resetTimeout := 60 * time.Second
	maxRequests := 3

	if config != nil {
		failureThreshold = config.CircuitBreakerFailureThreshold
		resetTimeout = config.CircuitBreakerResetTimeout
		maxRequests = config.CircuitBreakerMaxRequests
	}

	return &CircuitBreaker{
		failureThreshold: failureThreshold,
		resetTimeout:     resetTimeout,
		maxRequests:      maxRequests,
		endpoint:         endpoint,
		config:           config,
		state:            atomic.Int32{}, // Defaults to CircuitBreakerClosed (0)
	}
}

// IsOpen returns true if the circuit breaker is open (rejecting requests)
func (cb *CircuitBreaker) IsOpen() bool {
	state := CircuitBreakerState(cb.state.Load())
	return state == CircuitBreakerOpen
}

// shouldAttemptReset checks if enough time has passed to attempt reset
func (cb *CircuitBreaker) shouldAttemptReset() bool {
	lastFailure := time.Unix(cb.lastFailureTime.Load(), 0)
	return time.Since(lastFailure) >= cb.resetTimeout
}

// Execute runs the given function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() error) error {
	// Single atomic state load for consistency
	state := CircuitBreakerState(cb.state.Load())

	switch state {
	case CircuitBreakerOpen:
		if cb.shouldAttemptReset() {
			// Attempt transition to half-open
			if cb.state.CompareAndSwap(int32(CircuitBreakerOpen), int32(CircuitBreakerHalfOpen)) {
				cb.requests.Store(0)
				cb.successes.Store(0)
				if internal.LogLevel.InfoOrAbove() {
					internal.Logger.Printf(context.Background(), logs.CircuitBreakerTransitioningToHalfOpen(cb.endpoint))
				}
				// Fall through to half-open logic
			} else {
				return ErrCircuitBreakerOpen
			}
		} else {
			return ErrCircuitBreakerOpen
		}
		fallthrough
	case CircuitBreakerHalfOpen:
		requests := cb.requests.Add(1)
		if requests > int64(cb.maxRequests) {
			cb.requests.Add(-1) // Revert the increment
			return ErrCircuitBreakerOpen
		}
	}

	// Execute the function with consistent state
	err := fn()

	if err != nil {
		cb.recordFailure()
		return err
	}

	cb.recordSuccess()
	return nil
}

// recordFailure records a failure and potentially opens the circuit
func (cb *CircuitBreaker) recordFailure() {
	cb.lastFailureTime.Store(time.Now().Unix())
	failures := cb.failures.Add(1)

	state := CircuitBreakerState(cb.state.Load())

	switch state {
	case CircuitBreakerClosed:
		if failures >= int64(cb.failureThreshold) {
			if cb.state.CompareAndSwap(int32(CircuitBreakerClosed), int32(CircuitBreakerOpen)) {
				if internal.LogLevel.WarnOrAbove() {
					internal.Logger.Printf(context.Background(), logs.CircuitBreakerOpened(cb.endpoint, failures))
				}
			}
		}
	case CircuitBreakerHalfOpen:
		// Any failure in half-open state immediately opens the circuit
		if cb.state.CompareAndSwap(int32(CircuitBreakerHalfOpen), int32(CircuitBreakerOpen)) {
			if internal.LogLevel.WarnOrAbove() {
				internal.Logger.Printf(context.Background(), logs.CircuitBreakerReopened(cb.endpoint))
			}
		}
	}
}

// recordSuccess records a success and potentially closes the circuit
func (cb *CircuitBreaker) recordSuccess() {
	cb.lastSuccessTime.Store(time.Now().Unix())

	state := CircuitBreakerState(cb.state.Load())

	switch state {
	case CircuitBreakerClosed:
		// Reset failure count on success in closed state
		cb.failures.Store(0)
	case CircuitBreakerHalfOpen:
		successes := cb.successes.Add(1)

		// If we've had enough successful requests, close the circuit
		if successes >= int64(cb.maxRequests) {
			if cb.state.CompareAndSwap(int32(CircuitBreakerHalfOpen), int32(CircuitBreakerClosed)) {
				cb.failures.Store(0)
				if internal.LogLevel.InfoOrAbove() {
					internal.Logger.Printf(context.Background(), logs.CircuitBreakerClosed(cb.endpoint, successes))
				}
			}
		}
	}
}

// GetState returns the current state of the circuit breaker
func (cb *CircuitBreaker) GetState() CircuitBreakerState {
	return CircuitBreakerState(cb.state.Load())
}

// GetStats returns current statistics for monitoring
func (cb *CircuitBreaker) GetStats() CircuitBreakerStats {
	return CircuitBreakerStats{
		Endpoint:        cb.endpoint,
		State:           cb.GetState(),
		Failures:        cb.failures.Load(),
		Successes:       cb.successes.Load(),
		Requests:        cb.requests.Load(),
		LastFailureTime: time.Unix(cb.lastFailureTime.Load(), 0),
		LastSuccessTime: time.Unix(cb.lastSuccessTime.Load(), 0),
	}
}

// CircuitBreakerStats provides statistics about a circuit breaker
type CircuitBreakerStats struct {
	Endpoint        string
	State           CircuitBreakerState
	Failures        int64
	Successes       int64
	Requests        int64
	LastFailureTime time.Time
	LastSuccessTime time.Time
}

// CircuitBreakerEntry wraps a circuit breaker with access tracking
type CircuitBreakerEntry struct {
	breaker    *CircuitBreaker
	lastAccess atomic.Int64 // Unix timestamp
	created    time.Time
}

// CircuitBreakerManager manages circuit breakers for multiple endpoints
type CircuitBreakerManager struct {
	breakers    sync.Map // map[string]*CircuitBreakerEntry
	config      *Config
	cleanupStop chan struct{}
	cleanupMu   sync.Mutex
	lastCleanup atomic.Int64 // Unix timestamp
}

// newCircuitBreakerManager creates a new circuit breaker manager
func newCircuitBreakerManager(config *Config) *CircuitBreakerManager {
	cbm := &CircuitBreakerManager{
		config:      config,
		cleanupStop: make(chan struct{}),
	}
	cbm.lastCleanup.Store(time.Now().Unix())

	// Start background cleanup goroutine
	go cbm.cleanupLoop()

	return cbm
}

// GetCircuitBreaker returns the circuit breaker for an endpoint, creating it if necessary
func (cbm *CircuitBreakerManager) GetCircuitBreaker(endpoint string) *CircuitBreaker {
	now := time.Now().Unix()

	if entry, ok := cbm.breakers.Load(endpoint); ok {
		cbEntry := entry.(*CircuitBreakerEntry)
		cbEntry.lastAccess.Store(now)
		return cbEntry.breaker
	}

	// Create new circuit breaker with metadata
	newBreaker := newCircuitBreaker(endpoint, cbm.config)
	newEntry := &CircuitBreakerEntry{
		breaker: newBreaker,
		created: time.Now(),
	}
	newEntry.lastAccess.Store(now)

	actual, _ := cbm.breakers.LoadOrStore(endpoint, newEntry)
	return actual.(*CircuitBreakerEntry).breaker
}

// GetAllStats returns statistics for all circuit breakers
func (cbm *CircuitBreakerManager) GetAllStats() []CircuitBreakerStats {
	var stats []CircuitBreakerStats
	cbm.breakers.Range(func(key, value interface{}) bool {
		entry := value.(*CircuitBreakerEntry)
		stats = append(stats, entry.breaker.GetStats())
		return true
	})
	return stats
}

// cleanupLoop runs background cleanup of unused circuit breakers
func (cbm *CircuitBreakerManager) cleanupLoop() {
	ticker := time.NewTicker(5 * time.Minute) // Cleanup every 5 minutes
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cbm.cleanup()
		case <-cbm.cleanupStop:
			return
		}
	}
}

// cleanup removes circuit breakers that haven't been accessed recently
func (cbm *CircuitBreakerManager) cleanup() {
	// Prevent concurrent cleanups
	if !cbm.cleanupMu.TryLock() {
		return
	}
	defer cbm.cleanupMu.Unlock()

	now := time.Now()
	cutoff := now.Add(-30 * time.Minute).Unix() // 30 minute TTL

	var toDelete []string
	count := 0

	cbm.breakers.Range(func(key, value interface{}) bool {
		endpoint := key.(string)
		entry := value.(*CircuitBreakerEntry)

		count++

		// Remove if not accessed recently
		if entry.lastAccess.Load() < cutoff {
			toDelete = append(toDelete, endpoint)
		}

		return true
	})

	// Delete expired entries
	for _, endpoint := range toDelete {
		cbm.breakers.Delete(endpoint)
	}

	// Log cleanup results
	if len(toDelete) > 0 && internal.LogLevel.InfoOrAbove() {
		internal.Logger.Printf(context.Background(), logs.CircuitBreakerCleanup(len(toDelete), count))
	}

	cbm.lastCleanup.Store(now.Unix())
}

// Shutdown stops the cleanup goroutine
func (cbm *CircuitBreakerManager) Shutdown() {
	close(cbm.cleanupStop)
}

// Reset resets all circuit breakers (useful for testing)
func (cbm *CircuitBreakerManager) Reset() {
	cbm.breakers.Range(func(key, value interface{}) bool {
		entry := value.(*CircuitBreakerEntry)
		breaker := entry.breaker
		breaker.state.Store(int32(CircuitBreakerClosed))
		breaker.failures.Store(0)
		breaker.successes.Store(0)
		breaker.requests.Store(0)
		breaker.lastFailureTime.Store(0)
		breaker.lastSuccessTime.Store(0)
		return true
	})
}
