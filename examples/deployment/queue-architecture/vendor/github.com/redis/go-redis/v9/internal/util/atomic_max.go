/*
© 2023–present Harald Rudell <harald.rudell@gmail.com> (https://haraldrudell.github.io/haraldrudell/)
ISC License

Modified by htemelski-redis
Removed the treshold, adapted it to work with float64
*/

package util

import (
	"math"

	"go.uber.org/atomic"
)

// AtomicMax is a thread-safe max container
//   - hasValue indicator true if a value was equal to or greater than threshold
//   - optional threshold for minimum accepted max value
//   - if threshold is not used, initialization-free
//   - —
//   - wait-free CompareAndSwap mechanic
type AtomicMax struct {

	// value is current max
	value atomic.Float64
	// whether [AtomicMax.Value] has been invoked
	// with value equal or greater to threshold
	hasValue atomic.Bool
}

// NewAtomicMax returns a thread-safe max container
//   - if threshold is not used, AtomicMax is initialization-free
func NewAtomicMax() (atomicMax *AtomicMax) {
	m := AtomicMax{}
	m.value.Store((-math.MaxFloat64))
	return &m
}

// Value updates the container with a possible max value
//   - isNewMax is true if:
//   - — value is equal to or greater than any threshold and
//   - — invocation recorded the first 0 or
//   - — a new max
//   - upon return, Max and Max1 are guaranteed to reflect the invocation
//   - the return order of concurrent Value invocations is not guaranteed
//   - Thread-safe
func (m *AtomicMax) Value(value float64) (isNewMax bool) {
	//  -math.MaxFloat64 as max case
	var hasValue0 = m.hasValue.Load()
	if value == (-math.MaxFloat64) {
		if !hasValue0 {
			isNewMax = m.hasValue.CompareAndSwap(false, true)
		}
		return // -math.MaxFloat64 as max: isNewMax true for first 0 writer
	}

	// check against present value
	var current = m.value.Load()
	if isNewMax = value > current; !isNewMax {
		return // not a new max return: isNewMax false
	}

	// store the new max
	for {

		// try to write value to *max
		if isNewMax = m.value.CompareAndSwap(current, value); isNewMax {
			if !hasValue0 {
				// may be rarely written multiple times
				// still faster than CompareAndSwap
				m.hasValue.Store(true)
			}
			return // new max written return: isNewMax true
		}
		if current = m.value.Load(); current >= value {
			return // no longer a need to write return: isNewMax false
		}
	}
}

// Max returns current max and value-present flag
//   - hasValue true indicates that value reflects a Value invocation
//   - hasValue false: value is zero-value
//   - Thread-safe
func (m *AtomicMax) Max() (value float64, hasValue bool) {
	if hasValue = m.hasValue.Load(); !hasValue {
		return
	}
	value = m.value.Load()
	return
}

// Max1 returns current maximum whether zero-value or set by Value
//   - threshold is ignored
//   - Thread-safe
func (m *AtomicMax) Max1() (value float64) { return m.value.Load() }
