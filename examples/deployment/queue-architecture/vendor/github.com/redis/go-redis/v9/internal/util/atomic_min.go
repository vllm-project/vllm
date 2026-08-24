package util

/*
© 2023–present Harald Rudell <harald.rudell@gmail.com> (https://haraldrudell.github.io/haraldrudell/)
ISC License

Modified by htemelski-redis
Adapted from the modified atomic_max, but with inverted logic
*/

import (
	"math"

	"go.uber.org/atomic"
)

// AtomicMin is a thread-safe Min container
//   - hasValue indicator true if a value was equal to or greater than threshold
//   - optional threshold for minimum accepted Min value
//   - —
//   - wait-free CompareAndSwap mechanic
type AtomicMin struct {

	// value is current Min
	value atomic.Float64
	// whether [AtomicMin.Value] has been invoked
	// with value equal or greater to threshold
	hasValue atomic.Bool
}

// NewAtomicMin returns a thread-safe Min container
//   - if threshold is not used, AtomicMin is initialization-free
func NewAtomicMin() (atomicMin *AtomicMin) {
	m := AtomicMin{}
	m.value.Store(math.MaxFloat64)
	return &m
}

// Value updates the container with a possible Min value
//   - isNewMin is true if:
//   - — value is equal to or greater than any threshold and
//   - — invocation recorded the first 0 or
//   - — a new Min
//   - upon return, Min and Min1 are guaranteed to reflect the invocation
//   - the return order of concurrent Value invocations is not guaranteed
//   - Thread-safe
func (m *AtomicMin) Value(value float64) (isNewMin bool) {
	//  math.MaxFloat64 as Min case
	var hasValue0 = m.hasValue.Load()
	if value == math.MaxFloat64 {
		if !hasValue0 {
			isNewMin = m.hasValue.CompareAndSwap(false, true)
		}
		return // math.MaxFloat64 as Min: isNewMin true for first 0 writer
	}

	// check against present value
	var current = m.value.Load()
	if isNewMin = value < current; !isNewMin {
		return // not a new Min return: isNewMin false
	}

	// store the new Min
	for {

		// try to write value to *Min
		if isNewMin = m.value.CompareAndSwap(current, value); isNewMin {
			if !hasValue0 {
				// may be rarely written multiple times
				// still faster than CompareAndSwap
				m.hasValue.Store(true)
			}
			return // new Min written return: isNewMin true
		}
		if current = m.value.Load(); current <= value {
			return // no longer a need to write return: isNewMin false
		}
	}
}

// Min returns current min and value-present flag
//   - hasValue true indicates that value reflects a Value invocation
//   - hasValue false: value is zero-value
//   - Thread-safe
func (m *AtomicMin) Min() (value float64, hasValue bool) {
	if hasValue = m.hasValue.Load(); !hasValue {
		return
	}
	value = m.value.Load()
	return
}

// Min1 returns current Minimum whether zero-value or set by Value
//   - threshold is ignored
//   - Thread-safe
func (m *AtomicMin) Min1() (value float64) { return m.value.Load() }
