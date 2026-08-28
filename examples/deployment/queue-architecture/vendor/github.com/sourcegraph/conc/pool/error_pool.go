package pool

import (
	"context"
	"errors"
	"sync"
)

// ErrorPool is a pool that runs tasks that may return an error.
// Errors are collected and returned by Wait().
//
// The configuration methods (With*) will panic if they are used after calling
// Go() for the first time.
//
// A new ErrorPool should be created using `New().WithErrors()`.
type ErrorPool struct {
	pool Pool

	onlyFirstError bool

	mu   sync.Mutex
	errs []error
}

// Go submits a task to the pool. If all goroutines in the pool
// are busy, a call to Go() will block until the task can be started.
func (p *ErrorPool) Go(f func() error) {
	p.pool.Go(func() {
		p.addErr(f())
	})
}

// Wait cleans up any spawned goroutines, propagating any panics and
// returning any errors from tasks.
func (p *ErrorPool) Wait() error {
	p.pool.Wait()

	errs := p.errs
	p.errs = nil // reset errs

	if len(errs) == 0 {
		return nil
	} else if p.onlyFirstError {
		return errs[0]
	} else {
		return errors.Join(errs...)
	}
}

// WithContext converts the pool to a ContextPool for tasks that should
// run under the same context, such that they each respect shared cancellation.
// For example, WithCancelOnError can be configured on the returned pool to
// signal that all goroutines should be cancelled upon the first error.
func (p *ErrorPool) WithContext(ctx context.Context) *ContextPool {
	p.panicIfInitialized()
	ctx, cancel := context.WithCancel(ctx)
	return &ContextPool{
		errorPool: p.deref(),
		ctx:       ctx,
		cancel:    cancel,
	}
}

// WithFirstError configures the pool to only return the first error
// returned by a task. By default, Wait() will return a combined error.
func (p *ErrorPool) WithFirstError() *ErrorPool {
	p.panicIfInitialized()
	p.onlyFirstError = true
	return p
}

// WithMaxGoroutines limits the number of goroutines in a pool.
// Defaults to unlimited. Panics if n < 1.
func (p *ErrorPool) WithMaxGoroutines(n int) *ErrorPool {
	p.panicIfInitialized()
	p.pool.WithMaxGoroutines(n)
	return p
}

// deref is a helper that creates a shallow copy of the pool with the same
// settings. We don't want to just dereference the pointer because that makes
// the copylock lint angry.
func (p *ErrorPool) deref() ErrorPool {
	return ErrorPool{
		pool:           p.pool.deref(),
		onlyFirstError: p.onlyFirstError,
	}
}

func (p *ErrorPool) panicIfInitialized() {
	p.pool.panicIfInitialized()
}

func (p *ErrorPool) addErr(err error) {
	if err != nil {
		p.mu.Lock()
		p.errs = append(p.errs, err)
		p.mu.Unlock()
	}
}
