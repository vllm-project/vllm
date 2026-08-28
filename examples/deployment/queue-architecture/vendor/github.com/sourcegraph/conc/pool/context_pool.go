package pool

import (
	"context"
)

// ContextPool is a pool that runs tasks that take a context.
// A new ContextPool should be created with `New().WithContext(ctx)`.
//
// The configuration methods (With*) will panic if they are used after calling
// Go() for the first time.
type ContextPool struct {
	errorPool ErrorPool

	ctx    context.Context
	cancel context.CancelFunc

	cancelOnError bool
}

// Go submits a task. If it returns an error, the error will be
// collected and returned by Wait(). If all goroutines in the pool
// are busy, a call to Go() will block until the task can be started.
func (p *ContextPool) Go(f func(ctx context.Context) error) {
	p.errorPool.Go(func() error {
		if p.cancelOnError {
			// If we are cancelling on error, then we also want to cancel if a
			// panic is raised. To do this, we need to recover, cancel, and then
			// re-throw the caught panic.
			defer func() {
				if r := recover(); r != nil {
					p.cancel()
					panic(r)
				}
			}()
		}

		err := f(p.ctx)
		if err != nil && p.cancelOnError {
			// Leaky abstraction warning: We add the error directly because
			// otherwise, canceling could cause another goroutine to exit and
			// return an error before this error was added, which breaks the
			// expectations of WithFirstError().
			p.errorPool.addErr(err)
			p.cancel()
			return nil
		}
		return err
	})
}

// Wait cleans up all spawned goroutines, propagates any panics, and
// returns an error if any of the tasks errored.
func (p *ContextPool) Wait() error {
	// Make sure we call cancel after pool is done to avoid memory leakage.
	defer p.cancel()
	return p.errorPool.Wait()
}

// WithFirstError configures the pool to only return the first error
// returned by a task. By default, Wait() will return a combined error.
// This is particularly useful for (*ContextPool).WithCancelOnError(),
// where all errors after the first are likely to be context.Canceled.
func (p *ContextPool) WithFirstError() *ContextPool {
	p.panicIfInitialized()
	p.errorPool.WithFirstError()
	return p
}

// WithCancelOnError configures the pool to cancel its context as soon as
// any task returns an error or panics. By default, the pool's context is not
// canceled until the parent context is canceled.
//
// In this case, all errors returned from the pool after the first will
// likely be context.Canceled - you may want to also use
// (*ContextPool).WithFirstError() to configure the pool to only return
// the first error.
func (p *ContextPool) WithCancelOnError() *ContextPool {
	p.panicIfInitialized()
	p.cancelOnError = true
	return p
}

// WithFailFast is an alias for the combination of WithFirstError and
// WithCancelOnError. By default, the errors from all tasks are returned and
// the pool's context is not canceled until the parent context is canceled.
func (p *ContextPool) WithFailFast() *ContextPool {
	p.panicIfInitialized()
	p.WithFirstError()
	p.WithCancelOnError()
	return p
}

// WithMaxGoroutines limits the number of goroutines in a pool.
// Defaults to unlimited. Panics if n < 1.
func (p *ContextPool) WithMaxGoroutines(n int) *ContextPool {
	p.panicIfInitialized()
	p.errorPool.WithMaxGoroutines(n)
	return p
}

func (p *ContextPool) panicIfInitialized() {
	p.errorPool.panicIfInitialized()
}
