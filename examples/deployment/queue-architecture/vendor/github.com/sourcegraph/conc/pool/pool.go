package pool

import (
	"context"
	"sync"

	"github.com/sourcegraph/conc"
)

// New creates a new Pool.
func New() *Pool {
	return &Pool{}
}

// Pool is a pool of goroutines used to execute tasks concurrently.
//
// Tasks are submitted with Go(). Once all your tasks have been submitted, you
// must call Wait() to clean up any spawned goroutines and propagate any
// panics.
//
// Goroutines are started lazily, so creating a new pool is cheap. There will
// never be more goroutines spawned than there are tasks submitted.
//
// The configuration methods (With*) will panic if they are used after calling
// Go() for the first time.
//
// Pool is efficient, but not zero cost. It should not be used for very short
// tasks. Startup and teardown come with an overhead of around 1µs, and each
// task has an overhead of around 300ns.
type Pool struct {
	handle   conc.WaitGroup
	limiter  limiter
	tasks    chan func()
	initOnce sync.Once
}

// Go submits a task to be run in the pool. If all goroutines in the pool
// are busy, a call to Go() will block until the task can be started.
func (p *Pool) Go(f func()) {
	p.init()

	if p.limiter == nil {
		// No limit on the number of goroutines.
		select {
		case p.tasks <- f:
			// A goroutine was available to handle the task.
		default:
			// No goroutine was available to handle the task.
			// Spawn a new one and send it the task.
			p.handle.Go(func() {
				p.worker(f)
			})
		}
	} else {
		select {
		case p.limiter <- struct{}{}:
			// If we are below our limit, spawn a new worker rather
			// than waiting for one to become available.
			p.handle.Go(func() {
				p.worker(f)
			})
		case p.tasks <- f:
			// A worker is available and has accepted the task.
			return
		}
	}

}

// Wait cleans up spawned goroutines, propagating any panics that were
// raised by a tasks.
func (p *Pool) Wait() {
	p.init()

	close(p.tasks)

	// After Wait() returns, reset the struct so tasks will be reinitialized on
	// next use. This better matches the behavior of sync.WaitGroup
	defer func() { p.initOnce = sync.Once{} }()

	p.handle.Wait()
}

// MaxGoroutines returns the maximum size of the pool.
func (p *Pool) MaxGoroutines() int {
	return p.limiter.limit()
}

// WithMaxGoroutines limits the number of goroutines in a pool.
// Defaults to unlimited. Panics if n < 1.
func (p *Pool) WithMaxGoroutines(n int) *Pool {
	p.panicIfInitialized()
	if n < 1 {
		panic("max goroutines in a pool must be greater than zero")
	}
	p.limiter = make(limiter, n)
	return p
}

// init ensures that the pool is initialized before use. This makes the
// zero value of the pool usable.
func (p *Pool) init() {
	p.initOnce.Do(func() {
		p.tasks = make(chan func())
	})
}

// panicIfInitialized will trigger a panic if a configuration method is called
// after the pool has started any goroutines for the first time. In the case that
// new settings are needed, a new pool should be created.
func (p *Pool) panicIfInitialized() {
	if p.tasks != nil {
		panic("pool can not be reconfigured after calling Go() for the first time")
	}
}

// WithErrors converts the pool to an ErrorPool so the submitted tasks can
// return errors.
func (p *Pool) WithErrors() *ErrorPool {
	p.panicIfInitialized()
	return &ErrorPool{
		pool: p.deref(),
	}
}

// deref is a helper that creates a shallow copy of the pool with the same
// settings. We don't want to just dereference the pointer because that makes
// the copylock lint angry.
func (p *Pool) deref() Pool {
	p.panicIfInitialized()
	return Pool{
		limiter: p.limiter,
	}
}

// WithContext converts the pool to a ContextPool for tasks that should
// run under the same context, such that they each respect shared cancellation.
// For example, WithCancelOnError can be configured on the returned pool to
// signal that all goroutines should be cancelled upon the first error.
func (p *Pool) WithContext(ctx context.Context) *ContextPool {
	p.panicIfInitialized()
	ctx, cancel := context.WithCancel(ctx)
	return &ContextPool{
		errorPool: p.WithErrors().deref(),
		ctx:       ctx,
		cancel:    cancel,
	}
}

func (p *Pool) worker(initialFunc func()) {
	// The only time this matters is if the task panics.
	// This makes it possible to spin up new workers in that case.
	defer p.limiter.release()

	if initialFunc != nil {
		initialFunc()
	}

	for f := range p.tasks {
		f()
	}
}

type limiter chan struct{}

func (l limiter) limit() int {
	return cap(l)
}

func (l limiter) release() {
	if l != nil {
		<-l
	}
}
