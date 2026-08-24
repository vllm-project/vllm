package pool

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
)

const (
	stateDefault = 0
	stateInited  = 1
	stateClosed  = 2
)

type BadConnError struct {
	wrapped error
}

var _ error = (*BadConnError)(nil)

func (e BadConnError) Error() string {
	s := "redis: Conn is in a bad state"
	if e.wrapped != nil {
		s += ": " + e.wrapped.Error()
	}
	return s
}

func (e BadConnError) Unwrap() error {
	return e.wrapped
}

//------------------------------------------------------------------------------

type StickyConnPool struct {
	pool   Pooler
	shared atomic.Int32

	state atomic.Uint32
	ch    chan *Conn

	// onFirstConn runs once when this sticky pool claims a connection from its
	// parent. CSC uses it to revoke cache ownership before the connection leaves
	// the parent's background drainer.
	onFirstConn func(*Conn)

	_badConnError atomic.Value
}

var _ Pooler = (*StickyConnPool)(nil)

func NewStickyConnPool(pool Pooler) *StickyConnPool {
	p, ok := pool.(*StickyConnPool)
	if !ok {
		p = &StickyConnPool{
			pool: pool,
			ch:   make(chan *Conn, 1),
		}
	}
	p.shared.Add(1)
	return p
}

func (p *StickyConnPool) NewConn(ctx context.Context) (*Conn, error) {
	return p.pool.NewConn(ctx)
}

func (p *StickyConnPool) CloseConn(ctx context.Context, cn *Conn, reason string, fromState string) error {
	return p.pool.CloseConn(ctx, cn, reason, fromState)
}

func (p *StickyConnPool) Get(ctx context.Context) (*Conn, error) {
	// In worst case this races with Close which is not a very common operation.
	for i := 0; i < 1000; i++ {
		switch p.state.Load() {
		case stateDefault:
			cn, err := p.pool.Get(ctx)
			if err != nil {
				return nil, err
			}
			if p.state.CompareAndSwap(stateDefault, stateInited) {
				if p.onFirstConn != nil {
					p.onFirstConn(cn)
				}
				return cn, nil
			}
			p.pool.Remove(ctx, cn, ErrClosed)
		case stateInited:
			if err := p.badConnError(); err != nil {
				return nil, err
			}
			cn, ok := <-p.ch
			if !ok {
				return nil, ErrClosed
			}
			return cn, nil
		case stateClosed:
			return nil, ErrClosed
		default:
			panic("not reached")
		}
	}
	return nil, fmt.Errorf("redis: StickyConnPool.Get: infinite loop")
}

// SetOnFirstConn configures a callback that runs when the sticky pool first
// claims a parent connection. It must be called before the pool is used.
func (p *StickyConnPool) SetOnFirstConn(fn func(*Conn)) {
	p.onFirstConn = fn
}

func (p *StickyConnPool) Put(ctx context.Context, cn *Conn) {
	defer func() {
		if recover() != nil {
			p.freeConn(ctx, cn)
		}
	}()
	// A connection marked for removal on release (it may hold unread
	// replies) must not be served to the next Get: record it as a bad
	// connection — exactly like Remove — so Get refuses and the underlying
	// connection is removed from the parent pool when the sticky pool
	// unwinds (the parent's Put honors the same mark).
	if reason := cn.CloseOnPutReason(); reason != "" {
		p._badConnError.Store(BadConnError{wrapped: errors.New(reason)})
	}
	p.ch <- cn
}

func (p *StickyConnPool) freeConn(ctx context.Context, cn *Conn) {
	if err := p.badConnError(); err != nil {
		p.pool.Remove(ctx, cn, err)
	} else {
		p.pool.Put(ctx, cn)
	}
}

func (p *StickyConnPool) Remove(ctx context.Context, cn *Conn, reason error) {
	defer func() {
		if recover() != nil {
			p.pool.Remove(ctx, cn, ErrClosed)
		}
	}()
	p._badConnError.Store(BadConnError{wrapped: reason})
	p.ch <- cn
}

// RemoveWithoutTurn has the same behavior as Remove for StickyConnPool
// since StickyConnPool doesn't use a turn-based queue system.
func (p *StickyConnPool) RemoveWithoutTurn(ctx context.Context, cn *Conn, reason error) {
	p.Remove(ctx, cn, reason)
}

func (p *StickyConnPool) Close() error {
	if shared := p.shared.Add(-1); shared > 0 {
		return nil
	}

	for i := 0; i < 1000; i++ {
		state := p.state.Load()
		if state == stateClosed {
			return ErrClosed
		}
		if p.state.CompareAndSwap(state, stateClosed) {
			close(p.ch)
			cn, ok := <-p.ch
			if ok {
				p.freeConn(context.TODO(), cn)
			}
			return nil
		}
	}

	return errors.New("redis: StickyConnPool.Close: infinite loop")
}

func (p *StickyConnPool) Reset(ctx context.Context) error {
	if p.badConnError() == nil {
		return nil
	}

	select {
	case cn, ok := <-p.ch:
		if !ok {
			return ErrClosed
		}
		p.pool.Remove(ctx, cn, ErrClosed)
		p._badConnError.Store(BadConnError{wrapped: nil})
	default:
		return errors.New("redis: StickyConnPool does not have a Conn")
	}

	if !p.state.CompareAndSwap(stateInited, stateDefault) {
		state := p.state.Load()
		return fmt.Errorf("redis: invalid StickyConnPool state: %d", state)
	}

	return nil
}

func (p *StickyConnPool) badConnError() error {
	if v := p._badConnError.Load(); v != nil {
		if err := v.(BadConnError); err.wrapped != nil {
			return err
		}
	}
	return nil
}

func (p *StickyConnPool) Len() int {
	switch p.state.Load() {
	case stateDefault:
		return 0
	case stateInited:
		return 1
	case stateClosed:
		return 0
	default:
		panic("not reached")
	}
}

func (p *StickyConnPool) IdleLen() int {
	return len(p.ch)
}

// Size returns the maximum pool size, which is always 1 for StickyConnPool.
func (p *StickyConnPool) Size() int { return 1 }

func (p *StickyConnPool) Stats() *Stats {
	return &Stats{}
}

func (p *StickyConnPool) AddPoolHook(hook PoolHook) {}

func (p *StickyConnPool) RemovePoolHook(hook PoolHook) {}
