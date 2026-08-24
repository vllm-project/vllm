package pool

import (
	"context"
	"time"
)

// SingleConnPool is a pool that always returns the same connection.
// Note: This pool is not thread-safe.
// It is intended to be used by clients that need a single connection.
type SingleConnPool struct {
	pool      Pooler
	cn        *Conn
	stickyErr error
}

var _ Pooler = (*SingleConnPool)(nil)

// NewSingleConnPool creates a new single connection pool.
// The pool will always return the same connection.
// The pool will not:
// - Close the connection
// - Reconnect the connection
// - Track the connection in any way
func NewSingleConnPool(pool Pooler, cn *Conn) *SingleConnPool {
	return &SingleConnPool{
		pool: pool,
		cn:   cn,
	}
}

func (p *SingleConnPool) NewConn(ctx context.Context) (*Conn, error) {
	return p.pool.NewConn(ctx)
}

func (p *SingleConnPool) CloseConn(ctx context.Context, cn *Conn, reason string, fromState string) error {
	return p.pool.CloseConn(ctx, cn, reason, fromState)
}

func (p *SingleConnPool) Get(_ context.Context) (*Conn, error) {
	if p.stickyErr != nil {
		return nil, p.stickyErr
	}
	if p.cn == nil {
		return nil, ErrClosed
	}

	// NOTE: SingleConnPool is NOT thread-safe by design and is used in special scenarios:
	// - During initialization (connection is in INITIALIZING state)
	// - During re-authentication (connection is in UNUSABLE state)
	// - For transactions (connection might be in various states)
	// We use SetUsed() which forces the transition, rather than TryTransition() which
	// would fail if the connection is not in IDLE/CREATED state.
	p.cn.SetUsed(true)
	p.cn.SetUsedAt(time.Now())
	return p.cn, nil
}

func (p *SingleConnPool) Put(_ context.Context, cn *Conn) {
	if p.cn == nil {
		return
	}
	if p.cn != cn {
		return
	}
	p.cn.SetUsed(false)
}

func (p *SingleConnPool) Remove(_ context.Context, cn *Conn, reason error) {
	cn.SetUsed(false)
	p.cn = nil
	p.stickyErr = reason
}

// RemoveWithoutTurn has the same behavior as Remove for SingleConnPool
// since SingleConnPool doesn't use a turn-based queue system.
func (p *SingleConnPool) RemoveWithoutTurn(ctx context.Context, cn *Conn, reason error) {
	p.Remove(ctx, cn, reason)
}

func (p *SingleConnPool) Close() error {
	p.cn = nil
	p.stickyErr = ErrClosed
	return nil
}

func (p *SingleConnPool) Len() int {
	return 0
}

func (p *SingleConnPool) IdleLen() int {
	return 0
}

// Size returns the maximum pool size, which is always 1 for SingleConnPool.
func (p *SingleConnPool) Size() int { return 1 }

func (p *SingleConnPool) Stats() *Stats {
	return &Stats{}
}

func (p *SingleConnPool) AddPoolHook(_ PoolHook) {}

func (p *SingleConnPool) RemovePoolHook(_ PoolHook) {}
