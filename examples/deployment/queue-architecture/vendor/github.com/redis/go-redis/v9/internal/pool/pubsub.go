package pool

import (
	"context"
	"net"
	"sync"
	"sync/atomic"
)

// PubSubStats contains pub/sub connection pool stats.
//
// TODO(cxl): the uint32 fields below will be changed to atomic.Uint32 in v10,
// which is a breaking API change.
type PubSubStats struct {
	Created   uint32
	Untracked uint32
	Active    uint32
}

// PubSubPool manages a pool of PubSub connections.
type PubSubPool struct {
	opt       *Options
	netDialer func(ctx context.Context, network, addr string) (net.Conn, error)

	// Map to track active PubSub connections
	activeConns sync.Map // map[uint64]*Conn (connID -> conn)
	closed      atomic.Bool
	stats       PubSubStats
}

// NewPubSubPool implements a pool for PubSub connections.
// It intentionally does not implement the Pooler interface
func NewPubSubPool(opt *Options, netDialer func(ctx context.Context, network, addr string) (net.Conn, error)) *PubSubPool {
	return &PubSubPool{
		opt:       opt,
		netDialer: netDialer,
	}
}

func (p *PubSubPool) NewConn(ctx context.Context, network string, addr string, channels []string) (*Conn, error) {
	if p.closed.Load() {
		return nil, ErrClosed
	}

	netConn, err := p.netDialer(ctx, network, addr)
	if err != nil {
		return nil, err
	}
	cn := NewConnWithBufferSize(netConn, p.opt.ReadBufferSize, p.opt.WriteBufferSize)
	cn.pubsub = true
	// Set pool name for metrics
	cn.SetPoolName(p.opt.Name)
	atomic.AddUint32(&p.stats.Created, 1)
	return cn, nil
}

func (p *PubSubPool) TrackConn(cn *Conn) {
	atomic.AddUint32(&p.stats.Active, 1)
	p.activeConns.Store(cn.GetID(), cn)
	// Emit +1 used for PubSub connection
	if cb := getMetricConnectionCountCallback(); cb != nil {
		cb(context.Background(), 1, cn, "used", true)
	}
}

func (p *PubSubPool) UntrackConn(cn *Conn) {
	// LoadAndDelete ensures each connection is only decremented once,
	// guarding against double-decrement if Close() already untracked it.
	if _, loaded := p.activeConns.LoadAndDelete(cn.GetID()); !loaded {
		return
	}
	atomic.AddUint32(&p.stats.Active, ^uint32(0))
	atomic.AddUint32(&p.stats.Untracked, 1)
	// Emit -1 used for PubSub connection
	if cb := getMetricConnectionCountCallback(); cb != nil {
		cb(context.Background(), -1, cn, "used", true)
	}
}

func (p *PubSubPool) Close() error {
	p.closed.Store(true)
	cb := getMetricConnectionCountCallback()
	p.activeConns.Range(func(key, value interface{}) bool {
		cn := value.(*Conn)
		// Use LoadAndDelete to atomically claim ownership of this entry.
		// If a concurrent UntrackConn already removed it, skip to avoid double-decrement.
		if _, loaded := p.activeConns.LoadAndDelete(key); !loaded {
			return true
		}
		atomic.AddUint32(&p.stats.Active, ^uint32(0))
		atomic.AddUint32(&p.stats.Untracked, 1)
		// Emit -1 used for each PubSub connection being closed
		if cb != nil {
			cb(context.Background(), -1, cn, "used", true)
		}
		_ = cn.Close()
		return true
	})
	return nil
}

func (p *PubSubPool) Stats() *PubSubStats {
	// load stats atomically
	return &PubSubStats{
		Created:   atomic.LoadUint32(&p.stats.Created),
		Untracked: atomic.LoadUint32(&p.stats.Untracked),
		Active:    atomic.LoadUint32(&p.stats.Active),
	}
}
