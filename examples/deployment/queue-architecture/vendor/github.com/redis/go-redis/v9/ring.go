package redis

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/auth"
	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/hashtag"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/internal/proto"
)

var errRingShardsDown = errors.New("redis: all ring shards are down")

// defaultHeartbeatFn is the default function used to check the shard liveness
var defaultHeartbeatFn = func(ctx context.Context, client *Client) bool {
	err := client.Ping(ctx).Err()
	return err == nil || err == pool.ErrPoolTimeout
}

//------------------------------------------------------------------------------

type ConsistentHash interface {
	Get(string) string
}

func newRendezvous(shards []string) ConsistentHash {
	return hashtag.NewRendezvousHash(shards)
}

//------------------------------------------------------------------------------

// RingOptions are used to configure a ring client and should be
// passed to NewRing.
type RingOptions struct {
	// Map of name => host:port addresses of ring shards.
	Addrs map[string]string

	// NewClient creates a shard client with provided options.
	NewClient func(opt *Options) *Client

	// himport is the ring-wide HIMPORT fieldset registry, set by NewRing and
	// shared with every shard client (see himport.go, himport_cluster.go).
	himport *himportRegistry

	// ClientName will execute the `CLIENT SETNAME ClientName` command for each conn.
	ClientName string

	// Frequency of executing HeartbeatFn to check shards availability.
	// Shard is considered down after 3 subsequent failed checks.
	HeartbeatFrequency time.Duration

	// A function used to check the shard liveness
	// if not set, defaults to defaultHeartbeatFn
	HeartbeatFn func(ctx context.Context, client *Client) bool

	// NewConsistentHash returns a consistent hash that is used
	// to distribute keys across the shards.
	//
	// See https://medium.com/@dgryski/consistent-hashing-algorithmic-tradeoffs-ef6b8e2fcae8
	// for consistent hashing algorithmic tradeoffs.
	NewConsistentHash func(shards []string) ConsistentHash

	// Following options are copied from Options struct.

	Dialer    func(ctx context.Context, network, addr string) (net.Conn, error)
	OnConnect func(ctx context.Context, cn *Conn) error

	Protocol int
	Username string
	Password string
	// CredentialsProvider allows the username and password to be updated
	// before reconnecting. It should return the current username and password.
	CredentialsProvider func() (username string, password string)

	// CredentialsProviderContext is an enhanced parameter of CredentialsProvider,
	// done to maintain API compatibility. In the future,
	// there might be a merge between CredentialsProviderContext and CredentialsProvider.
	// There will be a conflict between them; if CredentialsProviderContext exists, we will ignore CredentialsProvider.
	CredentialsProviderContext func(ctx context.Context) (username string, password string, err error)

	// StreamingCredentialsProvider is used to retrieve the credentials
	// for the connection from an external source. Those credentials may change
	// during the connection lifetime. This is useful for managed identity
	// scenarios where the credentials are retrieved from an external source.
	//
	// Currently, this is a placeholder for the future implementation.
	StreamingCredentialsProvider auth.StreamingCredentialsProvider
	DB                           int

	MaxRetries      int
	MinRetryBackoff time.Duration
	MaxRetryBackoff time.Duration

	DialTimeout time.Duration

	// DialerRetries is the maximum number of retry attempts when dialing fails.
	//
	// default: 5
	DialerRetries int

	// DialerRetryTimeout is the backoff duration between retry attempts.
	//
	// default: 100 milliseconds
	DialerRetryTimeout time.Duration

	// DialerRetryBackoff controls the delay between dial retry attempts.
	// See Options.DialerRetryBackoff for details.
	DialerRetryBackoff func(attempt int) time.Duration

	ReadTimeout           time.Duration
	WriteTimeout          time.Duration
	ContextTimeoutEnabled bool

	// PoolFIFO uses FIFO mode for each node connection pool GET/PUT (default LIFO).
	PoolFIFO bool

	PoolSize              int
	PoolTimeout           time.Duration
	MinIdleConns          int
	MaxIdleConns          int
	MaxActiveConns        int
	ConnMaxIdleTime       time.Duration
	ConnMaxLifetime       time.Duration
	ConnMaxLifetimeJitter time.Duration

	// ReadBufferSize is the size of the bufio.Reader buffer for each connection.
	// Larger buffers can improve performance for commands that return large responses.
	// Smaller buffers can improve memory usage for larger pools.
	//
	// default: 32KiB (32768 bytes)
	ReadBufferSize int

	// WriteBufferSize is the size of the bufio.Writer buffer for each connection.
	// Larger buffers can improve performance for large pipelines and commands with many arguments.
	// Smaller buffers can improve memory usage for larger pools.
	//
	// default: 32KiB (32768 bytes)
	WriteBufferSize int

	// PipelineReadBufferSize, PipelineWriteBufferSize and PipelinePoolSize
	// configure an optional separate connection pool used for pipelining on
	// each shard, with its own (typically larger) buffers. See the same-named
	// fields on Options for details. The pool is created only when PipelineReadBufferSize or PipelineWriteBufferSize is set (PipelinePoolSize alone does not enable it).
	PipelineReadBufferSize  int
	PipelineWriteBufferSize int
	PipelinePoolSize        int

	TLSConfig *tls.Config
	Limiter   Limiter

	// DisableIndentity - Disable set-lib on connect.
	//
	// default: false
	//
	// Deprecated: Use DisableIdentity instead.
	DisableIndentity bool

	// DisableIdentity is used to disable CLIENT SETINFO command on connect.
	//
	// default: false
	DisableIdentity bool
	IdentitySuffix  string

	// Deprecated: All RediSearch commands now have stable RESP3 parsing and this
	// flag is a no-op. It is kept for backwards compatibility and will be removed
	// in a future release.
	UnstableResp3 bool
}

func (opt *RingOptions) init() {
	if opt.NewClient == nil {
		opt.NewClient = func(opt *Options) *Client {
			return NewClient(opt)
		}
	}

	if opt.HeartbeatFrequency == 0 {
		opt.HeartbeatFrequency = 500 * time.Millisecond
	}

	if opt.HeartbeatFn == nil {
		opt.HeartbeatFn = defaultHeartbeatFn
	}

	if opt.NewConsistentHash == nil {
		opt.NewConsistentHash = newRendezvous
	}

	switch opt.MaxRetries {
	case -1:
		opt.MaxRetries = 0
	case 0:
		opt.MaxRetries = 3
	}
	switch opt.MinRetryBackoff {
	case -1:
		opt.MinRetryBackoff = 0
	case 0:
		opt.MinRetryBackoff = 10 * time.Millisecond
	}
	switch opt.MaxRetryBackoff {
	case -1:
		opt.MaxRetryBackoff = 0
	case 0:
		opt.MaxRetryBackoff = time.Second
	}

	if opt.ReadBufferSize == 0 {
		opt.ReadBufferSize = proto.DefaultBufferSize
	}
	if opt.WriteBufferSize == 0 {
		opt.WriteBufferSize = proto.DefaultBufferSize
	}
}

func (opt *RingOptions) clientOptions() *Options {
	return &Options{
		ClientName: opt.ClientName,
		Dialer:     opt.Dialer,
		OnConnect:  opt.OnConnect,

		Protocol:                     opt.Protocol,
		Username:                     opt.Username,
		Password:                     opt.Password,
		CredentialsProvider:          opt.CredentialsProvider,
		CredentialsProviderContext:   opt.CredentialsProviderContext,
		StreamingCredentialsProvider: opt.StreamingCredentialsProvider,
		DB:                           opt.DB,

		MaxRetries: -1,

		DialTimeout:           opt.DialTimeout,
		DialerRetries:         opt.DialerRetries,
		DialerRetryTimeout:    opt.DialerRetryTimeout,
		DialerRetryBackoff:    opt.DialerRetryBackoff,
		ReadTimeout:           opt.ReadTimeout,
		WriteTimeout:          opt.WriteTimeout,
		ContextTimeoutEnabled: opt.ContextTimeoutEnabled,

		PoolFIFO:              opt.PoolFIFO,
		PoolSize:              opt.PoolSize,
		PoolTimeout:           opt.PoolTimeout,
		MinIdleConns:          opt.MinIdleConns,
		MaxIdleConns:          opt.MaxIdleConns,
		MaxActiveConns:        opt.MaxActiveConns,
		ConnMaxIdleTime:       opt.ConnMaxIdleTime,
		ConnMaxLifetime:       opt.ConnMaxLifetime,
		ConnMaxLifetimeJitter: opt.ConnMaxLifetimeJitter,
		ReadBufferSize:        opt.ReadBufferSize,
		WriteBufferSize:       opt.WriteBufferSize,

		PipelineReadBufferSize:  opt.PipelineReadBufferSize,
		PipelineWriteBufferSize: opt.PipelineWriteBufferSize,
		PipelinePoolSize:        opt.PipelinePoolSize,

		TLSConfig: opt.TLSConfig,
		Limiter:   opt.Limiter,

		DisableIdentity:  opt.DisableIdentity,
		DisableIndentity: opt.DisableIndentity,

		IdentitySuffix: opt.IdentitySuffix,
		UnstableResp3:  opt.UnstableResp3,
	}
}

//------------------------------------------------------------------------------

type ringShard struct {
	Client *Client
	down   atomic.Int32
	addr   string
}

func newRingShard(opt *RingOptions, addr string) *ringShard {
	clopt := opt.clientOptions()
	clopt.Addr = addr

	shard := &ringShard{
		Client: opt.NewClient(clopt),
		addr:   addr,
	}
	// Share the ring-wide HIMPORT fieldset registry so any shard connection
	// serving an HIMPORT SET can lazily replay the PREPARE.
	if opt.himport != nil {
		shard.Client.himport = opt.himport
	}
	return shard
}

func (shard *ringShard) String() string {
	var state string
	if shard.IsUp() {
		state = "up"
	} else {
		state = "down"
	}
	return fmt.Sprintf("%s is %s", shard.Client, state)
}

func (shard *ringShard) IsDown() bool {
	const threshold = 3
	return shard.down.Load() >= threshold
}

func (shard *ringShard) IsUp() bool {
	return !shard.IsDown()
}

// Vote votes to set shard state and returns true if state was changed.
func (shard *ringShard) Vote(up bool) bool {
	if up {
		changed := shard.IsDown()
		shard.down.Store(0)
		return changed
	}

	if shard.IsDown() {
		return false
	}

	shard.down.Add(1)
	return shard.IsDown()
}

//------------------------------------------------------------------------------

type ringSharding struct {
	opt *RingOptions

	mu        sync.RWMutex
	shards    *ringShards
	closed    bool
	hash      ConsistentHash
	numShard  int
	onNewNode []func(rdb *Client)

	// ensures exclusive access to SetAddrs so there is no need
	// to hold mu for the duration of potentially long shard creation
	setAddrsMu sync.Mutex
}

type ringShards struct {
	m    map[string]*ringShard
	list []*ringShard
}

func newRingSharding(opt *RingOptions) *ringSharding {
	c := &ringSharding{
		opt: opt,
	}
	c.SetAddrs(opt.Addrs)

	return c
}

func (c *ringSharding) OnNewNode(fn func(rdb *Client)) {
	c.mu.Lock()
	c.onNewNode = append(c.onNewNode, fn)
	c.mu.Unlock()
}

// SetAddrs replaces the shards in use, such that you can increase and
// decrease number of shards, that you use. It will reuse shards that
// existed before and close the ones that will not be used anymore.
func (c *ringSharding) SetAddrs(addrs map[string]string) {
	c.setAddrsMu.Lock()
	defer c.setAddrsMu.Unlock()

	cleanup := func(shards map[string]*ringShard) {
		for addr, shard := range shards {
			if err := shard.Client.Close(); err != nil {
				internal.Logger.Printf(context.Background(), "shard.Close %s failed: %s", addr, err)
			}
		}
	}

	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return
	}
	existing := c.shards
	onNewNode := c.onNewNode
	c.mu.RUnlock()

	shards, created, unused := c.newRingShards(addrs, existing, onNewNode)

	c.mu.Lock()
	if c.closed {
		cleanup(created)
		c.mu.Unlock()
		return
	}
	c.shards = shards
	c.rebalanceLocked()
	c.mu.Unlock()

	cleanup(unused)
}

func (c *ringSharding) newRingShards(
	addrs map[string]string, existing *ringShards, onNewNode []func(rdb *Client),
) (shards *ringShards, created, unused map[string]*ringShard) {
	shards = &ringShards{m: make(map[string]*ringShard, len(addrs))}
	created = make(map[string]*ringShard) // indexed by addr
	unused = make(map[string]*ringShard)  // indexed by addr

	if existing != nil {
		for _, shard := range existing.list {
			unused[shard.addr] = shard
		}
	}

	for name, addr := range addrs {
		if shard, ok := unused[addr]; ok {
			shards.m[name] = shard
			delete(unused, addr)
		} else {
			shard := newRingShard(c.opt, addr)
			shards.m[name] = shard
			created[addr] = shard

			for _, fn := range onNewNode {
				fn(shard.Client)
			}
		}
	}

	for _, shard := range shards.m {
		shards.list = append(shards.list, shard)
	}

	return
}

// Warning: External exposure of `c.shards.list` may cause data races.
// So keep internal or implement deep copy if exposed.
func (c *ringSharding) List() []*ringShard {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil
	}
	return c.shards.list
}

func (c *ringSharding) Hash(key string) string {
	key = hashtag.Key(key)

	var hash string

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.numShard > 0 {
		hash = c.hash.Get(key)
	}

	return hash
}

func (c *ringSharding) GetByKey(key string) (*ringShard, error) {
	key = hashtag.Key(key)

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, pool.ErrClosed
	}

	if c.numShard == 0 {
		return nil, errRingShardsDown
	}

	shardName := c.hash.Get(key)
	if shardName == "" {
		return nil, errRingShardsDown
	}
	return c.shards.m[shardName], nil
}

func (c *ringSharding) GetByName(shardName string) (*ringShard, error) {
	if shardName == "" {
		return c.Random()
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	shard, ok := c.shards.m[shardName]
	if !ok {
		return nil, errors.New("redis: the shard is not in the ring")
	}

	return shard, nil
}

func (c *ringSharding) Random() (*ringShard, error) {
	return c.GetByKey(strconv.Itoa(rand.Int()))
}

// Heartbeat monitors state of each shard in the ring.
func (c *ringSharding) Heartbeat(ctx context.Context, frequency time.Duration) {
	ticker := time.NewTicker(frequency)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			var rebalance bool

			// note: `c.List()` return a shadow copy of `[]*ringShard`.
			for _, shard := range c.List() {
				isUp := c.opt.HeartbeatFn(ctx, shard.Client)
				if shard.Vote(isUp) {
					internal.Logger.Printf(ctx, "ring shard state changed: %s", shard)
					rebalance = true
				}
			}

			if rebalance {
				c.mu.Lock()
				c.rebalanceLocked()
				c.mu.Unlock()
			}
		case <-ctx.Done():
			return
		}
	}
}

// rebalanceLocked removes dead shards from the Ring.
// Requires c.mu locked.
func (c *ringSharding) rebalanceLocked() {
	if c.closed {
		return
	}
	if c.shards == nil {
		return
	}

	liveShards := make([]string, 0, len(c.shards.m))

	for name, shard := range c.shards.m {
		if shard.IsUp() {
			liveShards = append(liveShards, name)
		}
	}

	c.hash = c.opt.NewConsistentHash(liveShards)
	c.numShard = len(liveShards)
}

func (c *ringSharding) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.numShard
}

func (c *ringSharding) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true

	var firstErr error

	for _, shard := range c.shards.list {
		if err := shard.Client.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	c.hash = nil
	c.shards = nil
	c.numShard = 0

	return firstErr
}

//------------------------------------------------------------------------------

// Ring is a Redis client that uses consistent hashing to distribute
// keys across multiple Redis servers (shards). It's safe for
// concurrent use by multiple goroutines.
//
// Ring monitors the state of each shard and removes dead shards from
// the ring. When a shard comes online it is added back to the ring. This
// gives you maximum availability and partition tolerance, but no
// consistency between different shards or even clients. Each client
// uses shards that are available to the client and does not do any
// coordination when shard state is changed.
//
// Ring should be used when you need multiple Redis servers for caching
// and can tolerate losing data when one of the servers dies.
// Otherwise you should use Redis Cluster.
type Ring struct {
	cmdable
	hooksMixin

	opt               *RingOptions
	sharding          *ringSharding
	cmdsInfoCache     *cmdsInfoCache
	heartbeatCancelFn context.CancelFunc
}

// NewRing returns a Redis Ring client to the Redis Server specified by RingOptions.
// Passing nil RingOptions will cause a panic.
func NewRing(opt *RingOptions) *Ring {
	if opt == nil {
		panic("redis: NewRing nil options")
	}
	// Shallow-copy the options: the ring-wide HIMPORT registry is carried
	// through them to shard construction, and reusing one caller-owned
	// RingOptions across several rings must not make the rings share (or
	// clobber each other's) registry.
	optCopy := *opt
	opt = &optCopy
	opt.init()
	// The registry must exist before the first shard is created; shards
	// adopt it in newRingShard.
	opt.himport = newHImportRegistry()

	hbCtx, hbCancel := context.WithCancel(context.Background())

	ring := Ring{
		opt:               opt,
		sharding:          newRingSharding(opt),
		heartbeatCancelFn: hbCancel,
	}

	ring.cmdsInfoCache = newCmdsInfoCache(ring.cmdsInfo)
	ring.cmdable = ring.Process

	ring.initHooks(hooks{
		process: ring.process,
		pipeline: func(ctx context.Context, cmds []Cmder) error {
			return ring.generalProcessPipeline(ctx, cmds, false)
		},
		txPipeline: func(ctx context.Context, cmds []Cmder) error {
			return ring.generalProcessPipeline(ctx, cmds, true)
		},
	})

	go ring.sharding.Heartbeat(hbCtx, opt.HeartbeatFrequency)

	return &ring
}

func (c *Ring) SetAddrs(addrs map[string]string) {
	c.sharding.SetAddrs(addrs)
}

func (c *Ring) Process(ctx context.Context, cmd Cmder) error {
	err := c.processHook(ctx, cmd)
	cmd.SetErr(err)
	return err
}

// Options returns read-only *RingOptions that were used to create the client.
// Any alteration of the returned *RingOptions may result in undefined behaviour.
func (c *Ring) Options() *RingOptions {
	return c.opt
}

func (c *Ring) retryBackoff(attempt int) time.Duration {
	return internal.RetryBackoff(attempt, c.opt.MinRetryBackoff, c.opt.MaxRetryBackoff)
}

// PoolStats returns accumulated connection pool stats.
func (c *Ring) PoolStats() *PoolStats {
	// note: `c.List()` return a shadow copy of `[]*ringShard`.
	shards := c.sharding.List()
	var acc PoolStats
	for _, shard := range shards {
		s := shard.Client.connPool.Stats()
		acc.Hits += s.Hits
		acc.Misses += s.Misses
		acc.Timeouts += s.Timeouts
		acc.TotalConns += s.TotalConns
		acc.IdleConns += s.IdleConns
	}
	return &acc
}

// Len returns the current number of shards in the ring.
func (c *Ring) Len() int {
	return c.sharding.Len()
}

// Subscribe subscribes the client to the specified channels.
func (c *Ring) Subscribe(ctx context.Context, channels ...string) *PubSub {
	if len(channels) == 0 {
		panic("at least one channel is required")
	}

	shard, err := c.sharding.GetByKey(channels[0])
	if err != nil {
		// TODO: return PubSub with sticky error
		panic(err)
	}
	return shard.Client.Subscribe(ctx, channels...)
}

// PSubscribe subscribes the client to the given patterns.
func (c *Ring) PSubscribe(ctx context.Context, channels ...string) *PubSub {
	if len(channels) == 0 {
		panic("at least one channel is required")
	}

	shard, err := c.sharding.GetByKey(channels[0])
	if err != nil {
		// TODO: return PubSub with sticky error
		panic(err)
	}
	return shard.Client.PSubscribe(ctx, channels...)
}

// SSubscribe Subscribes the client to the specified shard channels.
func (c *Ring) SSubscribe(ctx context.Context, channels ...string) *PubSub {
	if len(channels) == 0 {
		panic("at least one channel is required")
	}
	shard, err := c.sharding.GetByKey(channels[0])
	if err != nil {
		// TODO: return PubSub with sticky error
		panic(err)
	}
	return shard.Client.SSubscribe(ctx, channels...)
}

// Publish posts the message to the channel
func (c *Ring) Publish(ctx context.Context, channel string, message interface{}) *IntCmd {
	shard, err := c.sharding.GetByKey(channel)
	if err != nil {
		cmd := NewIntCmd(ctx, "publish", channel, message)
		cmd.SetErr(err)
		return cmd
	}
	return shard.Client.Publish(ctx, channel, message)
}

func (c *Ring) OnNewNode(fn func(rdb *Client)) {
	c.sharding.OnNewNode(fn)
}

// ForEachShard concurrently calls the fn on each live shard in the ring.
// It returns the first error if any.
func (c *Ring) ForEachShard(
	ctx context.Context,
	fn func(ctx context.Context, client *Client) error,
) error {
	// note: `c.List()` return a shadow copy of `[]*ringShard`.
	shards := c.sharding.List()
	var wg sync.WaitGroup
	errCh := make(chan error, 1)
	for _, shard := range shards {
		if shard.IsDown() {
			continue
		}

		wg.Add(1)
		go func(shard *ringShard) {
			defer wg.Done()
			err := fn(ctx, shard.Client)
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
			}
		}(shard)
	}
	wg.Wait()

	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

func (c *Ring) cmdsInfo(ctx context.Context) (map[string]*CommandInfo, error) {
	// note: `c.List()` return a shadow copy of `[]*ringShard`.
	shards := c.sharding.List()
	var firstErr error
	for _, shard := range shards {
		cmdsInfo, err := shard.Client.Command(ctx).Result()
		if err == nil {
			return cmdsInfo, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	if firstErr == nil {
		return nil, errRingShardsDown
	}
	return nil, firstErr
}

func (c *Ring) cmdShard(cmd Cmder) (*ringShard, error) {
	// TODO: populate cmdsInfoCache lazily (via cmdsInfoCache.Get) so that
	// the warm-cache branch in cmdFirstKeyPosWithInfo is reachable for Ring,
	// mirroring how ClusterClient.cmdInfo works. For now pass nil
	pos := cmdFirstKeyPosWithInfo(cmd, nil)
	if pos == 0 {
		return c.sharding.Random()
	}
	firstKey := cmd.stringArg(pos)
	return c.sharding.GetByKey(firstKey)
}

func (c *Ring) process(ctx context.Context, cmd Cmder) error {
	var lastErr error
	for attempt := 0; attempt <= c.opt.MaxRetries; attempt++ {
		if attempt > 0 {
			if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
				return err
			}
		}

		shard, err := c.cmdShard(cmd)
		if err != nil {
			return err
		}

		lastErr = shard.Client.Process(ctx, cmd)
		if lastErr == nil || !shouldRetry(lastErr, cmd.readTimeout() == nil) || cmd.NoRetry() {
			return lastErr
		}
	}
	return lastErr
}

func (c *Ring) Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.Pipeline().Pipelined(ctx, fn)
}

func (c *Ring) Pipeline() Pipeliner {
	pipe := Pipeline{
		exec: pipelineExecer(c.processPipelineHook),
	}
	pipe.init()
	return &pipe
}

// ErrRingAutoPipelineUnsupported is returned by Ring's AutoPipeline /
// AsyncAutoPipeline (and their WithOptions forms); check for it with
// errors.Is. Autopipelining is not implemented for Ring; use the per-shard
// clients or a ClusterClient. Ring is part of the UniversalClient
// interface, so these methods exist to satisfy it and fail explicitly rather
// than being silently absent.
var ErrRingAutoPipelineUnsupported = errors.New("redis: AutoPipeline is not supported by Ring")

// AutoPipeline is not supported by Ring; it returns ErrRingAutoPipelineUnsupported.
func (c *Ring) AutoPipeline() (*AutoPipeliner, error) {
	return c.AutoPipelineWithOptions(nil)
}

// AutoPipelineWithOptions is not supported by Ring; it returns ErrRingAutoPipelineUnsupported.
func (c *Ring) AutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error) {
	return nil, ErrRingAutoPipelineUnsupported
}

// AsyncAutoPipeline is not supported by Ring; it returns ErrRingAutoPipelineUnsupported.
func (c *Ring) AsyncAutoPipeline() (*AutoPipeliner, error) {
	return c.AsyncAutoPipelineWithOptions(nil)
}

// AsyncAutoPipelineWithOptions is not supported by Ring; it returns ErrRingAutoPipelineUnsupported.
func (c *Ring) AsyncAutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error) {
	return nil, ErrRingAutoPipelineUnsupported
}

func (c *Ring) TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.TxPipeline().Pipelined(ctx, fn)
}

func (c *Ring) TxPipeline() Pipeliner {
	pipe := Pipeline{
		exec: func(ctx context.Context, cmds []Cmder) error {
			cmds = wrapMultiExec(ctx, cmds)
			return c.processTxPipelineHook(ctx, cmds)
		},
	}
	pipe.init()
	return &pipe
}

func (c *Ring) generalProcessPipeline(
	ctx context.Context, cmds []Cmder, tx bool,
) error {
	if tx {
		// Trim multi .. exec.
		cmds = cmds[1 : len(cmds)-1]
	}

	cmdsMap := make(map[string][]Cmder)

	for _, cmd := range cmds {
		hash := cmd.stringArg(cmdFirstKeyPosWithInfo(cmd, nil))
		if hash != "" {
			hash = c.sharding.Hash(hash)
		}
		cmdsMap[hash] = append(cmdsMap[hash], cmd)
	}

	var wg sync.WaitGroup
	errs := make(chan error, len(cmdsMap))

	for hash, cmds := range cmdsMap {
		wg.Add(1)
		go func(hash string, cmds []Cmder) {
			defer wg.Done()

			// TODO: retry?
			shard, err := c.sharding.GetByName(hash)
			if err != nil {
				setCmdsErr(cmds, err)
				return
			}

			hook := shard.Client.processPipelineHook
			if tx {
				cmds = wrapMultiExec(ctx, cmds)
				hook = shard.Client.processTxPipelineHook
			}

			if err = hook(ctx, cmds); err != nil {
				errs <- err
			}
		}(hash, cmds)
	}

	wg.Wait()
	close(errs)

	if err := <-errs; err != nil {
		return err
	}
	return cmdsFirstErr(cmds)
}

func (c *Ring) Watch(ctx context.Context, fn func(*Tx) error, keys ...string) error {
	if len(keys) == 0 {
		return fmt.Errorf("redis: Watch requires at least one key")
	}

	var shards []*ringShard

	for _, key := range keys {
		if key != "" {
			shard, err := c.sharding.GetByKey(key)
			if err != nil {
				return err
			}

			shards = append(shards, shard)
		}
	}

	if len(shards) == 0 {
		return fmt.Errorf("redis: Watch requires at least one shard")
	}

	if len(shards) > 1 {
		for _, shard := range shards[1:] {
			if shard.Client != shards[0].Client {
				err := fmt.Errorf("redis: Watch requires all keys to be in the same shard")
				return err
			}
		}
	}

	return shards[0].Client.Watch(ctx, fn, keys...)
}

// Close closes the ring client, releasing any open resources.
//
// It is rare to Close a Ring, as the Ring is meant to be long-lived
// and shared between many goroutines.
func (c *Ring) Close() error {
	c.heartbeatCancelFn()

	return c.sharding.Close()
}

// GetShardClients returns a list of all shard clients in the ring.
// This can be used to create dedicated connections (e.g., PubSub) for each shard.
func (c *Ring) GetShardClients() []*Client {
	shards := c.sharding.List()
	clients := make([]*Client, 0, len(shards))
	for _, shard := range shards {
		if shard.IsUp() {
			clients = append(clients, shard.Client)
		}
	}
	return clients
}

// GetShardClientForKey returns the shard client that would handle the given key.
// This can be used to determine which shard a particular key/channel would be routed to.
func (c *Ring) GetShardClientForKey(key string) (*Client, error) {
	shard, err := c.sharding.GetByKey(key)
	if err != nil {
		return nil, err
	}
	return shard.Client, nil
}
