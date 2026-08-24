package redis

import (
	"cmp"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/url"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/auth"
	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/hashtag"
	"github.com/redis/go-redis/v9/internal/otel"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/internal/proto"
	"github.com/redis/go-redis/v9/internal/routing"
	"github.com/redis/go-redis/v9/maintnotifications"
	"github.com/redis/go-redis/v9/push"
)

const (
	minLatencyMeasurementInterval = 10 * time.Second
)

var (
	errClusterNoNodes = errors.New("redis: cluster has no nodes")
	errNoWatchKeys    = errors.New("redis: Watch requires at least one key")
	errWatchCrosslot  = errors.New("redis: Watch requires all keys to be in the same slot")
)

// ClusterOptions are used to configure a cluster client and should be
// passed to NewClusterClient.
type ClusterOptions struct {
	// A seed list of host:port addresses of cluster nodes.
	Addrs []string

	// ClientName will execute the `CLIENT SETNAME ClientName` command for each conn.
	ClientName string

	// NewClient creates a cluster node client with provided name and options.
	// If NewClient is set by the user, the user is responsible for handling maintnotifications upgrades and push notifications.
	NewClient func(opt *Options) *Client

	// The maximum number of retries before giving up. Command is retried
	// on network errors and MOVED/ASK redirects.
	// Default is 3 retries.
	MaxRedirects int

	// Enables read-only commands on slave nodes.
	ReadOnly bool
	// Allows routing read-only commands to the closest master or slave node.
	// It automatically enables ReadOnly.
	RouteByLatency bool
	// Allows routing read-only commands to the random master or slave node.
	// It automatically enables ReadOnly.
	RouteRandomly bool

	// Optional function that returns cluster slots information.
	// It is useful to manually create cluster of standalone Redis servers
	// and load-balance read/write operations between master and slaves.
	// It can use service like ZooKeeper to maintain configuration information
	// and Cluster.ReloadState to manually trigger state reloading.
	ClusterSlots func(context.Context) ([]ClusterSlot, error)

	// Following options are copied from Options struct.

	Dialer func(ctx context.Context, network, addr string) (net.Conn, error)

	OnConnect func(ctx context.Context, cn *Conn) error

	Protocol                     int
	Username                     string
	Password                     string
	CredentialsProvider          func() (username string, password string)
	CredentialsProviderContext   func(ctx context.Context) (username string, password string, err error)
	StreamingCredentialsProvider auth.StreamingCredentialsProvider

	// MaxRetries is the maximum number of retries before giving up.
	// For ClusterClient, retries are disabled by default (set to -1),
	// because the cluster client handles all kinds of retries internally.
	// This is intentional and differs from the standalone Options default.
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

	// MaxConcurrentDials is the maximum number of concurrent connection creation goroutines.
	// If <= 0, defaults to PoolSize. If > PoolSize, it will be capped at PoolSize.
	MaxConcurrentDials int

	PoolFIFO              bool
	PoolSize              int // applies per cluster node and not for the whole cluster
	PoolTimeout           time.Duration
	MinIdleConns          int
	MaxIdleConns          int
	MaxActiveConns        int // applies per cluster node and not for the whole cluster
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
	// each node, with its own (typically larger) buffers. See the same-named
	// fields on Options for details. The pool is created only when PipelineReadBufferSize or PipelineWriteBufferSize is set (PipelinePoolSize alone does not enable it).
	PipelineReadBufferSize  int
	PipelineWriteBufferSize int
	PipelinePoolSize        int

	// AutoPipelineOptions is the default config for BOTH autopipeliner faces
	// (AutoPipeline and AsyncAutoPipeline), applied when they are called
	// without explicit options. See Options.AutoPipelineOptions.
	AutoPipelineOptions *AutoPipelineOptions

	TLSConfig *tls.Config

	// DisableRoutingPolicies disables the request/response policy routing system.
	// When disabled, all commands use the legacy routing behavior.
	// Experimental. Will be removed when shard picker is fully implemented.
	DisableRoutingPolicies bool

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

	IdentitySuffix string // Add suffix to client name. Default is empty.

	// Deprecated: All RediSearch commands now have stable RESP3 parsing and this
	// flag is a no-op. It is kept for backwards compatibility and will be removed
	// in a future release.
	UnstableResp3 bool

	// PushNotificationProcessor is the processor for handling push notifications.
	// If nil, a default processor will be created for RESP3 connections.
	PushNotificationProcessor push.NotificationProcessor

	// FailingTimeoutSeconds is the timeout in seconds for marking a cluster node as failing.
	// When a node is marked as failing, it will be avoided for this duration.
	// Default is 15 seconds.
	FailingTimeoutSeconds int

	// MaintNotificationsConfig provides custom configuration for maintnotifications upgrades.
	// When MaintNotificationsConfig.Mode is not "disabled", the client will handle
	// cluster upgrade notifications gracefully and manage connection/pool state
	// transitions seamlessly. Requires Protocol: 3 (RESP3) for push notifications.
	// If nil, maintnotifications upgrades are in "auto" mode and will be enabled if the server supports it.
	// The ClusterClient supports SMIGRATING and SMIGRATED notifications for cluster state management.
	// Individual node clients handle other maintenance notifications (MOVING, MIGRATING, etc.).
	MaintNotificationsConfig *maintnotifications.Config
	// ShardPicker is used to pick a shard when the request_policy is
	// ReqDefault and the command has no keys.
	ShardPicker routing.ShardPicker

	// ClusterStateReloadInterval is the interval for reloading the cluster state.
	// MOVED/ASK redirects still trigger an immediate reactive reload, so this
	// only bounds how stale a topology can get without traffic errors.
	// Default is 60 seconds.
	ClusterStateReloadInterval time.Duration
}

func (opt *ClusterOptions) init() {
	switch opt.MaxRedirects {
	case -1:
		opt.MaxRedirects = 0
	case 0:
		opt.MaxRedirects = 3
	}

	if opt.RouteByLatency || opt.RouteRandomly {
		opt.ReadOnly = true
	}

	if opt.DialTimeout == 0 {
		opt.DialTimeout = 5 * time.Second
	}
	if opt.DialerRetries == 0 {
		opt.DialerRetries = 5
	}
	if opt.DialerRetryTimeout == 0 {
		opt.DialerRetryTimeout = 100 * time.Millisecond
	}

	if opt.PoolSize == 0 {
		opt.PoolSize = 5 * runtime.GOMAXPROCS(0)
	}
	if opt.MaxConcurrentDials <= 0 {
		opt.MaxConcurrentDials = opt.PoolSize
	} else if opt.MaxConcurrentDials > opt.PoolSize {
		opt.MaxConcurrentDials = opt.PoolSize
	}
	if opt.ReadBufferSize == 0 {
		opt.ReadBufferSize = proto.DefaultBufferSize
	}
	if opt.WriteBufferSize == 0 {
		opt.WriteBufferSize = proto.DefaultBufferSize
	}

	switch opt.ReadTimeout {
	case -1:
		opt.ReadTimeout = 0
	case 0:
		opt.ReadTimeout = 5 * time.Second
	}
	switch opt.WriteTimeout {
	case -1:
		opt.WriteTimeout = 0
	case 0:
		opt.WriteTimeout = opt.ReadTimeout
	}

	if opt.MaxRetries == 0 {
		opt.MaxRetries = -1
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

	if opt.NewClient == nil {
		opt.NewClient = NewClient
	}

	if opt.FailingTimeoutSeconds == 0 {
		opt.FailingTimeoutSeconds = 15
	}

	if opt.ShardPicker == nil {
		opt.ShardPicker = &routing.RoundRobinPicker{}
	}

	if opt.ClusterStateReloadInterval == 0 {
		opt.ClusterStateReloadInterval = 60 * time.Second
	}
}

// ParseClusterURL parses a URL into ClusterOptions that can be used to connect to Redis.
// The URL must be in the form:
//
//	redis://<user>:<password>@<host>:<port>
//	or
//	rediss://<user>:<password>@<host>:<port>
//
// To add additional addresses, specify the query parameter, "addr" one or more times. e.g:
//
//	redis://<user>:<password>@<host>:<port>?addr=<host2>:<port2>&addr=<host3>:<port3>
//	or
//	rediss://<user>:<password>@<host>:<port>?addr=<host2>:<port2>&addr=<host3>:<port3>
//
// Most Option fields can be set using query parameters, with the following restrictions:
//   - field names are mapped using snake-case conversion: to set MaxRetries, use max_retries
//   - only scalar type fields are supported (bool, int, time.Duration)
//   - for time.Duration fields, values must be a valid input for time.ParseDuration();
//     additionally a plain integer as value (i.e. without unit) is interpreted as seconds
//   - to disable a duration field, use value less than or equal to 0; to use the default
//     value, leave the value blank or remove the parameter
//   - only the last value is interpreted if a parameter is given multiple times
//   - fields "network", "addr", "username" and "password" can only be set using other
//     URL attributes (scheme, host, userinfo, resp.), query parameters using these
//     names will be treated as unknown parameters
//   - unknown parameter names will result in an error
//
// Example:
//
//	redis://user:password@localhost:6789?dial_timeout=3&read_timeout=6s&addr=localhost:6790&addr=localhost:6791
//	is equivalent to:
//	&ClusterOptions{
//		Addr:        ["localhost:6789", "localhost:6790", "localhost:6791"]
//		DialTimeout: 3 * time.Second, // no time unit = seconds
//		ReadTimeout: 6 * time.Second,
//	}
func ParseClusterURL(redisURL string) (*ClusterOptions, error) {
	o := &ClusterOptions{}

	u, err := url.Parse(redisURL)
	if err != nil {
		return nil, err
	}

	// add base URL to the array of addresses
	// more addresses may be added through the URL params
	h, p := getHostPortWithDefaults(u)
	o.Addrs = append(o.Addrs, net.JoinHostPort(h, p))

	// setup username, password, and other configurations
	o, err = setupClusterConn(u, h, o)
	if err != nil {
		return nil, err
	}

	return o, nil
}

// setupClusterConn gets the username and password from the URL and the query parameters.
func setupClusterConn(u *url.URL, host string, o *ClusterOptions) (*ClusterOptions, error) {
	switch u.Scheme {
	case "rediss":
		o.TLSConfig = &tls.Config{ServerName: host}
		fallthrough
	case "redis":
		o.Username, o.Password = getUserPassword(u)
	default:
		return nil, fmt.Errorf("redis: invalid URL scheme: %s", u.Scheme)
	}

	// retrieve the configuration from the query parameters
	o, err := setupClusterQueryParams(u, o)
	if err != nil {
		return nil, err
	}

	return o, nil
}

// setupClusterQueryParams converts query parameters in u to option value in o.
func setupClusterQueryParams(u *url.URL, o *ClusterOptions) (*ClusterOptions, error) {
	q := queryOptions{q: u.Query()}

	o.Protocol = q.int("protocol")
	o.ClientName = q.string("client_name")
	o.MaxRedirects = q.int("max_redirects")
	o.ReadOnly = q.bool("read_only")
	o.RouteByLatency = q.bool("route_by_latency")
	o.RouteRandomly = q.bool("route_randomly")
	o.MaxRetries = q.int("max_retries")
	o.MinRetryBackoff = q.duration("min_retry_backoff")
	o.MaxRetryBackoff = q.duration("max_retry_backoff")
	o.DialTimeout = q.duration("dial_timeout")
	o.DialerRetries = q.int("dialer_retries")
	o.DialerRetryTimeout = q.duration("dialer_retry_timeout")
	o.ReadTimeout = q.duration("read_timeout")
	o.WriteTimeout = q.duration("write_timeout")
	o.PoolFIFO = q.bool("pool_fifo")
	o.PoolSize = q.int("pool_size")
	o.MaxConcurrentDials = q.int("max_concurrent_dials")
	o.MinIdleConns = q.int("min_idle_conns")
	o.MaxIdleConns = q.int("max_idle_conns")
	o.MaxActiveConns = q.int("max_active_conns")
	o.PoolTimeout = q.duration("pool_timeout")
	o.ConnMaxLifetime = q.duration("conn_max_lifetime")
	if q.has("conn_max_lifetime_jitter") {
		o.ConnMaxLifetimeJitter = min(q.duration("conn_max_lifetime_jitter"), o.ConnMaxLifetime)
	}
	o.ConnMaxIdleTime = q.duration("conn_max_idle_time")
	o.FailingTimeoutSeconds = q.int("failing_timeout_seconds")

	if q.err != nil {
		return nil, q.err
	}

	// addr can be specified as many times as needed
	addrs := q.strings("addr")
	for _, addr := range addrs {
		h, p, err := net.SplitHostPort(addr)
		if err != nil || h == "" || p == "" {
			return nil, fmt.Errorf("redis: unable to parse addr param: %s", addr)
		}

		o.Addrs = append(o.Addrs, net.JoinHostPort(h, p))
	}

	// any parameters left?
	if r := q.remaining(); len(r) > 0 {
		return nil, fmt.Errorf("redis: unexpected option: %s", strings.Join(r, ", "))
	}

	return o, nil
}

func (opt *ClusterOptions) clientOptions() *Options {
	// Clone MaintNotificationsConfig to avoid sharing between cluster node clients
	var maintNotificationsConfig *maintnotifications.Config
	if opt.MaintNotificationsConfig != nil {
		configClone := *opt.MaintNotificationsConfig
		maintNotificationsConfig = &configClone
	}

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

		MaxRetries:      opt.MaxRetries,
		MinRetryBackoff: opt.MinRetryBackoff,
		MaxRetryBackoff: opt.MaxRetryBackoff,

		DialTimeout:        opt.DialTimeout,
		DialerRetries:      opt.DialerRetries,
		DialerRetryTimeout: opt.DialerRetryTimeout,
		DialerRetryBackoff: opt.DialerRetryBackoff,
		ReadTimeout:        opt.ReadTimeout,
		WriteTimeout:       opt.WriteTimeout,

		ContextTimeoutEnabled: opt.ContextTimeoutEnabled,

		PoolFIFO:              opt.PoolFIFO,
		PoolSize:              opt.PoolSize,
		MaxConcurrentDials:    opt.MaxConcurrentDials,
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
		DisableIdentity:         opt.DisableIdentity,
		DisableIndentity:        opt.DisableIndentity,
		IdentitySuffix:          opt.IdentitySuffix,
		FailingTimeoutSeconds:   opt.FailingTimeoutSeconds,
		TLSConfig:               opt.TLSConfig,
		// If ClusterSlots is populated, then we probably have an artificial
		// cluster whose nodes are not in clustering mode (otherwise there isn't
		// much use for ClusterSlots config).  This means we cannot execute the
		// READONLY command against that node -- setting readOnly to false in such
		// situations in the options below will prevent that from happening.
		readOnly:                  opt.ReadOnly && opt.ClusterSlots == nil,
		UnstableResp3:             opt.UnstableResp3,
		MaintNotificationsConfig:  maintNotificationsConfig,
		PushNotificationProcessor: opt.PushNotificationProcessor,
	}
}

//------------------------------------------------------------------------------

type clusterNode struct {
	Client *Client

	latency    atomic.Uint32
	generation atomic.Uint32
	failing    atomic.Uint32
	loaded     atomic.Uint32

	// last time the latency measurement was performed for the node, stored in nanoseconds from epoch
	lastLatencyMeasurement atomic.Int64
}

func newClusterNodeWithNodeAddress(clOpt *ClusterOptions, addr, nodeAddress string) *clusterNode {
	opt := clOpt.clientOptions()
	opt.Addr = addr
	opt.NodeAddress = nodeAddress
	node := clusterNode{
		Client: clOpt.NewClient(opt),
	}

	node.latency.Store(math.MaxUint32)
	if clOpt.RouteByLatency {
		go node.updateLatency()
	}

	return &node
}

func (n *clusterNode) String() string {
	return n.Client.String()
}

func (n *clusterNode) Close() error {
	return n.Client.Close()
}

const maximumNodeLatency = 1 * time.Minute

func (n *clusterNode) updateLatency() {
	const numProbe = 10
	var dur uint64

	successes := 0
	for i := 0; i < numProbe; i++ {
		time.Sleep(time.Duration(10+rand.Intn(10)) * time.Millisecond)

		start := time.Now()
		err := n.Client.Ping(context.TODO()).Err()
		if err == nil {
			dur += uint64(time.Since(start) / time.Microsecond)
			successes++
		}
	}

	var latency float64
	if successes == 0 {
		// If none of the pings worked, set latency to some arbitrarily high value so this node gets
		// least priority.
		latency = float64(maximumNodeLatency / time.Microsecond)
	} else {
		latency = float64(dur) / float64(successes)
	}
	n.latency.Store(uint32(latency + 0.5))
	n.SetLastLatencyMeasurement(time.Now())
}

func (n *clusterNode) Latency() time.Duration {
	latency := n.latency.Load()
	return time.Duration(latency) * time.Microsecond
}

func (n *clusterNode) MarkAsFailing() {
	n.failing.Store(uint32(time.Now().Unix()))
	n.loaded.Store(0)
}

func (n *clusterNode) Failing() bool {
	timeout := int64(n.Client.opt.FailingTimeoutSeconds)

	failing := n.failing.Load()
	if failing == 0 {
		return false
	}
	if time.Now().Unix()-int64(failing) < timeout {
		return true
	}
	n.failing.Store(0)
	return false
}

func (n *clusterNode) Generation() uint32 {
	return n.generation.Load()
}

func (n *clusterNode) LastLatencyMeasurement() int64 {
	return n.lastLatencyMeasurement.Load()
}

func (n *clusterNode) SetGeneration(gen uint32) {
	for {
		v := n.generation.Load()
		if gen < v || n.generation.CompareAndSwap(v, gen) {
			break
		}
	}
}

func (n *clusterNode) SetLastLatencyMeasurement(t time.Time) {
	for {
		v := n.lastLatencyMeasurement.Load()
		if t.UnixNano() < v || n.lastLatencyMeasurement.CompareAndSwap(v, t.UnixNano()) {
			break
		}
	}
}

func (n *clusterNode) Loading() bool {
	loaded := n.loaded.Load()
	if loaded == 1 {
		return false
	}

	// check if the node is loading
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	err := n.Client.Ping(ctx).Err()
	loading := err != nil && isLoadingError(err)
	if !loading {
		n.loaded.Store(1)
	}
	return loading
}

//------------------------------------------------------------------------------

type clusterNodes struct {
	opt *ClusterOptions

	mu          sync.RWMutex
	addrs       []string
	nodes       map[string]*clusterNode
	activeAddrs []string
	closed      bool
	onNewNode   []func(rdb *Client)

	generation atomic.Uint32
}

func newClusterNodes(opt *ClusterOptions) *clusterNodes {
	return &clusterNodes{
		opt:   opt,
		addrs: opt.Addrs,
		nodes: make(map[string]*clusterNode),
	}
}

func (c *clusterNodes) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true

	var firstErr error
	for _, node := range c.nodes {
		if err := node.Client.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	c.nodes = nil
	c.activeAddrs = nil

	return firstErr
}

func (c *clusterNodes) OnNewNode(fn func(rdb *Client)) {
	c.mu.Lock()
	c.onNewNode = append(c.onNewNode, fn)
	c.mu.Unlock()
}

func (c *clusterNodes) Addrs() ([]string, error) {
	var addrs []string

	c.mu.RLock()
	closed := c.closed //nolint:ifshort
	if !closed {
		if len(c.activeAddrs) > 0 {
			addrs = make([]string, len(c.activeAddrs))
			copy(addrs, c.activeAddrs)
		} else {
			addrs = make([]string, len(c.addrs))
			copy(addrs, c.addrs)
		}
	}
	c.mu.RUnlock()

	if closed {
		return nil, pool.ErrClosed
	}
	if len(addrs) == 0 {
		return nil, errClusterNoNodes
	}
	return addrs, nil
}

func (c *clusterNodes) NextGeneration() uint32 {
	return c.generation.Add(1)
}

// GC removes unused nodes.
func (c *clusterNodes) GC(generation uint32) {
	var collected []*clusterNode

	c.mu.Lock()

	c.activeAddrs = c.activeAddrs[:0]
	now := time.Now()
	for addr, node := range c.nodes {
		if node.Generation() >= generation {
			c.activeAddrs = append(c.activeAddrs, addr)
			if c.opt.RouteByLatency && node.LastLatencyMeasurement() < now.Add(-minLatencyMeasurementInterval).UnixNano() {
				go node.updateLatency()
			}
			continue
		}

		delete(c.nodes, addr)
		collected = append(collected, node)
	}

	c.mu.Unlock()

	for _, node := range collected {
		_ = node.Client.Close()
	}
}

func (c *clusterNodes) GetOrCreate(addr string) (*clusterNode, error) {
	return c.GetOrCreateWithNodeAddress(addr, "")
}

func (c *clusterNodes) GetOrCreateWithNodeAddress(addr, nodeAddress string) (*clusterNode, error) {
	node, err := c.get(addr)
	if err != nil {
		return nil, err
	}
	if node != nil {
		return node, nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, pool.ErrClosed
	}

	node, ok := c.nodes[addr]
	if ok {
		return node, nil
	}

	node = newClusterNodeWithNodeAddress(c.opt, addr, nodeAddress)
	for _, fn := range c.onNewNode {
		fn(node.Client)
	}

	c.addrs = appendIfNotExist(c.addrs, addr)
	c.nodes[addr] = node

	return node, nil
}

func (c *clusterNodes) get(addr string) (*clusterNode, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, pool.ErrClosed
	}
	return c.nodes[addr], nil
}

func (c *clusterNodes) All() ([]*clusterNode, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, pool.ErrClosed
	}

	cp := make([]*clusterNode, 0, len(c.nodes))
	for _, node := range c.nodes {
		cp = append(cp, node)
	}
	return cp, nil
}

func (c *clusterNodes) Random() (*clusterNode, error) {
	addrs, err := c.Addrs()
	if err != nil {
		return nil, err
	}

	n := rand.Intn(len(addrs))
	return c.GetOrCreate(addrs[n])
}

//------------------------------------------------------------------------------

type clusterSlot struct {
	start int
	end   int
	nodes []*clusterNode
}

type clusterState struct {
	nodes   *clusterNodes
	Masters []*clusterNode
	Slaves  []*clusterNode

	slots []*clusterSlot

	generation uint32
	createdAt  time.Time
}

func newClusterState(
	nodes *clusterNodes, slots []ClusterSlot, origin string,
) (*clusterState, error) {
	c := clusterState{
		nodes: nodes,

		slots: make([]*clusterSlot, 0, len(slots)),

		generation: nodes.NextGeneration(),
		createdAt:  time.Now(),
	}

	originHost, originPort, _ := net.SplitHostPort(origin)
	isLoopbackOrigin := isLoopback(originHost)

	for _, slot := range slots {
		var nodes []*clusterNode
		for i, slotNode := range slot.Nodes {
			// slotNode.Addr is the node address from CLUSTER SLOTS
			nodeAddress := slotNode.Addr
			addr := nodeAddress
			if !isLoopbackOrigin {
				addr = replaceLoopbackHost(addr, originHost)
			}
			// TLS-only clusters (`--port 0 --tls-port 6379`) report port 0
			// in CLUSTER SLOTS. Fall back to the origin port — by definition
			// reachable, since it is the port that returned this slot map.
			// See https://github.com/redis/go-redis/issues/3726.
			addr = replaceZeroPort(addr, originPort)

			node, err := c.nodes.GetOrCreateWithNodeAddress(addr, nodeAddress)
			if err != nil {
				return nil, err
			}

			node.SetGeneration(c.generation)
			nodes = append(nodes, node)

			if i == 0 {
				c.Masters = appendIfNotExist(c.Masters, node)
			} else {
				c.Slaves = appendIfNotExist(c.Slaves, node)
			}
		}

		c.slots = append(c.slots, &clusterSlot{
			start: slot.Start,
			end:   slot.End,
			nodes: nodes,
		})
	}

	slices.SortFunc(c.slots, func(a, b *clusterSlot) int {
		return cmp.Compare(a.start, b.start)
	})

	time.AfterFunc(time.Minute, func() {
		nodes.GC(c.generation)
	})

	return &c, nil
}

func replaceLoopbackHost(nodeAddr, originHost string) string {
	nodeHost, nodePort, err := net.SplitHostPort(nodeAddr)
	if err != nil {
		return nodeAddr
	}

	nodeIP := net.ParseIP(nodeHost)
	if nodeIP == nil {
		return nodeAddr
	}

	if !nodeIP.IsLoopback() {
		return nodeAddr
	}

	// Use origin host which is not loopback and node port.
	return net.JoinHostPort(originHost, nodePort)
}

// replaceZeroPort substitutes originPort for a node port of "0", which is
// what CLUSTER SLOTS reports for TLS-only clusters started with
// `--port 0 --tls-port <port>`. Non-zero ports and addresses without a
// recoverable origin port are returned unchanged.
func replaceZeroPort(nodeAddr, originPort string) string {
	if originPort == "" || originPort == "0" {
		return nodeAddr
	}
	nodeHost, nodePort, err := net.SplitHostPort(nodeAddr)
	if err != nil || nodePort != "0" {
		return nodeAddr
	}
	return net.JoinHostPort(nodeHost, originPort)
}

// isLoopback returns true if the host is a loopback address.
// For IP addresses, it uses net.IP.IsLoopback().
// For hostnames, it recognizes well-known loopback hostnames like "localhost"
// and Docker-specific loopback patterns like "*.docker.internal".
func isLoopback(host string) bool {
	ip := net.ParseIP(host)
	if ip != nil {
		return ip.IsLoopback()
	}

	if strings.ToLower(host) == "localhost" {
		return true
	}

	if strings.HasSuffix(strings.ToLower(host), ".docker.internal") {
		return true
	}

	return false
}

func (c *clusterState) slotMasterNode(slot int) (*clusterNode, error) {
	nodes := c.slotNodes(slot)
	if len(nodes) > 0 {
		return nodes[0], nil
	}
	return c.nodes.Random()
}

func (c *clusterState) slotSlaveNode(slot int) (*clusterNode, error) {
	nodes := c.slotNodes(slot)
	switch len(nodes) {
	case 0:
		return c.nodes.Random()
	case 1:
		return nodes[0], nil
	case 2:
		slave := nodes[1]
		if !slave.Failing() && !slave.Loading() {
			return slave, nil
		}
		return nodes[0], nil
	default:
		var slave *clusterNode
		for i := 0; i < 10; i++ {
			n := rand.Intn(len(nodes)-1) + 1
			slave = nodes[n]
			if !slave.Failing() && !slave.Loading() {
				return slave, nil
			}
		}

		// All slaves are loading - use master.
		return nodes[0], nil
	}
}

func (c *clusterState) slotClosestNode(slot int) (*clusterNode, error) {
	nodes := c.slotNodes(slot)
	if len(nodes) == 0 {
		return c.nodes.Random()
	}

	allNodesFailing := true
	var (
		closestNonFailingNode *clusterNode
		closestNode           *clusterNode
		minLatency            time.Duration
	)

	// setting the max possible duration as zerovalue for minlatency
	minLatency = time.Duration(math.MaxInt64)

	for _, n := range nodes {
		if closestNode == nil || n.Latency() < minLatency {
			closestNode = n
			minLatency = n.Latency()
			if !n.Failing() {
				closestNonFailingNode = n
				allNodesFailing = false
			}
		}
	}

	// pick the healthly node with the lowest latency
	if !allNodesFailing && closestNonFailingNode != nil {
		return closestNonFailingNode, nil
	}

	// if all nodes are failing, we will pick the temporarily failing node with lowest latency
	if minLatency < maximumNodeLatency && closestNode != nil {
		internal.Logger.Printf(context.TODO(), "redis: all nodes are marked as failed, picking the temporarily failing node with lowest latency")
		return closestNode, nil
	}

	// If all nodes are having the maximum latency(all pings are failing) - return a random node across the cluster
	internal.Logger.Printf(context.TODO(), "redis: pings to all nodes are failing, picking a random node across the cluster")
	return c.nodes.Random()
}

func (c *clusterState) slotRandomNode(slot int) (*clusterNode, error) {
	nodes := c.slotNodes(slot)
	if len(nodes) == 0 {
		return c.nodes.Random()
	}
	if len(nodes) == 1 {
		return nodes[0], nil
	}
	randomNodes := rand.Perm(len(nodes))
	for _, idx := range randomNodes {
		if node := nodes[idx]; !node.Failing() {
			return node, nil
		}
	}
	return nodes[randomNodes[0]], nil
}

func (c *clusterState) slotShardPickerSlaveNode(slot int, shardPicker routing.ShardPicker) (*clusterNode, error) {
	nodes := c.slotNodes(slot)
	if len(nodes) == 0 {
		return c.nodes.Random()
	}

	// nodes[0] is master, nodes[1:] are slaves
	// First, try all slave nodes for this slot using ShardPicker order
	slaves := nodes[1:]
	if len(slaves) > 0 {
		for i := 0; i < len(slaves); i++ {
			idx := shardPicker.Next(len(slaves))
			slave := slaves[idx]
			if !slave.Failing() && !slave.Loading() {
				return slave, nil
			}
		}
	}

	// All slaves are failing or loading - return master
	return nodes[0], nil
}

func (c *clusterState) slotNodes(slot int) []*clusterNode {
	i := sort.Search(len(c.slots), func(i int) bool {
		return c.slots[i].end >= slot
	})
	if i >= len(c.slots) {
		return nil
	}
	x := c.slots[i]
	if slot >= x.start && slot <= x.end {
		return x.nodes
	}
	return nil
}

//------------------------------------------------------------------------------

type clusterStateHolder struct {
	load func(ctx context.Context) (*clusterState, error)

	reloadInterval time.Duration
	state          atomic.Value
	reloading      atomic.Uint32
	reloadPending  atomic.Uint32 // set to 1 when reload is requested during active reload
}

func newClusterStateHolder(load func(ctx context.Context) (*clusterState, error), reloadInterval time.Duration) *clusterStateHolder {
	return &clusterStateHolder{
		load:           load,
		reloadInterval: reloadInterval,
	}
}

func (c *clusterStateHolder) Reload(ctx context.Context) (*clusterState, error) {
	state, err := c.load(ctx)
	if err != nil {
		return nil, err
	}
	c.state.Store(state)
	return state, nil
}

func (c *clusterStateHolder) LazyReload() {
	// If already reloading, mark that another reload is pending
	if !c.reloading.CompareAndSwap(0, 1) {
		c.reloadPending.Store(1)
		return
	}

	go func() {
		for {
			_, err := c.Reload(context.Background())
			if err != nil {
				c.reloadPending.Store(0)
				c.reloading.Store(0)
				return
			}

			// Clear pending flag after reload completes, before cooldown
			// This captures notifications that arrived during the reload
			c.reloadPending.Store(0)

			// Wait cooldown period
			time.Sleep(200 * time.Millisecond)

			// Check if another reload was requested during cooldown
			if c.reloadPending.Load() == 0 {
				// No pending reload, we're done
				c.reloading.Store(0)
				return
			}

			// Pending reload requested, loop to reload again
		}
	}()
}

func (c *clusterStateHolder) Get(ctx context.Context) (*clusterState, error) {
	v := c.state.Load()
	if v == nil {
		return c.Reload(ctx)
	}

	state := v.(*clusterState)
	if time.Since(state.createdAt) > c.reloadInterval {
		c.LazyReload()
	}
	return state, nil
}

func (c *clusterStateHolder) ReloadOrGet(ctx context.Context) (*clusterState, error) {
	state, err := c.Reload(ctx)
	if err == nil {
		return state, nil
	}
	return c.Get(ctx)
}

//------------------------------------------------------------------------------

// ClusterClient is a Redis Cluster client representing a pool of zero
// or more underlying connections. It's safe for concurrent use by
// multiple goroutines.
type ClusterClient struct {
	opt             *ClusterOptions
	nodes           *clusterNodes
	state           *clusterStateHolder
	cmdsInfoCache   *cmdsInfoCache
	cmdInfoResolver *commandInfoResolver
	cmdable
	hooksMixin

	// himport is the cluster-wide HIMPORT fieldset registry, shared with
	// every node client (masters and replicas alike — roles change with the
	// topology) so any connection serving an HIMPORT SET can lazily replay
	// the PREPARE (see himport.go, himport_cluster.go).
	himport *himportRegistry

	autopipelinerMu     *sync.Mutex    // guards the autopipeliner fields against concurrent first-call creation
	autopipeliner       *AutoPipeliner // blocking face (ClusterClient.AutoPipeline)
	asyncAutopipeliner  *AutoPipeliner // deferred face (ClusterClient.AsyncAutoPipeline)
	autopipelinerClosed bool           // set by Close: refuse to resurrect a pipeliner on a closed client
}

// NewClusterClient returns a Redis Cluster client as described in
// https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec.
// Passing nil ClusterOptions will cause a panic.
func NewClusterClient(opt *ClusterOptions) *ClusterClient {
	if opt == nil {
		panic("redis: NewClusterClient nil options")
	}
	opt.init()

	c := &ClusterClient{
		opt:             opt,
		nodes:           newClusterNodes(opt),
		himport:         newHImportRegistry(),
		autopipelinerMu: &sync.Mutex{},
	}

	// Every node client shares the cluster-wide fieldset registry, replicas
	// included: a promoted replica's connections carry no prepared flags, so
	// the first HIMPORT SET routed to it replays the PREPARE lazily.
	c.nodes.OnNewNode(func(nodeClient *Client) {
		nodeClient.himport = c.himport
	})

	c.cmdsInfoCache = newCmdsInfoCache(c.cmdsInfo)

	c.state = newClusterStateHolder(c.loadState, opt.ClusterStateReloadInterval)

	c.SetCommandInfoResolver(NewDefaultCommandPolicyResolver())

	c.cmdable = c.Process
	c.initHooks(hooks{
		dial:       nil,
		process:    c.process,
		pipeline:   c.processPipeline,
		txPipeline: c.processTxPipeline,
	})

	// Set up SMIGRATED notification handling for cluster state reload
	// When a node client receives a SMIGRATED notification, it should trigger
	// cluster state reload on the parent ClusterClient
	if opt.MaintNotificationsConfig != nil {
		c.nodes.OnNewNode(func(nodeClient *Client) {
			manager := nodeClient.GetMaintNotificationsManager()
			if manager != nil {
				manager.SetClusterStateReloadCallback(func(ctx context.Context, hostPort string, slotRanges []string) {
					// Log the migration details for now
					if internal.LogLevel.InfoOrAbove() {
						internal.Logger.Printf(ctx, "cluster: slots %v migrated to %s, reloading cluster state", slotRanges, hostPort)
					}
					// Currently we reload the entire cluster state
					// In the future, this could be optimized to reload only the specific slots
					c.state.LazyReload()
				})
			}
		})
	}

	return c
}

// Options returns read-only *ClusterOptions that were used to create the client.
// Any alteration of the returned *ClusterOptions may result in undefined behaviour.
func (c *ClusterClient) Options() *ClusterOptions {
	return c.opt
}

// ReloadState reloads cluster state. If available it calls ClusterSlots func
// to get cluster slots information.
func (c *ClusterClient) ReloadState(ctx context.Context) {
	c.state.LazyReload()
}

// Close closes the cluster client, releasing any open resources.
//
// It is rare to Close a ClusterClient, as the ClusterClient is meant
// to be long-lived and shared between many goroutines.
func (c *ClusterClient) Close() error {
	// Stop both cached autopipeliners (blocking and async faces) before
	// closing nodes, so its background flusher goroutines don't outlive the
	// client. AutoPipeliner.Close is idempotent and nil-safe here.
	c.autopipelinerMu.Lock()
	ap, async := c.autopipeliner, c.asyncAutopipeliner
	c.autopipeliner, c.asyncAutopipeliner = nil, nil
	c.autopipelinerClosed = true // getters refuse to resurrect on a closed client
	c.autopipelinerMu.Unlock()
	var firstErr error
	for _, p := range []*AutoPipeliner{ap, async} {
		if p != nil {
			if err := p.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	if err := c.nodes.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

func (c *ClusterClient) Process(ctx context.Context, cmd Cmder) error {
	err := c.processHook(ctx, cmd)
	cmd.SetErr(err)
	return err
}

func (c *ClusterClient) process(ctx context.Context, cmd Cmder) error {
	slot := c.cmdSlot(cmd, -1)
	var node *clusterNode
	var moved bool
	var ask bool
	var lastErr error
	for attempt := 0; attempt <= c.opt.MaxRedirects; attempt++ {
		// MOVED and ASK responses are not transient errors that require retry delay; they
		// should be attempted immediately.
		if attempt > 0 && !moved && !ask {
			if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
				return err
			}
		}

		if node == nil {
			var err error
			if !c.opt.DisableRoutingPolicies && c.opt.ShardPicker != nil {
				node, err = c.cmdNodeWithShardPicker(ctx, cmd.Name(), slot, c.opt.ShardPicker)
			} else {
				node, err = c.cmdNode(ctx, cmd.Name(), slot)
			}
			if err != nil {
				return err
			}
		}

		if ask {
			ask = false
			pipe := node.Client.Pipeline()
			_ = pipe.Process(ctx, NewCmd(ctx, "asking"))
			_ = pipe.Process(ctx, cmd)
			_, lastErr = pipe.Exec(ctx)
		} else {
			if !c.opt.DisableRoutingPolicies {
				lastErr = c.routeAndRun(ctx, cmd, node)
			} else {
				lastErr = node.Client.Process(ctx, cmd)
			}
		}

		// If there is no error - we are done.
		if lastErr == nil {
			return nil
		}
		if isReadOnly := isReadOnlyError(lastErr); isReadOnly || lastErr == pool.ErrClosed {
			if isReadOnly {
				c.state.LazyReload()
			}
			node = nil
			continue
		}

		// If slave is loading - pick another node.
		if c.opt.ReadOnly && isLoadingError(lastErr) {
			node.MarkAsFailing()
			node = nil
			continue
		}

		var addr string
		moved, ask, addr = isMovedError(lastErr)
		if moved || ask {
			c.state.LazyReload()

			// Record error metrics
			if errorCallback := pool.GetMetricErrorCallback(); errorCallback != nil {
				errorType := "MOVED"
				statusCode := "MOVED"
				if ask {
					errorType = "ASK"
					statusCode = "ASK"
				}
				// MOVED/ASK are not internal errors, and this is the first attempt (retry count = 0)
				errorCallback(ctx, errorType, nil, statusCode, false, 0)
			}

			var err error
			node, err = c.nodes.GetOrCreate(addr)
			if err != nil {
				return err
			}
			continue
		}

		if shouldRetry(lastErr, cmd.readTimeout() == nil) && !cmd.NoRetry() {
			// First retry the same node.
			if attempt == 0 {
				continue
			}

			// Second try another node.
			node.MarkAsFailing()
			node = nil
			continue
		}

		return lastErr
	}
	return lastErr
}

func (c *ClusterClient) OnNewNode(fn func(rdb *Client)) {
	c.nodes.OnNewNode(fn)
}

// ForEachMaster concurrently calls the fn on each master node in the cluster.
// It returns the first error if any.
func (c *ClusterClient) ForEachMaster(
	ctx context.Context,
	fn func(ctx context.Context, client *Client) error,
) error {
	state, err := c.state.ReloadOrGet(ctx)
	if err != nil {
		return err
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 1)

	for _, master := range state.Masters {
		wg.Add(1)
		go func(node *clusterNode) {
			defer wg.Done()
			err := fn(ctx, node.Client)
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
			}
		}(master)
	}

	wg.Wait()

	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

// ForEachSlave concurrently calls the fn on each slave node in the cluster.
// It returns the first error if any.
func (c *ClusterClient) ForEachSlave(
	ctx context.Context,
	fn func(ctx context.Context, client *Client) error,
) error {
	state, err := c.state.ReloadOrGet(ctx)
	if err != nil {
		return err
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 1)

	for _, slave := range state.Slaves {
		wg.Add(1)
		go func(node *clusterNode) {
			defer wg.Done()
			err := fn(ctx, node.Client)
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
			}
		}(slave)
	}

	wg.Wait()

	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

// ForEachShard concurrently calls the fn on each known node in the cluster.
// It returns the first error if any.
func (c *ClusterClient) ForEachShard(
	ctx context.Context,
	fn func(ctx context.Context, client *Client) error,
) error {
	state, err := c.state.ReloadOrGet(ctx)
	if err != nil {
		return err
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 1)

	worker := func(node *clusterNode) {
		defer wg.Done()
		err := fn(ctx, node.Client)
		if err != nil {
			select {
			case errCh <- err:
			default:
			}
		}
	}

	for _, node := range state.Masters {
		wg.Add(1)
		go worker(node)
	}
	for _, node := range state.Slaves {
		wg.Add(1)
		go worker(node)
	}

	wg.Wait()

	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

// PoolStats returns accumulated connection pool stats.
func (c *ClusterClient) PoolStats() *PoolStats {
	var acc PoolStats

	state, _ := c.state.Get(context.TODO())
	if state == nil {
		return &acc
	}

	for _, node := range state.Masters {
		s := node.Client.connPool.Stats()
		acc.Hits += s.Hits
		acc.Misses += s.Misses
		acc.Timeouts += s.Timeouts
		acc.WaitCount += s.WaitCount
		acc.WaitDurationNs += s.WaitDurationNs

		acc.TotalConns += s.TotalConns
		acc.IdleConns += s.IdleConns
		acc.StaleConns += s.StaleConns
	}

	for _, node := range state.Slaves {
		s := node.Client.connPool.Stats()
		acc.Hits += s.Hits
		acc.Misses += s.Misses
		acc.Timeouts += s.Timeouts
		acc.WaitCount += s.WaitCount
		acc.WaitDurationNs += s.WaitDurationNs

		acc.TotalConns += s.TotalConns
		acc.IdleConns += s.IdleConns
		acc.StaleConns += s.StaleConns
	}

	return &acc
}

func (c *ClusterClient) loadState(ctx context.Context) (*clusterState, error) {
	if c.opt.ClusterSlots != nil {
		slots, err := c.opt.ClusterSlots(ctx)
		if err != nil {
			return nil, err
		}
		return newClusterState(c.nodes, slots, "")
	}

	addrs, err := c.nodes.Addrs()
	if err != nil {
		return nil, err
	}

	var firstErr error

	for _, idx := range rand.Perm(len(addrs)) {
		addr := addrs[idx]

		node, err := c.nodes.GetOrCreate(addr)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}

		slots, err := node.Client.ClusterSlots(ctx).Result()
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}

		return newClusterState(c.nodes, slots, addr)
	}

	/*
	 * No node is connectable. It's possible that all nodes' IP has changed.
	 * Clear activeAddrs to let client be able to re-connect using the initial
	 * setting of the addresses (e.g. [redis-cluster-0:6379, redis-cluster-1:6379]),
	 * which might have chance to resolve domain name and get updated IP address.
	 */
	c.nodes.mu.Lock()
	c.nodes.activeAddrs = nil
	c.nodes.mu.Unlock()

	return nil, firstErr
}

func (c *ClusterClient) Pipeline() Pipeliner {
	pipe := Pipeline{
		exec: pipelineExecer(c.processPipelineHook),
	}
	pipe.init()
	return &pipe
}

// clusterAutoPipelineOptions applies the cluster shard-count default: commands
// are routed to shards by slot (see installAutoPipelineSharding), so unlike a
// standalone client — which defaults to a single deep queue — a cluster client
// wants several shards to keep concurrent nodes' batches separate. The caller's
// config is copied before the default is filled in, never mutated.
func clusterAutoPipelineOptions(cfg *AutoPipelineOptions) *AutoPipelineOptions {
	c2 := *cfg
	if c2.NumShards == 0 {
		c2.NumShards = numAutoPipelineShards()
	}
	// A cluster always routes by slot, so per-key order holds regardless of shard
	// count; mark it so construction's NumShards ordering check (which targets
	// round-robin sharding) does not reject the cluster default or an explicit
	// NumShards on the deferred (async) face.
	c2.contentSharded = true
	return &c2
}

// AutoPipeline returns the blocking autopipeliner for this cluster client: each
// command call blocks until executed (drop-in shape) while the engine batches
// concurrent callers into pipelines. Commands keep per-goroutine order; across
// nodes, ordering is per key (slot routing keeps a key on one shard and node
// sub-pipelines execute concurrently). Use AutoPipelineWithOptions to override
// DefaultBlockingAutoPipelineOptions. Cached/shared; first call's config wins.
// Close it (or the client) to release its goroutines.
//
// It returns an error if the supplied config is invalid (e.g. MaxConcurrentBatches>1
// without Unordered, or a negative size); on error no instance is cached.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *ClusterClient) AutoPipeline() (*AutoPipeliner, error) {
	return c.AutoPipelineWithOptions(nil)
}

// AutoPipelineWithOptions is AutoPipeline with explicit options instead of
// ClusterOptions.AutoPipelineOptions / the default. Cached/shared; first call wins.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *ClusterClient) AutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error) {
	return getOrCreateAutoPipeliner(c.autopipelinerMu, &c.autopipeliner, &c.autopipelinerClosed, nil, config,
		func() *AutoPipelineOptions {
			if c.opt.AutoPipelineOptions != nil {
				return c.opt.AutoPipelineOptions
			}
			return DefaultBlockingAutoPipelineOptions()
		},
		func(cfg *AutoPipelineOptions) (*AutoPipeliner, error) {
			ap, err := newAutoPipeliner(c, clusterAutoPipelineOptions(cfg), true)
			if err != nil {
				return nil, err
			}
			c.installAutoPipelineSharding(ap)
			return ap, nil
		})
}

// installAutoPipelineSharding routes commands to shards by cluster slot so each
// shard's batch lands on a single master node, keeping per-node pipelines deep
// instead of splitting every batch across all nodes at flush. Cluster slots are
// contiguous per node, so bucketing by slot range (slot*shards/16384) keeps a
// node's slots together. Keyless commands hash to slot -1 → bucket 0; multi-node
// commands are already rejected from pipelines, so only single-node commands
// reach here.
func (c *ClusterClient) installAutoPipelineSharding(ap *AutoPipeliner) {
	// Reject commands whose request policy cannot ride a pipeline (ReqAllNodes/
	// ReqAllShards/ReqMultiShard) at submit, BEFORE they can join a merged
	// batch: mapCmdsByNode fails a whole mapping on such a command (user
	// pipelines are all-or-nothing), and one autopipeline caller must not be
	// able to poison unrelated callers' batches. Rejecting here also keeps the
	// lone-command fast path consistent with batched dispatch — the command is
	// refused regardless of what it happens to coalesce with.
	ap.setPreflight(func(ctx context.Context, cmd Cmder) error {
		if c.cmdInfoResolver == nil {
			return nil
		}
		if policy := c.cmdInfoResolver.GetCommandPolicy(ctx, cmd); policy != nil && !policy.CanBeUsedInPipeline() {
			return fmt.Errorf(
				"redis: cannot pipeline command %q with request policy ReqAllNodes/ReqAllShards/ReqMultiShard; Note: This behavior is subject to change in the future", cmd.Name(),
			)
		}
		return nil
	})
	// Commands whose routing is not slot-derived must not be coalesced: a solo
	// flush reaches ClusterClient.process and its special handling (FT.CURSOR
	// READ/DEL are sticky to the node holding the cursor), but inside a batch
	// mapCmdsByNode routes by slot and can hit the wrong shard — visible only
	// under concurrent traffic, which is the worst way to find it. Divert them
	// instead of rejecting: they work fine on their own connection (review
	// finding by codex on #3942).
	ap.setMustDivert(func(ctx context.Context, cmd Cmder) bool {
		if c.cmdInfoResolver == nil {
			return false
		}
		policy := c.cmdInfoResolver.GetCommandPolicy(ctx, cmd)
		return policy != nil && policy.Request == routing.ReqSpecial
	})

	const slots = 16384
	n := ap.numShards()
	ap.setShardFn(func(cmd Cmder) int {
		// Compute the exact slot once and cache it on the command; the flush
		// router (mapCmdsByNode) reuses the cached value, so the slot is resolved
		// once per command, not twice. Keyless (slot -1) buckets to shard 0.
		slot := c.cmdSlot(cmd, -1)
		if slot < 0 {
			return 0
		}
		return slot * n / slots
	})
}

// AsyncAutoPipeline returns the deferred autopipeliner: command calls return
// immediately and the result accessors block. Submit a window then read results
// for the highest throughput. By default,
// ClusterOptions.AutoPipelineOptions is used if set, otherwise
// DefaultAutoPipelineOptions. Ordering across nodes is per key: slot routing
// keeps a key on one shard, and node sub-pipelines execute concurrently. Use
// AsyncAutoPipelineWithOptions to override. Cached/shared; first call's config wins.
//
// It returns an error if the supplied config is invalid (e.g. MaxConcurrentBatches>1
// without Unordered, or a negative size); on error no instance is cached.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *ClusterClient) AsyncAutoPipeline() (*AutoPipeliner, error) {
	return c.AsyncAutoPipelineWithOptions(nil)
}

// AsyncAutoPipelineWithOptions is AsyncAutoPipeline with an explicit config
// instead of ClusterOptions.AutoPipelineOptions / the default. Cached/shared.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *ClusterClient) AsyncAutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error) {
	return getOrCreateAutoPipeliner(c.autopipelinerMu, &c.asyncAutopipeliner, &c.autopipelinerClosed, nil, config,
		func() *AutoPipelineOptions {
			if c.opt.AutoPipelineOptions != nil {
				return c.opt.AutoPipelineOptions
			}
			return DefaultAutoPipelineOptions()
		},
		func(cfg *AutoPipelineOptions) (*AutoPipeliner, error) {
			ap, err := newAutoPipeliner(c, clusterAutoPipelineOptions(cfg), false)
			if err != nil {
				return nil, err
			}
			c.installAutoPipelineSharding(ap)
			return ap, nil
		})
}

func (c *ClusterClient) Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.Pipeline().Pipelined(ctx, fn)
}

func (c *ClusterClient) processPipeline(ctx context.Context, cmds []Cmder) error {
	// Only call time.Now() if pipeline operation duration callback is set to avoid overhead
	var operationStart time.Time
	pipelineOpDurationCallback := otel.GetPipelineOperationDurationCallback()
	if pipelineOpDurationCallback != nil {
		operationStart = time.Now()
	}
	totalAttempts := 0

	cmdsMap := newCmdsMap()

	if err := c.mapCmdsByNode(ctx, cmdsMap, cmds); err != nil {
		setCmdsErr(cmds, err)
		if pipelineOpDurationCallback != nil {
			operationDuration := time.Since(operationStart)
			pipelineOpDurationCallback(ctx, operationDuration, "PIPELINE", len(cmds), 1, err, nil, 0)
		}
		return err
	}

	var lastErr error
	for attempt := 0; attempt <= c.opt.MaxRedirects; attempt++ {
		totalAttempts++
		if attempt > 0 {
			if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
				setCmdsErr(cmds, err)
				if pipelineOpDurationCallback != nil {
					operationDuration := time.Since(operationStart)
					pipelineOpDurationCallback(ctx, operationDuration, "PIPELINE", len(cmds), totalAttempts, err, nil, 0)
				}
				return err
			}
		}

		failedCmds := newCmdsMap()
		var wg sync.WaitGroup

		for node, cmds := range cmdsMap.m {
			wg.Add(1)
			go func(node *clusterNode, cmds []Cmder) {
				defer wg.Done()
				c.processPipelineNode(ctx, node, cmds, failedCmds)
			}(node, cmds)
		}

		wg.Wait()
		if len(failedCmds.m) == 0 {
			break
		}
		cmdsMap = failedCmds
		lastErr = cmdsFirstErr(cmds)
	}

	// Record pipeline operation duration
	if pipelineOpDurationCallback != nil {
		operationDuration := time.Since(operationStart)
		finalErr := cmdsFirstErr(cmds)
		if finalErr == nil {
			finalErr = lastErr
		}
		pipelineOpDurationCallback(ctx, operationDuration, "PIPELINE", len(cmds), totalAttempts, finalErr, nil, 0)
	}

	return cmdsFirstErr(cmds)
}

func (c *ClusterClient) mapCmdsByNode(ctx context.Context, cmdsMap *cmdsMap, cmds []Cmder) error {
	state, err := c.state.Get(ctx)
	if err != nil {
		return err
	}

	if c.opt.ReadOnly && c.cmdsAreReadOnly(ctx, cmds) {
		for _, cmd := range cmds {
			var policy *routing.CommandPolicy
			if c.cmdInfoResolver != nil {
				policy = c.cmdInfoResolver.GetCommandPolicy(ctx, cmd)
			}
			if policy != nil && !policy.CanBeUsedInPipeline() {
				// All-or-nothing: a user Pipeline() relies on the whole batch
				// either dispatching or failing before anything executes, so a
				// non-pipelineable command fails the entire mapping pre-dispatch.
				// Autopipeline batches never reach here with such a command: the
				// cluster face rejects them at submit (see the preflight installed
				// by installAutoPipelineSharding), so one caller's bad command
				// cannot poison a merged batch.
				err := fmt.Errorf(
					"redis: cannot pipeline command %q with request policy ReqAllNodes/ReqAllShards/ReqMultiShard; Note: This behavior is subject to change in the future", cmd.Name(),
				)
				setCmdsErr(cmds, err)
				return err
			}
			slot := c.cmdSlot(cmd, -1)
			var node *clusterNode
			// For keyless commands (slot == -1), use ShardPicker if routing policies are enabled
			if slot == -1 && !c.opt.DisableRoutingPolicies && c.opt.ShardPicker != nil {
				if len(state.Masters) == 0 {
					return errClusterNoNodes
				}
				// For read-only keyless commands, pick from all nodes (masters + slaves).
				// Index directly instead of building a combined slice, which would
				// append into the shared snapshot's spare capacity and race.
				idx := c.opt.ShardPicker.Next(len(state.Masters) + len(state.Slaves))
				if idx < len(state.Masters) {
					node = state.Masters[idx]
				} else {
					node = state.Slaves[idx-len(state.Masters)]
				}
			} else {
				node, err = c.slotReadOnlyNode(state, slot)
				if err != nil {
					return err
				}
			}
			cmdsMap.Add(node, cmd)
		}
		return nil
	}

	for _, cmd := range cmds {
		var policy *routing.CommandPolicy
		if c.cmdInfoResolver != nil {
			policy = c.cmdInfoResolver.GetCommandPolicy(ctx, cmd)
		}
		if policy != nil && !policy.CanBeUsedInPipeline() {
			// All-or-nothing: a user Pipeline() relies on the whole batch
			// either dispatching or failing before anything executes, so a
			// non-pipelineable command fails the entire mapping pre-dispatch.
			// Autopipeline batches never reach here with such a command: the
			// cluster face rejects them at submit (see the preflight installed
			// by installAutoPipelineSharding), so one caller's bad command
			// cannot poison a merged batch.
			err := fmt.Errorf(
				"redis: cannot pipeline command %q with request policy ReqAllNodes/ReqAllShards/ReqMultiShard; Note: This behavior is subject to change in the future", cmd.Name(),
			)
			setCmdsErr(cmds, err)
			return err
		}
		slot := c.cmdSlot(cmd, -1)
		var node *clusterNode
		// For keyless commands (slot == -1), use ShardPicker if routing policies are enabled
		if slot == -1 && !c.opt.DisableRoutingPolicies && c.opt.ShardPicker != nil {
			if len(state.Masters) == 0 {
				return errClusterNoNodes
			}
			idx := c.opt.ShardPicker.Next(len(state.Masters))
			node = state.Masters[idx]
		} else {
			node, err = state.slotMasterNode(slot)
			if err != nil {
				return err
			}
		}
		cmdsMap.Add(node, cmd)
	}
	return nil
}

func (c *ClusterClient) cmdsAreReadOnly(ctx context.Context, cmds []Cmder) bool {
	for _, cmd := range cmds {
		cmdInfo := c.cmdInfo(ctx, cmd.Name())
		if cmdInfo == nil || !cmdInfo.ReadOnly {
			return false
		}
	}
	return true
}

func (c *ClusterClient) processPipelineNode(
	ctx context.Context, node *clusterNode, cmds []Cmder, failedCmds *cmdsMap,
) {
	// This call runs on a per-node fan-out goroutine, so register it as an
	// executor of every deferred-face batch among cmds: a NODE-level hook
	// (OnNewNode — redisotel's tracing) reading a result before next() must
	// get the not-yet-executed view from the accessor guards instead of
	// blocking on a batch only this call chain completes (reproduced as a
	// permanent wedge with a rediscmd-shaped Err() peek).
	unregister := registerBatchExecutors(cmds)
	defer unregister()

	// executed guards against a node-level hook short-circuiting (returning
	// without calling next): the inner callback then never runs, and without
	// surfacing the chain's error the cluster pipeline would report success
	// for commands that were never sent.
	executed := false
	err := node.Client.withProcessPipelineHook(ctx, cmds, func(ctx context.Context, cmds []Cmder) error {
		executed = true
		// Acquire through the node's dedicated pipeline pool when one is
		// configured (Pipeline*BufferSize propagate to node clients via
		// clientOptions); withPipelineConn falls back to the main pool
		// otherwise, preserving the previous behavior. entered distinguishes
		// an acquisition failure (fn never ran) from an execution error.
		entered := false
		err := node.Client.withPipelineConn(ctx, func(ctx context.Context, cn *pool.Conn) error {
			entered = true
			return c.processPipelineNodeConn(ctx, node, cn, cmds, failedCmds)
		})
		if err != nil && !entered {
			if !isContextError(err) {
				node.MarkAsFailing()
			}
			_ = c.mapCmdsByNode(ctx, failedCmds, cmds)
			setCmdsErr(cmds, err)
		}
		return err
	})
	if !executed {
		// A hook returned without calling next. If it supplied an error that is
		// a deliberate abort: set it and do not remap for retry (a retry would
		// re-run the same hook). If it returned nil it short-circuited
		// SUCCESSFULLY, having served the batch itself — the same thing a plain
		// Pipeline hook may do — so setCmdsErr(nil) leaves the values it set
		// intact (review finding by codex on #3942).
		setCmdsErr(cmds, err)
		return
	}
	if err != nil && cmdsFirstErr(cmds) == nil {
		// Post-next verdict from a node-level hook on an all-clean sub-batch:
		// the exec fully succeeded, so the error can only be the hook's own —
		// apply it, mirroring AutoPipeliner.dispatchCmds. On a mixed batch the
		// exec-recorded outcomes win (hooks conventionally echo next's error,
		// and stamping the echo would overwrite successful replies). No remap:
		// retrying would re-run the same hook.
		setCmdsErr(cmds, err)
	}
}

func (c *ClusterClient) processPipelineNodeConn(
	ctx context.Context, node *clusterNode, cn *pool.Conn, cmds []Cmder, failedCmds *cmdsMap,
) error {
	// HIMPORT bookkeeping: pending discards for this session and PREPAREs
	// for registered fieldsets the batch references get written ahead of
	// the batch (see himport.go).
	injected := node.Client.himportInjectedCmds(ctx, cn, cmds)

	if err := cn.WithWriter(c.context(ctx), c.opt.WriteTimeout, func(wr *proto.Writer) error {
		for _, ic := range injected {
			if err := writeCmd(wr, ic); err != nil {
				return err
			}
		}
		return writeCmds(wr, cmds)
	}); err != nil {
		if isBadConn(err, false, node.Client.getAddr()) {
			node.MarkAsFailing()
		}
		if shouldRetry(err, true) && !cmdsContainNoRetry(cmds) {
			_ = c.mapCmdsByNode(ctx, failedCmds, cmds)
		}
		setCmdsErr(cmds, err)
		return err
	}

	return cn.WithReader(c.context(ctx), c.opt.ReadTimeout, func(rd *proto.Reader) error {
		if err := node.Client.himportReadInjectedReplies(ctx, cn, rd, injected); err != nil {
			// Transport error with the batch replies unread: same handling
			// as a write error — the batch may be retried on a fresh
			// connection.
			if isBadConn(err, false, node.Client.getAddr()) {
				node.MarkAsFailing()
			}
			if shouldRetry(err, true) && !cmdsContainNoRetry(cmds) {
				_ = c.mapCmdsByNode(ctx, failedCmds, cmds)
			}
			setCmdsErr(cmds, err)
			return err
		}
		err := c.pipelineReadCmds(ctx, node, cn, rd, cmds, failedCmds)
		if err == nil || isRedisError(err) {
			node.Client.himportAfterBatch(cn, injected, cmds)
			// SETs of registered fieldsets that lost their session state
			// re-queue for the next attempt, which re-prepares lazily —
			// the cluster equivalent of himportRetryFailedSets, bounded by
			// the pipeline's attempt budget. A non-nil redis error here
			// means pipelineReadCmds already re-queued the whole batch
			// (retryable first-command error); adding the SETs again would
			// duplicate them in the next attempt.
			if err == nil {
				c.himportRequeueFailedSets(ctx, cmds, failedCmds)
			}
		}
		return err
	})
}

func (c *ClusterClient) pipelineReadCmds(
	ctx context.Context,
	node *clusterNode,
	cn *pool.Conn,
	rd *proto.Reader,
	cmds []Cmder,
	failedCmds *cmdsMap,
) error {
	for i, cmd := range cmds {
		// Drain any buffered RESP3 push notifications before reading each
		// reply — otherwise a push frame (e.g. a maintnotifications MOVING
		// notification) is consumed AS the command's reply and every
		// subsequent reply in the pipeline shifts by one command. The
		// standalone pipeline and the cluster TxPipeline read loops already
		// do this; this loop was the only push-blind reader, and the
		// autopipeliner routes all cluster traffic through it.
		if err := node.Client.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
			internal.Logger.Printf(ctx, "push: error processing pending notifications before reading reply: %v", err)
		}
		err := cmd.readReply(rd)
		cmd.SetErr(err)

		if err == nil {
			continue
		}

		if c.checkMovedErr(ctx, cmd, err, failedCmds) {
			continue
		}

		if c.opt.ReadOnly && isBadConn(err, false, node.Client.getAddr()) {
			node.MarkAsFailing()
		}

		if !isRedisError(err) {
			if shouldRetry(err, true) && !cmdsContainNoRetry(cmds) {
				_ = c.mapCmdsByNode(ctx, failedCmds, cmds)
			}
			setCmdsErr(cmds[i+1:], err)
			return err
		}
	}

	// rawErr: execution path; never await an async command's batch here.
	if err := cmds[0].rawErr(); err != nil && shouldRetry(err, true) && !cmdsContainNoRetry(cmds) {
		_ = c.mapCmdsByNode(ctx, failedCmds, cmds)
		return err
	}

	return nil
}

func (c *ClusterClient) checkMovedErr(
	ctx context.Context, cmd Cmder, err error, failedCmds *cmdsMap,
) bool {
	moved, ask, addr := isMovedError(err)
	if !moved && !ask {
		return false
	}

	node, err := c.nodes.GetOrCreate(addr)
	if err != nil {
		return false
	}

	if moved {
		c.state.LazyReload()
		failedCmds.Add(node, cmd)
		return true
	}

	if ask {
		failedCmds.Add(node, NewCmd(ctx, "asking"), cmd)
		return true
	}

	panic("not reached")
}

// TxPipeline acts like Pipeline, but wraps queued commands with MULTI/EXEC.
func (c *ClusterClient) TxPipeline() Pipeliner {
	pipe := Pipeline{
		exec: func(ctx context.Context, cmds []Cmder) error {
			cmds = wrapMultiExec(ctx, cmds)
			return c.processTxPipelineHook(ctx, cmds)
		},
	}
	pipe.init()
	return &pipe
}

func (c *ClusterClient) TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.TxPipeline().Pipelined(ctx, fn)
}

// A cluster tx pipeline sends MULTI, c1..cN, EXEC — N+2 commands, or N+3 with a
// leading ASKING — and always receives exactly that many replies, so every
// redirect/abort path leaves the connection clean.
//
// Possible reply sequences:
//	1. Slot owned here, no migration:
//	     +OK,  +QUEUED x N,  *N (array of N results)   -> success
//	2. Slot already migrated away:
//	     +OK,  -MOVED x N,  -EXECABORT                 -> re-route whole tx
//	3. Slot in migrating state (still owned here, keys draining out). Per
//	   cmd, the queue reply is +QUEUED / -ASK / -TRYAGAIN (keys present /
//	   all gone / some gone); any -ASK or -TRYAGAIN dirties the tx, so
//	   EXEC is -EXECABORT. Still N+2 replies, like the cases above:
//	     +OK,  (+QUEUED|-ASK|-TRYAGAIN) x N,  -EXECABORT  -> follow first redirect
//	4. Narrow race (all +QUEUED, slot moves before EXEC):
//	     +OK,  +QUEUED x N,  -MOVED <slot> <addr>         -> re-route whole tx
//	5. Non-cluster command error (arity / ACL / unknown):
//	     +OK,  +QUEUED..., -ERR...,  -EXECABORT           -> surface, not retryable
//	6. Narrow race (all +QUEUED, slot still migrating, keys drain before EXEC):
//	     +OK,  +QUEUED x N,  -ASK / -TRYAGAIN             -> re-route on -ASK, back off on -TRYAGAIN
//
// EXEC reply — the reply that decides the outcome:
//
//	*N                    success; read N per-command results
//	-EXECABORT            a queue-stage command failed; follow the first queue
//	                      redirect (MOVED/ASK/TRYAGAIN), else surface the trigger
//	-MOVED <slot> <addr>  case 4; re-route whole tx to addr, reload topology
//	-ASK <slot> <addr>    race: slot entered migrating state; re-route to addr
//	                      with a top-level ASKING before MULTI
//	-TRYAGAIN             race: migrating with split keys, or slot being trimmed
//	                      (CLUSTER_REDIR_TRIMMING on a write); back off and retry
//	                      the whole tx (same node still owns it)
//	-CLUSTERDOWN          cluster degraded; back off and retry whole tx
//
// ASK retry: the ASKING flag is NOT cleared between commands inside a MULTI
// so one top-level ASKING before MULTI covers the whole tx and lets the importing
// slot serve at EXEC. ASKING placed inside the MULTI would be queued and leave
// the flag unset during queueing, so the keyed commands would still get MOVED.
//
// Out of scope: WATCH's null-array EXEC and -CROSSSLOT;
// cluster TxPipeline is not used with WATCH and cross-slot is rejected client-side.

type txOutcomeKind int

const (
	txSuccess       txOutcomeKind = iota // transaction executed; per-command results are set
	txRetryMoved                         // MOVED: reload topology and re-route the whole tx
	txRetryAsk                           // ASK: re-route to the target with a top-level ASKING
	txRetryTryAgain                      // TRYAGAIN: back off and re-route the whole tx
	txRetryConn                          // connection/write/read failure: re-route the whole tx
	txFatal                              // non-retryable error; surface to the caller
)

// txOutcome is the result of a single tx attempt. err is the error to report
// when the redirect/retry loop is exhausted (or the fatal error to surface);
// addr is the ASK target; execErr is the EXEC reply error used to mark
// aborted commands; unreadReplies forces the connection to be discarded
// when the read loop exited before consuming all N+2 replies, leaving bytes
// on the wire.
type txOutcome struct {
	kind          txOutcomeKind
	err           error
	addr          string
	execErr       error
	unreadReplies bool
}

// txRedirect records the first queue-stage redirect (MOVED/ASK/TRYAGAIN) seen
// while reading +QUEUED replies. Redis dirties and aborts the transaction on
// any such reply, so the EXEC reply will be EXECABORT and the client must
// follow the recorded redirect with the whole transaction.
type txRedirect struct {
	moved    bool
	ask      bool
	tryAgain bool
	addr     string
	err      error
}

// errTxDirtyConn forces releaseConn to discard a connection that may still have
// unread transaction replies on it (an early exit before consuming all N+2).
var errTxDirtyConn = errors.New("redis: connection has unread transaction replies")

func (c *ClusterClient) processTxPipeline(ctx context.Context, cmds []Cmder) (retErr error) {
	var operationStart time.Time
	pipelineOpDurationCallback := otel.GetPipelineOperationDurationCallback()
	if pipelineOpDurationCallback != nil {
		operationStart = time.Now()
	}
	totalAttempts := 0
	var lastErr error

	defer func() {
		if pipelineOpDurationCallback == nil {
			return
		}
		finalErr := cmp.Or(retErr, cmdsFirstErr(cmds), lastErr)
		pipelineOpDurationCallback(ctx, time.Since(operationStart), "MULTI", len(cmds), totalAttempts, finalErr, nil, 0)
	}()

	// Trim multi .. exec.
	cmds = cmds[1 : len(cmds)-1]
	if len(cmds) == 0 {
		return nil
	}

	state, err := c.state.Get(ctx)
	if err != nil {
		setCmdsErr(cmds, err)
		return err
	}

	keyedCmdsBySlot := c.slottedKeyedCommands(ctx, cmds)
	slot := -1
	switch len(keyedCmdsBySlot) {
	case 0:
		slot = hashtag.RandomSlot()
	case 1:
		for sl := range keyedCmdsBySlot {
			slot = sl
		}
	default:
		// TxPipeline does not support cross slot transaction.
		setCmdsErr(cmds, ErrCrossSlot)
		return ErrCrossSlot
	}

	node, err := state.slotMasterNode(slot)
	if err != nil {
		setCmdsErr(cmds, err)
		return err
	}

	asking := false
	// MOVED/ASK are routing changes, not transient failures: follow them immediately.
	redirected := false
	for attempt := 0; attempt <= c.opt.MaxRedirects; attempt++ {
		totalAttempts++
		if attempt > 0 && !redirected {
			if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
				setCmdsErr(cmds, err)
				return err
			}
		}

		outcome := c.processTxPipelineNode(ctx, node, cmds, asking)
		lastErr = outcome.err
		redirected = false
		switch outcome.kind {
		case txSuccess:
			return cmdsFirstErr(cmds)
		case txRetryMoved:
			// Route directly to the authoritative addr from the MOVED; the
			// cached slot state may be stale until LazyReload lands.
			redirected = true
			asking = false
			c.state.LazyReload()
			if node, err = c.nodes.GetOrCreate(outcome.addr); err != nil {
				setCmdsErr(cmds, err)
				return err
			}
		case txRetryAsk:
			redirected = true
			asking = true
			if node, err = c.nodes.GetOrCreate(outcome.addr); err != nil {
				setCmdsErr(cmds, err)
				return err
			}
		case txRetryTryAgain, txRetryConn:
			// Same node, fresh connection: TRYAGAIN comes from the migrating
			// source (still the owner), and a conn failure only needs a new
			// connection. Preserve a prior ASKING flag: if we followed an ASK
			// to the importing target, the retry must still send ASKING (the
			// slot is still importing). ASKING is harmless if the migration
			// has since completed, since the flag is only consulted for
			// importing slots.
		case txFatal:
			// Mark every queued-but-never-executed command with the abort
			// error; the command that triggered EXECABORT already has its
			// own error and keeps it, so callers can tell what went wrong.
			abortErr := cmp.Or(outcome.execErr, outcome.err)
			for _, cmd := range cmds {
				if cmd.Err() == nil {
					cmd.SetErr(abortErr)
				}
			}
			return lastErr
		}
	}

	if lastErr != nil {
		setCmdsErr(cmds, lastErr)
	}
	return cmdsFirstErr(cmds)
}

// slottedKeyedCommands returns a map of slot to commands taking into account
// only commands that have keys.
func (c *ClusterClient) slottedKeyedCommands(_ context.Context, cmds []Cmder) map[int][]Cmder {
	cmdsSlots := map[int][]Cmder{}

	// Peek once outside the loop, one RLock for the whole batch instead of
	// two per command (one for the keyless check, one inside cmdSlot).
	cachedInfo := c.cmdsInfoCache.Peek()

	prefferedRandomSlot := -1
	for _, cmd := range cmds {
		var info *CommandInfo
		if cachedInfo != nil {
			info = cachedInfo[cmd.Name()]
		}

		pos := cmdFirstKeyPosWithInfo(cmd, info)
		if pos == 0 {
			continue
		}

		slot := c.cmdSlotWithPos(cmd, pos, prefferedRandomSlot)
		if prefferedRandomSlot == -1 {
			prefferedRandomSlot = slot
		}

		cmdsSlots[slot] = append(cmdsSlots[slot], cmd)
	}

	return cmdsSlots
}

func (c *ClusterClient) processTxPipelineNode(
	ctx context.Context, node *clusterNode, cmds []Cmder, asking bool,
) *txOutcome {
	wire := wrapMultiExec(ctx, cmds)
	if asking {
		// ASKING must precede MULTI so the flag stays set for the whole tx.
		wire = append([]Cmder{NewCmd(ctx, "asking")}, wire...)
	}

	var outcome *txOutcome
	// executed guards against a node-level hook short-circuiting (returning
	// without calling next) — same treatment as processPipelineNode.
	executed := false
	chainErr := node.Client.withProcessPipelineHook(ctx, wire, func(ctx context.Context, wire []Cmder) error {
		executed = true
		// Acquire through the node's dedicated pipeline pool when configured
		// (same routing as processPipelineNode); withPipelineConn falls back
		// to the main pool otherwise. The inner fn's return value drives the
		// connection release exactly like the explicit releaseConn did:
		// redis errors keep the conn poolable, unread replies poison it.
		entered := false
		err := node.Client.withPipelineConn(ctx, func(ctx context.Context, cn *pool.Conn) error {
			entered = true
			outcome = c.processTxPipelineNodeConn(ctx, node, cn, wire, cmds, asking)
			connErr := outcome.err
			if isRedisError(outcome.err) {
				connErr = nil
			}
			if outcome.unreadReplies {
				connErr = errTxDirtyConn
			}
			return connErr
		})
		if !entered && err != nil {
			// Connection acquisition failed — fn never ran.
			if shouldRetry(err, true) && !cmdsContainNoRetry(cmds) {
				outcome = &txOutcome{kind: txRetryConn, err: err}
			} else {
				outcome = &txOutcome{kind: txFatal, err: err}
			}
		}
		return err
	})

	if !executed && chainErr != nil {
		// A node-level hook aborted with an error: surface its verdict. A hook
		// that returned nil short-circuited successfully (it served the batch),
		// which is legal for plain pipelines too, so it is not turned into a
		// fatal outcome (review finding by codex on #3942).
		outcome = &txOutcome{kind: txFatal, err: chainErr}
	}
	if outcome == nil {
		outcome = &txOutcome{kind: txFatal, err: fmt.Errorf("redis: tx pipeline produced no outcome")}
	}
	return outcome
}

func (c *ClusterClient) processTxPipelineNodeConn(
	ctx context.Context, node *clusterNode, cn *pool.Conn, wire []Cmder, cmds []Cmder, asking bool,
) *txOutcome {
	// HIMPORT bookkeeping: pending discards and PREPAREs for registered
	// fieldsets the transaction references get written ahead of the wire
	// batch (before ASKING/MULTI; the session state is visible at EXEC).
	injected := node.Client.himportInjectedCmds(ctx, cn, cmds)

	if err := cn.WithWriter(c.context(ctx), c.opt.WriteTimeout, func(wr *proto.Writer) error {
		for _, ic := range injected {
			if err := writeCmd(wr, ic); err != nil {
				return err
			}
		}
		return writeCmds(wr, wire)
	}); err != nil {
		// Write failure: re-route the whole tx on a fresh connection.
		if shouldRetry(err, true) && !cmdsContainNoRetry(cmds) {
			return &txOutcome{kind: txRetryConn, err: err}
		}
		return &txOutcome{kind: txFatal, err: err}
	}

	var outcome *txOutcome
	readErr := cn.WithReader(c.context(ctx), c.opt.ReadTimeout, func(rd *proto.Reader) error {
		if err := node.Client.himportReadInjectedReplies(ctx, cn, rd, injected); err != nil {
			// Transport error with the tx replies unread; the batch was
			// written and may have committed — fatal, discard the conn.
			outcome = c.txReadFatal(err)
			return nil
		}
		outcome = c.readTxPipelineReplies(ctx, node, cn, rd, cmds, asking)
		if outcome != nil && outcome.kind == txSuccess {
			node.Client.himportAfterBatch(cn, injected, cmds)
		}
		return nil
	})

	if readErr != nil {
		// Reader-level failure (deadline setup, nil conn) around the read loop.
		// The batch was already written, so the server may have committed;
		// surface the error as fatal and discard the suspect connection rather
		// than re-executing the transaction.
		return c.txReadFatal(readErr)
	}
	return outcome
}

// readTxPipelineReplies reads the replies of one MULTI..EXEC unit and
// classifies the outcome. The reply count always matches the number of sent
// commands, so success/redirect paths leave the connection clean; only an early
// MULTI read failure can leave unread replies.
func (c *ClusterClient) readTxPipelineReplies(
	ctx context.Context, node *clusterNode, cn *pool.Conn, rd *proto.Reader, cmds []Cmder, asking bool,
) *txOutcome {
	scratch := NewStatusCmd(ctx)

	readStatus := func() error {
		c.txProcessPush(ctx, node, cn, rd)
		return scratch.readReply(rd)
	}

	// Optional top-level ASKING reply (+OK, or a retryable error such as -LOADING).
	if asking {
		if err := readStatus(); err != nil {
			return c.txPreQueueErrorOutcome(err, cmds)
		}
	}

	// MULTI reply (+OK, or an error such as -LOADING during failover).
	if err := readStatus(); err != nil {
		return c.txPreQueueErrorOutcome(err, cmds)
	}

	// Queue replies: +QUEUED, or a redirect / command error that dirties the tx.
	var firstRedirect *txRedirect
	var firstFatal error
	for _, cmd := range cmds {
		err := readStatus()
		if err == nil {
			continue // +QUEUED
		}
		if !isRedisError(err) {
			return c.txReadFatal(err) // IO error
		}
		if moved, ask, addr := isMovedError(err); moved || ask {
			if firstRedirect == nil {
				firstRedirect = &txRedirect{moved: moved, ask: ask, addr: addr, err: err}
			}
			continue
		}
		if proto.IsTryAgainError(err) {
			if firstRedirect == nil {
				firstRedirect = &txRedirect{tryAgain: true, err: err}
			}
			continue
		}
		// Non-redirect command error (e.g. wrong arity) dirties the tx.
		cmd.SetErr(err)
		if firstFatal == nil {
			firstFatal = err
		}
	}

	// EXEC reply. ReadLine parses error lines into typed errors, so a non-nil
	// err means EXEC returned an error rather than the result array.
	c.txProcessPush(ctx, node, cn, rd)
	line, err := rd.ReadLine()
	if err != nil {
		if !isRedisError(err) {
			return c.txReadFatal(err) // IO error
		}
		return c.classifyExecError(err, firstRedirect, firstFatal)
	}

	if line[0] != proto.RespArray {
		err := fmt.Errorf("redis: unexpected EXEC reply %q", line)
		setCmdsErr(cmds, err)
		// A non-array aggregate reply may carry an unread payload.
		return &txOutcome{kind: txFatal, err: err, unreadReplies: true}
	}

	// Success: read the N command results.
	if err := node.Client.pipelineReadCmds(ctx, cn, rd, cmds); err != nil && !isRedisError(err) {
		return c.txReadFatal(err) // IO error mid-results
	}
	return &txOutcome{kind: txSuccess}
}

func (c *ClusterClient) txProcessPush(ctx context.Context, node *clusterNode, cn *pool.Conn, rd *proto.Reader) {
	if err := node.Client.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
		internal.Logger.Printf(ctx, "push: error processing pending notifications before reading reply: %v", err)
	}
}

// txReadFatal classifies a read-phase IO error. The MULTI..EXEC batch was
// already written, so the server may have committed the transaction; retrying
// would re-execute it, double-applying non-idempotent commands (INCR/APPEND,
// which are not NoRetry). Surface the error as fatal and discard the
// connection, since replies may still be unread on the wire.
func (c *ClusterClient) txReadFatal(err error) *txOutcome {
	return &txOutcome{kind: txFatal, err: err, unreadReplies: true}
}

// txPreQueueErrorOutcome classifies a setup-phase reply error: the top-level
// ASKING reply or the MULTI reply. The transaction body never executes (EXEC
// returns -EXECABORT), so retryable errors such as -LOADING are safe to retry
// on a fresh connection. A failed setup reply still leaves the remaining
// replies on the wire -- the server replies to each following command and to
// EXEC regardless -- so the connection is always discarded.
func (c *ClusterClient) txPreQueueErrorOutcome(err error, cmds []Cmder) *txOutcome {
	if !isRedisError(err) {
		return c.txReadFatal(err)
	}
	if shouldRetry(err, true) && !cmdsContainNoRetry(cmds) {
		return &txOutcome{kind: txRetryConn, err: err, unreadReplies: true}
	}
	return &txOutcome{kind: txFatal, err: err, unreadReplies: true}
}

// classifyExecError turns an EXEC reply error into a retry/fatal outcome.
func (c *ClusterClient) classifyExecError(execErr error, firstRedirect *txRedirect, firstFatal error) *txOutcome {
	if moved, ask, addr := isMovedError(execErr); moved || ask {
		// Narrow race: the slot moved after every command was queued.
		if ask {
			return &txOutcome{kind: txRetryAsk, err: execErr, addr: addr}
		}
		return &txOutcome{kind: txRetryMoved, err: execErr, addr: addr}
	}
	if proto.IsTryAgainError(execErr) {
		return &txOutcome{kind: txRetryTryAgain, err: execErr}
	}
	if proto.IsClusterDownError(execErr) {
		// Cluster degraded: back off and retry. Replies were fully consumed.
		return &txOutcome{kind: txRetryConn, err: execErr}
	}
	if proto.IsExecAbortError(execErr) {
		if firstFatal != nil {
			return &txOutcome{kind: txFatal, err: firstFatal, execErr: execErr}
		}
		if firstRedirect != nil {
			switch {
			case firstRedirect.moved:
				return &txOutcome{kind: txRetryMoved, err: firstRedirect.err, addr: firstRedirect.addr}
			case firstRedirect.ask:
				return &txOutcome{kind: txRetryAsk, err: firstRedirect.err, addr: firstRedirect.addr}
			case firstRedirect.tryAgain:
				return &txOutcome{kind: txRetryTryAgain, err: firstRedirect.err}
			}
		}
		return &txOutcome{kind: txFatal, err: execErr, execErr: execErr}
	}
	return &txOutcome{kind: txFatal, err: execErr}
}

func (c *ClusterClient) Watch(ctx context.Context, fn func(*Tx) error, keys ...string) error {
	if len(keys) == 0 {
		return errNoWatchKeys
	}

	slot := hashtag.Slot(keys[0])
	for _, key := range keys[1:] {
		if hashtag.Slot(key) != slot {
			return errWatchCrosslot
		}
	}

	node, err := c.slotMasterNode(ctx, slot)
	if err != nil {
		return err
	}

	for attempt := 0; attempt <= c.opt.MaxRedirects; attempt++ {
		if attempt > 0 {
			if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
				return err
			}
		}

		// Track callback errors separately to avoid retrying user failures through cluster retry classification.
		var fnErr error
		err = node.Client.Watch(ctx, func(tx *Tx) error {
			fnErr = fn(tx)
			return fnErr
		}, keys...)
		if err == nil {
			break
		}
		if fnErr != nil {
			return fnErr
		}

		moved, ask, addr := isMovedError(err)
		if moved || ask {
			node, err = c.nodes.GetOrCreate(addr)
			if err != nil {
				return err
			}
			continue
		}

		if isReadOnly := isReadOnlyError(err); isReadOnly || err == pool.ErrClosed {
			if isReadOnly {
				c.state.LazyReload()
			}
			node, err = c.slotMasterNode(ctx, slot)
			if err != nil {
				return err
			}
			continue
		}

		if shouldRetry(err, true) {
			continue
		}

		return err
	}

	return err
}

// maintenance notifications won't work here for now
func (c *ClusterClient) pubSub() *PubSub {
	var node *clusterNode
	pubsub := &PubSub{
		opt: c.opt.clientOptions(),
		newConn: func(ctx context.Context, addr string, channels []string) (*pool.Conn, error) {
			if node != nil {
				panic("node != nil")
			}

			var err error

			if len(channels) > 0 {
				slot := hashtag.Slot(channels[0])

				// newConn in PubSub is only used for subscription connections, so it is safe to
				// assume that a slave node can always be used when client options specify ReadOnly.
				if c.opt.ReadOnly {
					state, err := c.state.Get(ctx)
					if err != nil {
						return nil, err
					}

					node, err = c.slotReadOnlyNode(state, slot)
					if err != nil {
						return nil, err
					}
				} else {
					node, err = c.slotMasterNode(ctx, slot)
					if err != nil {
						return nil, err
					}
				}
			} else {
				node, err = c.nodes.Random()
				if err != nil {
					return nil, err
				}
			}
			cn, err := node.Client.pubSubPool.NewConn(ctx, node.Client.opt.Network, node.Client.opt.Addr, channels)
			if err != nil {
				node = nil
				return nil, err
			}
			// will return nil if already initialized
			err = node.Client.initConn(ctx, cn)
			if err != nil {
				_ = cn.Close()
				node = nil
				return nil, err
			}
			node.Client.pubSubPool.TrackConn(cn)
			return cn, nil
		},
		closeConn: func(cn *pool.Conn) error {
			// Untrack connection from PubSubPool
			node.Client.pubSubPool.UntrackConn(cn)
			err := cn.Close()
			node = nil
			return err
		},
	}
	pubsub.init()

	return pubsub
}

// Subscribe subscribes the client to the specified channels.
// Channels can be omitted to create empty subscription.
func (c *ClusterClient) Subscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.Subscribe(ctx, channels...)
	}
	return pubsub
}

// PSubscribe subscribes the client to the given patterns.
// Patterns can be omitted to create empty subscription.
func (c *ClusterClient) PSubscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.PSubscribe(ctx, channels...)
	}
	return pubsub
}

// SSubscribe Subscribes the client to the specified shard channels.
func (c *ClusterClient) SSubscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.SSubscribe(ctx, channels...)
	}
	return pubsub
}

func (c *ClusterClient) retryBackoff(attempt int) time.Duration {
	return internal.RetryBackoff(attempt, c.opt.MinRetryBackoff, c.opt.MaxRetryBackoff)
}

func (c *ClusterClient) cmdsInfo(ctx context.Context) (map[string]*CommandInfo, error) {
	// Try 3 random nodes.
	const nodeLimit = 3

	addrs, err := c.nodes.Addrs()
	if err != nil {
		return nil, err
	}

	var firstErr error

	perm := rand.Perm(len(addrs))
	if len(perm) > nodeLimit {
		perm = perm[:nodeLimit]
	}

	for _, idx := range perm {
		addr := addrs[idx]
		node, err := c.nodes.GetOrCreate(addr)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}

		info, err := node.Client.Command(ctx).Result()
		if err == nil {
			return info, nil
		}

		if firstErr == nil {
			firstErr = err
		}
	}

	if firstErr == nil {
		panic("not reached")
	}
	return nil, firstErr
}

// cmdInfo will fetch and cache the command policies after the first execution
func (c *ClusterClient) cmdInfo(ctx context.Context, name string) *CommandInfo {
	// Use a separate context that won't be canceled to ensure command info lookup
	// doesn't fail due to original context cancellation
	cmdInfoCtx := c.context(ctx)
	if c.opt.ContextTimeoutEnabled && ctx != nil {
		// If context timeout is enabled, still use a reasonable timeout
		var cancel context.CancelFunc
		cmdInfoCtx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
	}

	cmdsInfo, err := c.cmdsInfoCache.Get(cmdInfoCtx)
	if err != nil {
		internal.Logger.Printf(cmdInfoCtx, "getting command info: %s", err)
		return nil
	}

	info := cmdsInfo[name]
	if info == nil {
		internal.Logger.Printf(cmdInfoCtx, "info for cmd=%s not found", name)
	}

	return info
}

// cmdInfoPeek returns the cached CommandInfo for the named command without
// triggering a round-trip to Redis. It returns nil when the cache is cold.
func (c *ClusterClient) cmdInfoPeek(name string) *CommandInfo {
	if cmds := c.cmdsInfoCache.Peek(); cmds != nil {
		return cmds[name]
	}
	return nil
}

func (c *ClusterClient) cmdSlot(cmd Cmder, prefferedSlot int) int {
	// Serve/populate the per-command slot cache only on the natural-slot path
	// (prefferedSlot == -1). A forced prefferedSlot (retry re-routing) must not be
	// cached or served from cache. The cache lets the autopipeline shard router
	// and the pipeline-flush router (mapCmdsByNode) share one slot computation
	// instead of each recomputing it.
	if prefferedSlot == -1 {
		if slot, ok := cmd.cachedSlot(); ok {
			return slot
		}
	}
	info := c.cmdInfoPeek(cmd.Name())
	slot := c.cmdSlotWithPos(cmd, cmdFirstKeyPosWithInfo(cmd, info), prefferedSlot)
	if prefferedSlot == -1 && slot >= 0 {
		cmd.setCachedSlot(slot)
	}
	return slot
}

// cmdSlotWithPos computes the cluster slot for cmd given a pre-resolved first key
// position. Separating pos resolution from slot computation lets callers that
// already know pos avoid a redundant Peek() call.
func (c *ClusterClient) cmdSlotWithPos(cmd Cmder, pos int, prefferedSlot int) int {
	args := cmd.Args()
	if args[0] == "cluster" && (args[1] == "getkeysinslot" || args[1] == "countkeysinslot") {
		return args[2].(int)
	}
	return cmdSlot(cmd, pos, prefferedSlot)
}

func cmdSlot(cmd Cmder, pos int, prefferedRandomSlot int) int {
	if pos == 0 {
		if prefferedRandomSlot != -1 {
			return prefferedRandomSlot
		}
		// Return -1 for keyless commands to signal that ShardPicker should be used
		return -1
	}
	firstKey := cmd.stringArg(pos)
	return hashtag.Slot(firstKey)
}

func (c *ClusterClient) cmdNode(
	ctx context.Context,
	cmdName string,
	slot int,
) (*clusterNode, error) {
	state, err := c.state.Get(ctx)
	if err != nil {
		return nil, err
	}

	if c.opt.ReadOnly {
		cmdInfo := c.cmdInfo(ctx, cmdName)
		if cmdInfo != nil && cmdInfo.ReadOnly {
			return c.slotReadOnlyNode(state, slot)
		}
	}
	return state.slotMasterNode(slot)
}

func (c *ClusterClient) cmdNodeWithShardPicker(
	ctx context.Context,
	cmdName string,
	slot int,
	shardPicker routing.ShardPicker,
) (*clusterNode, error) {
	state, err := c.state.Get(ctx)
	if err != nil {
		return nil, err
	}

	// For keyless commands (slot == -1), use ShardPicker to select a shard
	// This respects the user's configured ShardPicker policy
	if slot == -1 {
		if len(state.Masters) == 0 {
			return nil, errClusterNoNodes
		}
		idx := shardPicker.Next(len(state.Masters))
		return state.Masters[idx], nil
	}

	if c.opt.ReadOnly {
		cmdInfo := c.cmdInfo(ctx, cmdName)
		if cmdInfo != nil && cmdInfo.ReadOnly {
			return c.slotReadOnlyNode(state, slot)
		}
	}
	return state.slotMasterNode(slot)
}

func (c *ClusterClient) slotReadOnlyNode(state *clusterState, slot int) (*clusterNode, error) {
	if c.opt.RouteByLatency {
		return state.slotClosestNode(slot)
	}
	if c.opt.RouteRandomly {
		return state.slotRandomNode(slot)
	}

	if c.opt.ShardPicker != nil {
		return state.slotShardPickerSlaveNode(slot, c.opt.ShardPicker)
	}

	return state.slotSlaveNode(slot)
}

func (c *ClusterClient) slotMasterNode(ctx context.Context, slot int) (*clusterNode, error) {
	state, err := c.state.Get(ctx)
	if err != nil {
		return nil, err
	}
	return state.slotMasterNode(slot)
}

// SlaveForKey gets a client for a replica node to run any command on it.
// This is especially useful if we want to run a particular lua script which has
// only read only commands on the replica.
// This is because other redis commands generally have a flag that points that
// they are read only and automatically run on the replica nodes
// if ClusterOptions.ReadOnly flag is set to true.
func (c *ClusterClient) SlaveForKey(ctx context.Context, key string) (*Client, error) {
	state, err := c.state.Get(ctx)
	if err != nil {
		return nil, err
	}
	slot := hashtag.Slot(key)
	node, err := c.slotReadOnlyNode(state, slot)
	if err != nil {
		return nil, err
	}
	return node.Client, err
}

// MasterForKey return a client to the master node for a particular key.
func (c *ClusterClient) MasterForKey(ctx context.Context, key string) (*Client, error) {
	slot := hashtag.Slot(key)
	node, err := c.slotMasterNode(ctx, slot)
	if err != nil {
		return nil, err
	}
	return node.Client, nil
}

func (c *ClusterClient) context(ctx context.Context) context.Context {
	if c.opt.ContextTimeoutEnabled {
		return ctx
	}
	return context.Background()
}

func (c *ClusterClient) GetResolver() *commandInfoResolver {
	return c.cmdInfoResolver
}

func (c *ClusterClient) SetCommandInfoResolver(cmdInfoResolver *commandInfoResolver) {
	c.cmdInfoResolver = cmdInfoResolver
}

// extractCommandInfo retrieves the routing policy for a command
func (c *ClusterClient) extractCommandInfo(ctx context.Context, cmd Cmder) *routing.CommandPolicy {
	if cmdInfo := c.cmdInfo(ctx, cmd.Name()); cmdInfo != nil && cmdInfo.CommandPolicy != nil {
		return cmdInfo.CommandPolicy
	}

	return nil
}

// NewDynamicResolver returns a CommandInfoResolver
// that uses the underlying cmdInfo cache to resolve the policies
func (c *ClusterClient) NewDynamicResolver() *commandInfoResolver {
	return &commandInfoResolver{
		resolveFunc: c.extractCommandInfo,
	}
}

func appendIfNotExist[T comparable](vals []T, newVal T) []T {
	if slices.Contains(vals, newVal) {
		return vals
	}
	return append(vals, newVal)
}

//------------------------------------------------------------------------------

type cmdsMap struct {
	mu sync.Mutex
	m  map[*clusterNode][]Cmder
}

func newCmdsMap() *cmdsMap {
	return &cmdsMap{
		m: make(map[*clusterNode][]Cmder),
	}
}

func (m *cmdsMap) Add(node *clusterNode, cmds ...Cmder) {
	m.mu.Lock()
	m.m[node] = append(m.m[node], cmds...)
	m.mu.Unlock()
}
