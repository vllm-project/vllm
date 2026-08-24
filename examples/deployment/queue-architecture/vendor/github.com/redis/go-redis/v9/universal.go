package redis

import (
	"context"
	"crypto/tls"
	"net"
	"time"

	"github.com/redis/go-redis/v9/auth"
	"github.com/redis/go-redis/v9/maintnotifications"
	"github.com/redis/go-redis/v9/push"
)

// UniversalOptions information is required by UniversalClient to establish
// connections.
type UniversalOptions struct {
	// Either a single address or a seed list of host:port addresses
	// of cluster/sentinel nodes.
	Addrs []string

	// ClientName will execute the `CLIENT SETNAME ClientName` command for each conn.
	ClientName string

	// Database to be selected after connecting to the server.
	// Only single-node and failover clients.
	DB int

	// Common options.

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

	SentinelUsername string
	SentinelPassword string

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

	ReadTimeout           time.Duration
	WriteTimeout          time.Duration
	ContextTimeoutEnabled bool

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

	// PoolFIFO uses FIFO mode for each node connection pool GET/PUT (default LIFO).
	PoolFIFO bool

	PoolSize int

	// MaxConcurrentDials is the maximum number of concurrent connection creation goroutines.
	// If <= 0, defaults to PoolSize. If > PoolSize, it will be capped at PoolSize.
	MaxConcurrentDials int

	PoolTimeout           time.Duration
	MinIdleConns          int
	MaxIdleConns          int
	MaxActiveConns        int
	ConnMaxIdleTime       time.Duration
	ConnMaxLifetime       time.Duration
	ConnMaxLifetimeJitter time.Duration

	TLSConfig *tls.Config

	// Only cluster clients.

	MaxRedirects   int
	ReadOnly       bool
	RouteByLatency bool
	RouteRandomly  bool

	// MasterName is the sentinel master name.
	// Only for failover clients.
	MasterName string

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

	IdentitySuffix string

	// FailingTimeoutSeconds is the timeout in seconds for marking a cluster node as failing.
	// When a node is marked as failing, it will be avoided for this duration.
	// Only applies to cluster clients. Default is 15 seconds.
	FailingTimeoutSeconds int

	// Deprecated: All RediSearch commands now have stable RESP3 parsing and this
	// flag is a no-op. It is kept for backwards compatibility and will be removed
	// in a future release.
	UnstableResp3 bool

	// PushNotificationProcessor is the processor for handling push notifications.
	// If nil, a default processor will be created for RESP3 connections.
	PushNotificationProcessor push.NotificationProcessor

	// IsClusterMode can be used when only one Addrs is provided (e.g. Elasticache supports setting up cluster mode with configuration endpoint).
	IsClusterMode bool

	// AutoPipelineOptions is the default config for the client's
	// autopipeliner faces (AutoPipeline / AsyncAutoPipeline), applied when
	// they are called without explicit options. See Options.AutoPipelineOptions.
	AutoPipelineOptions *AutoPipelineOptions

	// MaintNotificationsConfig provides configuration for maintnotifications upgrades.
	MaintNotificationsConfig *maintnotifications.Config

	// ClientSideCacheConfig enables client-side caching when NewUniversalClient
	// selects a standalone Client. See Options.ClientSideCacheConfig.
	//
	// Experimental: this API may change in a minor release.
	ClientSideCacheConfig *ClientSideCacheConfig

	// ClientSideCache supplies an explicit cache when NewUniversalClient selects
	// a standalone Client. See Options.ClientSideCache.
	//
	// Experimental: this API may change in a minor release.
	ClientSideCache Cache

	// ClientSideCacheStrategy selects the standalone client's invalidation
	// strategy. See Options.ClientSideCacheStrategy.
	//
	// Experimental: this API may change in a minor release.
	ClientSideCacheStrategy CSCStrategy
}

// Cluster returns cluster options created from the universal options.
func (o *UniversalOptions) Cluster() *ClusterOptions {
	if len(o.Addrs) == 0 {
		o.Addrs = []string{"127.0.0.1:6379"}
	}

	return &ClusterOptions{
		Addrs:      o.Addrs,
		ClientName: o.ClientName,
		Dialer:     o.Dialer,
		OnConnect:  o.OnConnect,

		Protocol:                     o.Protocol,
		Username:                     o.Username,
		Password:                     o.Password,
		CredentialsProvider:          o.CredentialsProvider,
		CredentialsProviderContext:   o.CredentialsProviderContext,
		StreamingCredentialsProvider: o.StreamingCredentialsProvider,

		MaxRedirects:   o.MaxRedirects,
		ReadOnly:       o.ReadOnly,
		RouteByLatency: o.RouteByLatency,
		RouteRandomly:  o.RouteRandomly,

		MaxRetries:      o.MaxRetries,
		MinRetryBackoff: o.MinRetryBackoff,
		MaxRetryBackoff: o.MaxRetryBackoff,

		DialTimeout:        o.DialTimeout,
		DialerRetries:      o.DialerRetries,
		DialerRetryTimeout: o.DialerRetryTimeout,
		ReadTimeout:        o.ReadTimeout,
		WriteTimeout:       o.WriteTimeout,

		ContextTimeoutEnabled: o.ContextTimeoutEnabled,

		ReadBufferSize:  o.ReadBufferSize,
		WriteBufferSize: o.WriteBufferSize,

		PoolFIFO:              o.PoolFIFO,
		PoolSize:              o.PoolSize,
		MaxConcurrentDials:    o.MaxConcurrentDials,
		PoolTimeout:           o.PoolTimeout,
		MinIdleConns:          o.MinIdleConns,
		MaxIdleConns:          o.MaxIdleConns,
		MaxActiveConns:        o.MaxActiveConns,
		ConnMaxIdleTime:       o.ConnMaxIdleTime,
		ConnMaxLifetime:       o.ConnMaxLifetime,
		ConnMaxLifetimeJitter: o.ConnMaxLifetimeJitter,

		TLSConfig: o.TLSConfig,

		DisableIdentity:           o.DisableIdentity,
		DisableIndentity:          o.DisableIndentity,
		IdentitySuffix:            o.IdentitySuffix,
		AutoPipelineOptions:       o.AutoPipelineOptions,
		FailingTimeoutSeconds:     o.FailingTimeoutSeconds,
		UnstableResp3:             o.UnstableResp3,
		PushNotificationProcessor: o.PushNotificationProcessor,
		MaintNotificationsConfig:  o.MaintNotificationsConfig,
	}
}

// Failover returns failover options created from the universal options.
func (o *UniversalOptions) Failover() *FailoverOptions {
	if len(o.Addrs) == 0 {
		o.Addrs = []string{"127.0.0.1:26379"}
	}

	return &FailoverOptions{
		SentinelAddrs: o.Addrs,
		MasterName:    o.MasterName,
		ClientName:    o.ClientName,

		Dialer:    o.Dialer,
		OnConnect: o.OnConnect,

		DB:                           o.DB,
		Protocol:                     o.Protocol,
		Username:                     o.Username,
		Password:                     o.Password,
		CredentialsProvider:          o.CredentialsProvider,
		CredentialsProviderContext:   o.CredentialsProviderContext,
		StreamingCredentialsProvider: o.StreamingCredentialsProvider,

		SentinelUsername: o.SentinelUsername,
		SentinelPassword: o.SentinelPassword,

		RouteByLatency: o.RouteByLatency,
		RouteRandomly:  o.RouteRandomly,

		MaxRetries:      o.MaxRetries,
		MinRetryBackoff: o.MinRetryBackoff,
		MaxRetryBackoff: o.MaxRetryBackoff,

		DialTimeout:        o.DialTimeout,
		DialerRetries:      o.DialerRetries,
		DialerRetryTimeout: o.DialerRetryTimeout,
		ReadTimeout:        o.ReadTimeout,
		WriteTimeout:       o.WriteTimeout,

		ContextTimeoutEnabled: o.ContextTimeoutEnabled,

		ReadBufferSize:  o.ReadBufferSize,
		WriteBufferSize: o.WriteBufferSize,

		PoolFIFO:              o.PoolFIFO,
		PoolSize:              o.PoolSize,
		MaxConcurrentDials:    o.MaxConcurrentDials,
		PoolTimeout:           o.PoolTimeout,
		MinIdleConns:          o.MinIdleConns,
		MaxIdleConns:          o.MaxIdleConns,
		MaxActiveConns:        o.MaxActiveConns,
		ConnMaxIdleTime:       o.ConnMaxIdleTime,
		ConnMaxLifetime:       o.ConnMaxLifetime,
		ConnMaxLifetimeJitter: o.ConnMaxLifetimeJitter,

		TLSConfig: o.TLSConfig,

		ReplicaOnly: o.ReadOnly,

		DisableIdentity:           o.DisableIdentity,
		DisableIndentity:          o.DisableIndentity,
		IdentitySuffix:            o.IdentitySuffix,
		AutoPipelineOptions:       o.AutoPipelineOptions,
		UnstableResp3:             o.UnstableResp3,
		PushNotificationProcessor: o.PushNotificationProcessor,
		// Note: MaintNotificationsConfig not supported for FailoverOptions
	}
}

// Simple returns basic options created from the universal options.
func (o *UniversalOptions) Simple() *Options {
	addr := "127.0.0.1:6379"
	if len(o.Addrs) > 0 {
		addr = o.Addrs[0]
	}

	return &Options{
		Addr:       addr,
		ClientName: o.ClientName,
		Dialer:     o.Dialer,
		OnConnect:  o.OnConnect,

		DB:                           o.DB,
		Protocol:                     o.Protocol,
		Username:                     o.Username,
		Password:                     o.Password,
		CredentialsProvider:          o.CredentialsProvider,
		CredentialsProviderContext:   o.CredentialsProviderContext,
		StreamingCredentialsProvider: o.StreamingCredentialsProvider,

		MaxRetries:      o.MaxRetries,
		MinRetryBackoff: o.MinRetryBackoff,
		MaxRetryBackoff: o.MaxRetryBackoff,

		DialTimeout:        o.DialTimeout,
		DialerRetries:      o.DialerRetries,
		DialerRetryTimeout: o.DialerRetryTimeout,
		ReadTimeout:        o.ReadTimeout,
		WriteTimeout:       o.WriteTimeout,

		ContextTimeoutEnabled: o.ContextTimeoutEnabled,

		ReadBufferSize:  o.ReadBufferSize,
		WriteBufferSize: o.WriteBufferSize,

		PoolFIFO:              o.PoolFIFO,
		PoolSize:              o.PoolSize,
		MaxConcurrentDials:    o.MaxConcurrentDials,
		PoolTimeout:           o.PoolTimeout,
		MinIdleConns:          o.MinIdleConns,
		MaxIdleConns:          o.MaxIdleConns,
		MaxActiveConns:        o.MaxActiveConns,
		ConnMaxIdleTime:       o.ConnMaxIdleTime,
		ConnMaxLifetime:       o.ConnMaxLifetime,
		ConnMaxLifetimeJitter: o.ConnMaxLifetimeJitter,

		TLSConfig: o.TLSConfig,

		DisableIdentity:           o.DisableIdentity,
		DisableIndentity:          o.DisableIndentity,
		IdentitySuffix:            o.IdentitySuffix,
		AutoPipelineOptions:       o.AutoPipelineOptions,
		UnstableResp3:             o.UnstableResp3,
		PushNotificationProcessor: o.PushNotificationProcessor,
		MaintNotificationsConfig:  o.MaintNotificationsConfig,
		ClientSideCacheConfig:     o.ClientSideCacheConfig,
		ClientSideCache:           o.ClientSideCache,
		ClientSideCacheStrategy:   o.ClientSideCacheStrategy,
	}
}

// --------------------------------------------------------------------

// UniversalClient is an abstract client which - based on the provided options -
// represents either a ClusterClient, a FailoverClient, or a single-node Client.
// This can be useful for testing cluster-specific applications locally or having different
// clients in different environments.
type UniversalClient interface {
	Cmdable
	AddHook(Hook)
	Watch(ctx context.Context, fn func(*Tx) error, keys ...string) error
	Do(ctx context.Context, args ...interface{}) *Cmd
	Process(ctx context.Context, cmd Cmder) error
	// AutoPipeline / AsyncAutoPipeline return an AutoPipeliner for the concrete
	// client. Supported on *Client (including sentinel-backed failover clients)
	// and *ClusterClient; *Ring returns an error (not supported).
	//
	// EXPERIMENTAL: this API is subject to change, use with caution.
	AutoPipeline() (*AutoPipeliner, error)
	AutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error)
	AsyncAutoPipeline() (*AutoPipeliner, error)
	AsyncAutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error)
	Subscribe(ctx context.Context, channels ...string) *PubSub
	PSubscribe(ctx context.Context, channels ...string) *PubSub
	SSubscribe(ctx context.Context, channels ...string) *PubSub
	Close() error
	PoolStats() *PoolStats
}

var (
	_ UniversalClient = (*Client)(nil)
	_ UniversalClient = (*ClusterClient)(nil)
	_ UniversalClient = (*Ring)(nil)
	// AutoPipeliner is a drop-in for the real clients; non-data operations
	// delegate to the underlying client.
	_ UniversalClient = (*AutoPipeliner)(nil)
)

// NewUniversalClient returns a new multi client. The type of the returned client depends
// on the following conditions:
//
//  1. If the MasterName option is specified with RouteByLatency, RouteRandomly or IsClusterMode,
//     a FailoverClusterClient is returned.
//  2. If the MasterName option is specified without RouteByLatency, RouteRandomly or IsClusterMode,
//     a sentinel-backed FailoverClient is returned.
//  3. If the number of Addrs is two or more, or IsClusterMode option is specified,
//     a ClusterClient is returned.
//  4. Otherwise, a single-node Client is returned.
//
// Passing nil UniversalOptions will cause a panic.
func NewUniversalClient(opts *UniversalOptions) UniversalClient {
	if opts == nil {
		panic("redis: NewUniversalClient nil options")
	}

	switch {
	case opts.MasterName != "" && (opts.RouteByLatency || opts.RouteRandomly || opts.IsClusterMode):
		return NewFailoverClusterClient(opts.Failover())
	case opts.MasterName != "":
		return NewFailoverClient(opts.Failover())
	case len(opts.Addrs) > 1 || opts.IsClusterMode:
		return NewClusterClient(opts.Cluster())
	default:
		return NewClient(opts.Simple())
	}
}
