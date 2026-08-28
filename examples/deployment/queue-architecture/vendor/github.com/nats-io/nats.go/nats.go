// Copyright 2012-2026 The NATS Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A Go client for the NATS messaging system (https://nats.io).
package nats

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/nats-io/nkeys"
	"github.com/nats-io/nuid"

	"github.com/nats-io/nats.go/util"
)

// Default Constants
const (
	Version                   = "1.53.1"
	DefaultURL                = "nats://127.0.0.1:4222"
	DefaultPort               = 4222
	DefaultMaxReconnect       = 60
	DefaultReconnectWait      = 2 * time.Second
	DefaultReconnectJitter    = 100 * time.Millisecond
	DefaultReconnectJitterTLS = time.Second
	DefaultTimeout            = 2 * time.Second
	DefaultPingInterval       = 2 * time.Minute
	DefaultMaxPingOut         = 2
	DefaultMaxChanLen         = 64 * 1024       // 64k
	DefaultReconnectBufSize   = 8 * 1024 * 1024 // 8MB
	DefaultWriteBufSize       = defaultBufSize
	RequestChanLen            = 8
	DefaultDrainTimeout       = 30 * time.Second
	DefaultFlusherTimeout     = time.Minute
	LangString                = "go"
)

const (
	// STALE_CONNECTION is for detection and proper handling of stale connections.
	STALE_CONNECTION = "stale connection"

	// PERMISSIONS_ERR is for when nats server subject authorization has failed.
	PERMISSIONS_ERR = "permissions violation"

	// AUTHORIZATION_ERR is for when nats server user authorization has failed.
	AUTHORIZATION_ERR = "authorization violation"

	// AUTHENTICATION_EXPIRED_ERR is for when nats server user authorization has expired.
	AUTHENTICATION_EXPIRED_ERR = "user authentication expired"

	// AUTHENTICATION_REVOKED_ERR is for when user authorization has been revoked.
	AUTHENTICATION_REVOKED_ERR = "user authentication revoked"

	// ACCOUNT_AUTHENTICATION_EXPIRED_ERR is for when nats server account authorization has expired.
	ACCOUNT_AUTHENTICATION_EXPIRED_ERR = "account authentication expired"

	// MAX_CONNECTIONS_ERR is for when nats server denies the connection due to server max_connections limit.
	MAX_CONNECTIONS_ERR = "maximum connections exceeded"

	// MAX_ACCOUNT_CONNECTIONS_ERR is for when nats server denies the connection due to server max_connections limit on the account.
	MAX_ACCOUNT_CONNECTIONS_ERR = `maximum account active connections exceeded`

	// MAX_SUBSCRIPTIONS_ERR is for when nats server denies the connection due to server subscriptions limit.
	MAX_SUBSCRIPTIONS_ERR = "maximum subscriptions exceeded"
)

// Errors
var (
	ErrConnectionClosed              = errors.New("nats: connection closed")
	ErrConnectionDraining            = errors.New("nats: connection draining")
	ErrDrainTimeout                  = errors.New("nats: draining connection timed out")
	ErrConnectionReconnecting        = errors.New("nats: connection reconnecting")
	ErrSecureConnRequired            = errors.New("nats: secure connection required")
	ErrSecureConnWanted              = errors.New("nats: secure connection not available")
	ErrBadSubscription               = errors.New("nats: invalid subscription")
	ErrTypeSubscription              = errors.New("nats: invalid subscription type")
	ErrBadSubject                    = errors.New("nats: invalid subject")
	ErrBadQueueName                  = errors.New("nats: invalid queue name")
	ErrSlowConsumer                  = errors.New("nats: slow consumer, messages dropped")
	ErrTimeout                       = errors.New("nats: timeout")
	ErrBadTimeout                    = errors.New("nats: timeout invalid")
	ErrAuthorization                 = errors.New("nats: authorization violation")
	ErrAuthExpired                   = errors.New("nats: authentication expired")
	ErrAuthRevoked                   = errors.New("nats: authentication revoked")
	ErrPermissionViolation           = errors.New("nats: permissions violation")
	ErrAccountAuthExpired            = errors.New("nats: account authentication expired")
	ErrNoServers                     = errors.New("nats: no servers available for connection")
	ErrJsonParse                     = errors.New("nats: connect message, json parse error")
	ErrChanArg                       = errors.New("nats: argument needs to be a channel type")
	ErrMaxPayload                    = errors.New("nats: maximum payload exceeded")
	ErrMaxMessages                   = errors.New("nats: maximum messages delivered")
	ErrSyncSubRequired               = errors.New("nats: illegal call on an async subscription")
	ErrMultipleTLSConfigs            = errors.New("nats: multiple tls.Configs not allowed")
	ErrClientCertOrRootCAsRequired   = errors.New("nats: at least one of certCB or rootCAsCB must be set")
	ErrNoInfoReceived                = errors.New("nats: protocol exception, INFO not received")
	ErrReconnectBufExceeded          = errors.New("nats: outbound buffer limit exceeded")
	ErrInvalidConnection             = errors.New("nats: invalid connection")
	ErrInvalidMsg                    = errors.New("nats: invalid message or message nil")
	ErrInvalidArg                    = errors.New("nats: invalid argument")
	ErrInvalidContext                = errors.New("nats: invalid context")
	ErrNoDeadlineContext             = errors.New("nats: context requires a deadline")
	ErrNoEchoNotSupported            = errors.New("nats: no echo option not supported by this server")
	ErrClientIDNotSupported          = errors.New("nats: client ID not supported by this server")
	ErrUserButNoSigCB                = errors.New("nats: user callback defined without a signature handler")
	ErrNkeyButNoSigCB                = errors.New("nats: nkey defined without a signature handler")
	ErrNoUserCB                      = errors.New("nats: user callback not defined")
	ErrNkeyAndUser                   = errors.New("nats: user callback and nkey defined")
	ErrNkeysNotSupported             = errors.New("nats: nkeys not supported by the server")
	ErrStaleConnection               = errors.New("nats: " + STALE_CONNECTION)
	ErrTokenAlreadySet               = errors.New("nats: token and token handler both set")
	ErrUserInfoAlreadySet            = errors.New("nats: cannot set user info callback and user/pass")
	ErrMsgNotBound                   = errors.New("nats: message is not bound to subscription/connection")
	ErrMsgNoReply                    = errors.New("nats: message does not have a reply")
	ErrClientIPNotSupported          = errors.New("nats: client IP not supported by this server")
	ErrDisconnected                  = errors.New("nats: server is disconnected")
	ErrHeadersNotSupported           = errors.New("nats: headers not supported by this server")
	ErrBadHeaderMsg                  = errors.New("nats: message could not decode headers")
	ErrNoResponders                  = errors.New("nats: no responders available for request")
	ErrMaxConnectionsExceeded        = errors.New("nats: server maximum connections exceeded")
	ErrMaxAccountConnectionsExceeded = errors.New("nats: maximum account active connections exceeded")
	ErrConnectionNotTLS              = errors.New("nats: connection is not tls")
	ErrTLS                           = errors.New("nats: tls error")
	ErrMaxSubscriptionsExceeded      = errors.New("nats: server maximum subscriptions exceeded")
	ErrWebSocketHeadersAlreadySet    = errors.New("nats: websocket connection headers already set")
	ErrServerNotInPool               = errors.New("nats: selected server is not in the pool")
	ErrMixingWebsocketSchemes        = errors.New("nats: mixing of websocket and non websocket URLs is not allowed")
)

// GetDefaultOptions returns default configuration options for the client.
func GetDefaultOptions() Options {
	return Options{
		AllowReconnect:     true,
		MaxReconnect:       DefaultMaxReconnect,
		ReconnectWait:      DefaultReconnectWait,
		ReconnectJitter:    DefaultReconnectJitter,
		ReconnectJitterTLS: DefaultReconnectJitterTLS,
		Timeout:            DefaultTimeout,
		PingInterval:       DefaultPingInterval,
		MaxPingsOut:        DefaultMaxPingOut,
		SubChanLen:         DefaultMaxChanLen,
		ReconnectBufSize:   DefaultReconnectBufSize,
		DrainTimeout:       DefaultDrainTimeout,
		FlusherTimeout:     DefaultFlusherTimeout,
	}
}

// Deprecated: Use GetDefaultOptions() instead.
// DefaultOptions is not safe for use by multiple clients.
// For details see #308.
var DefaultOptions = GetDefaultOptions()

// Status represents the state of the connection.
type Status int

const (
	DISCONNECTED = Status(iota)
	CONNECTED
	CLOSED
	RECONNECTING
	CONNECTING
	DRAINING_SUBS
	DRAINING_PUBS
)

func (s Status) String() string {
	switch s {
	case DISCONNECTED:
		return "DISCONNECTED"
	case CONNECTED:
		return "CONNECTED"
	case CLOSED:
		return "CLOSED"
	case RECONNECTING:
		return "RECONNECTING"
	case CONNECTING:
		return "CONNECTING"
	case DRAINING_SUBS:
		return "DRAINING_SUBS"
	case DRAINING_PUBS:
		return "DRAINING_PUBS"
	}
	return "unknown status"
}

// ConnHandler is used for asynchronous events such as
// disconnected and closed connections.
type ConnHandler func(*Conn)

// ConnErrHandler is used to process asynchronous events like
// disconnected connection with the error (if any).
type ConnErrHandler func(*Conn, error)

// ErrHandler is used to process asynchronous errors encountered
// while processing inbound messages.
type ErrHandler func(*Conn, *Subscription, error)

// UserJWTHandler is used to fetch and return the account signed
// JWT for this user.
type UserJWTHandler func() (string, error)

// TLSCertHandler is used to fetch and return tls certificate.
type TLSCertHandler func() (tls.Certificate, error)

// RootCAsHandler is used to fetch and return a set of root certificate
// authorities that clients use when verifying server certificates.
type RootCAsHandler func() (*x509.CertPool, error)

// SignatureHandler is used to sign a nonce from the server while
// authenticating with nkeys. The user should sign the nonce and
// return the raw signature. The client will base64 encode this to
// send to the server.
type SignatureHandler func([]byte) ([]byte, error)

// AuthTokenHandler is used to generate a new token.
type AuthTokenHandler func() string

// UserInfoCB is used to pass the username and password when establishing a connection.
type UserInfoCB func() (string, string)

// ReconnectDelayHandler is used to get from the user the desired
// delay the library should pause before attempting to reconnect
// again. Note that this is invoked after the library tried the
// whole list of URLs and failed to reconnect.
type ReconnectDelayHandler func(attempts int) time.Duration

// WebSocketHeadersHandler is an optional callback handler for generating token used for WebSocket connections.
type WebSocketHeadersHandler func() (http.Header, error)

// ReconnectToServerHandler is used to determine the server to reconnect to during
// the reconnection process. The handler receives a snapshot of available servers
// in the pool and should return a pointer to one of the servers from the provided
// slice and a delay before attempting the connection.
//
// Return values:
//   - *Server: The server to connect to. Must be a pointer to an element from the
//     provided servers slice. If the returned server is not in the pool, the library
//     will fire ReconnectErrCB with ErrServerNotInPool and fall back to default
//     server selection.
//   - time.Duration: The delay before attempting the connection. If zero, the
//     connection attempt is made immediately. If non-zero, the library sleeps
//     for exactly that duration before attempting.
//
// MaxReconnect limits are enforced automatically: servers exceeding the configured
// MaxReconnect attempts are removed from the pool before the handler is called.
// To disable this limit, set MaxReconnect to a negative value.
type ReconnectToServerHandler func([]Server, ServerInfo) (*Server, time.Duration)

// asyncCB is used to preserve order for async callbacks.
type asyncCB struct {
	f    func()
	next *asyncCB
}

type asyncCallbacksHandler struct {
	mu   sync.Mutex
	cond *sync.Cond
	head *asyncCB
	tail *asyncCB
}

// Option is a function on the options for a connection.
type Option func(*Options) error

// CustomDialer can be used to specify any dialer, not necessarily a
// *net.Dialer.  A CustomDialer may also implement `SkipTLSHandshake() bool`
// in order to skip the TLS handshake in case not required.
type CustomDialer interface {
	Dial(network, address string) (net.Conn, error)
}

type InProcessConnProvider interface {
	InProcessConn() (net.Conn, error)
}

// Options can be used to create a customized connection.
type Options struct {
	// Url represents a single NATS server url to which the client
	// will be connecting. If the Servers option is also set, it
	// then becomes the first server in the Servers array.
	Url string

	// InProcessServer represents a NATS server running within the
	// same process. If this is set then we will attempt to connect
	// to the server directly rather than using external TCP conns.
	InProcessServer InProcessConnProvider

	// Servers is a configured set of servers which this client
	// will use when attempting to connect.
	Servers []string

	// NoRandomize configures whether we will randomize the
	// server pool.
	NoRandomize bool

	// NoEcho configures whether the server will echo back messages
	// that are sent on this connection if we also have matching subscriptions.
	// Note this is supported on servers >= version 1.2. Proto 1 or greater.
	NoEcho bool

	// Name is an optional name label which will be sent to the server
	// on CONNECT to identify the client.
	Name string

	// Verbose signals the server to send an OK ack for commands
	// successfully processed by the server.
	Verbose bool

	// Pedantic signals the server whether it should be doing further
	// validation of subjects.
	Pedantic bool

	// Secure enables TLS secure connections that skip server
	// verification by default. NOT RECOMMENDED.
	Secure bool

	// TLSConfig is a custom TLS configuration to use for secure
	// transports.
	TLSConfig *tls.Config

	// TLSCertCB is used to fetch and return custom tls certificate.
	TLSCertCB TLSCertHandler

	// TLSHandshakeFirst is used to instruct the library perform
	// the TLS handshake right after the connect and before receiving
	// the INFO protocol from the server. If this option is enabled
	// but the server is not configured to perform the TLS handshake
	// first, the connection will fail.
	TLSHandshakeFirst bool

	// RootCAsCB is used to fetch and return a set of root certificate
	// authorities that clients use when verifying server certificates.
	RootCAsCB RootCAsHandler

	// AllowReconnect enables reconnection logic to be used when we
	// encounter a disconnect from the current server.
	AllowReconnect bool

	// MaxReconnect sets the number of reconnect attempts that will be
	// tried before giving up. If negative, then it will never give up
	// trying to reconnect.
	// Defaults to 60.
	MaxReconnect int

	// ReconnectWait sets the time to back off after attempting a reconnect
	// to a server that we were already connected to previously.
	// Defaults to 2s.
	ReconnectWait time.Duration

	// CustomReconnectDelayCB is invoked after the library tried every
	// URL in the server list and failed to reconnect. It passes to the
	// user the current number of attempts. This function returns the
	// amount of time the library will sleep before attempting to reconnect
	// again. It is strongly recommended that this value contains some
	// jitter to prevent all connections to attempt reconnecting at the same time.
	CustomReconnectDelayCB ReconnectDelayHandler

	// ReconnectJitter sets the upper bound for a random delay added to
	// ReconnectWait during a reconnect when no TLS is used.
	// Defaults to 100ms.
	ReconnectJitter time.Duration

	// ReconnectJitterTLS sets the upper bound for a random delay added to
	// ReconnectWait during a reconnect when TLS is used.
	// Defaults to 1s.
	ReconnectJitterTLS time.Duration

	// Timeout sets the timeout for a Dial operation on a connection.
	// Defaults to 2s.
	Timeout time.Duration

	// DrainTimeout sets the timeout for a drain operation to complete.
	// Defaults to 30s.
	DrainTimeout time.Duration

	// FlusherTimeout is the maximum time to wait for write operations
	// to the underlying connection to complete (including the flusher loop).
	// Defaults to 1m.
	FlusherTimeout time.Duration

	// ReconnectOnFlusherError, when set to true, causes the client to
	// trigger a reconnect if the background flusher fails to write to the
	// underlying connection for any reason (timeout, broken pipe,
	// connection reset, EOF etc.).
	//
	// This is an advanced option. Most applications do not need to enable
	// it: the server-side stale connection detection (via PingInterval /
	// MaxPingsOut) and the read loop's own error handling will eventually
	// notice a dead connection and the client will reconnect. Enable this
	// only if you need faster recovery from a stalled or broken TCP write
	// — for example, in latency-sensitive setups where waiting for a ping
	// timeout is unacceptable.
	//
	// Messages buffered at the time of the error are lost, as they are
	// with any flusher write error. The purpose of this option is to
	// limit the blast radius by preventing further messages from being
	// buffered into a potentially corrupted connection, not to recover
	// the in-flight data.
	//
	// When triggered, the standard DisconnectErrHandler and
	// ReconnectHandler callbacks are invoked as with any other reconnect.
	// The first reconnect attempt bypasses the configured ReconnectWait
	// so that recovery is as fast as possible; if that attempt fails,
	// subsequent attempts obey the normal backoff.
	//
	// Defaults to false.
	ReconnectOnFlusherError bool

	// PingInterval is the period at which the client will be sending ping
	// commands to the server, disabled if 0 or negative.
	// Defaults to 2m.
	PingInterval time.Duration

	// MaxPingsOut is the maximum number of pending ping commands that can
	// be awaiting a response before raising an ErrStaleConnection error.
	// Defaults to 2.
	MaxPingsOut int

	// ClosedCB sets the closed handler that is called when a client will
	// no longer be connected.
	ClosedCB ConnHandler

	// DisconnectedCB sets the disconnected handler that is called
	// whenever the connection is disconnected.
	// Will not be called if DisconnectedErrCB is set
	// Deprecated. Use DisconnectedErrCB which passes error that caused
	// the disconnect event.
	DisconnectedCB ConnHandler

	// DisconnectedErrCB sets the disconnected error handler that is called
	// whenever the connection is disconnected.
	// Disconnected error could be nil, for instance when user explicitly closes the connection.
	// DisconnectedCB will not be called if DisconnectedErrCB is set
	DisconnectedErrCB ConnErrHandler

	// ConnectedCB sets the connected handler called when the initial connection
	// is established. It is not invoked on successful reconnects - for reconnections,
	// use ReconnectedCB. ConnectedCB can be used in conjunction with RetryOnFailedConnect
	// to detect whether the initial connect was successful.
	ConnectedCB ConnHandler

	// ReconnectedCB sets the reconnected handler called whenever
	// the connection is successfully reconnected.
	ReconnectedCB ConnHandler

	// DiscoveredServersCB sets the callback that is invoked whenever a new
	// server has joined the cluster.
	DiscoveredServersCB ConnHandler

	// AsyncErrorCB sets the async error handler (e.g. slow consumer errors)
	AsyncErrorCB ErrHandler

	// ReconnectErrCB sets the callback that is invoked whenever a
	// reconnect attempt fails.
	ReconnectErrCB ConnErrHandler

	// ReconnectToServerCB is called before reconnection attempt.
	// It is used to determine the server to reconnect to out of
	// the list of available servers.
	// If a reconnect attempt is not successful, this callback will
	// be called again before the next attempt.
	ReconnectToServerCB ReconnectToServerHandler

	// ReconnectBufSize is the size of the backing bufio during reconnect.
	// Once this has been exhausted publish operations will return an error.
	// Defaults to 8388608 bytes (8MB).
	ReconnectBufSize int

	// SubChanLen is the size of the buffered channel used between the socket
	// goroutine and the message delivery for SyncSubscriptions.
	// NOTE: This does not affect AsyncSubscriptions which are
	// dictated by PendingLimits().
	// Defaults to 65536.
	SubChanLen int

	// UserJWT sets the callback handler that will fetch a user's JWT.
	UserJWT UserJWTHandler

	// Nkey sets the public nkey that will be used to authenticate
	// when connecting to the server. UserJWT and Nkey are mutually exclusive
	// and if defined, UserJWT will take precedence.
	Nkey string

	// SignatureCB designates the function used to sign the nonce
	// presented from the server.
	SignatureCB SignatureHandler

	// User sets the username to be used when connecting to the server.
	User string

	// Password sets the password to be used when connecting to a server.
	Password string

	// UserInfo sets the callback handler that will fetch the username and password.
	UserInfo UserInfoCB

	// Token sets the token to be used when connecting to a server.
	Token string

	// TokenHandler designates the function used to generate the token
	// used when connecting to a server.
	TokenHandler AuthTokenHandler

	// Dialer allows a custom net.Dialer when forming connections.
	// Deprecated: should use CustomDialer instead.
	Dialer *net.Dialer

	// CustomDialer allows to specify a custom dialer (not necessarily
	// a *net.Dialer).
	CustomDialer CustomDialer

	// UseOldRequestStyle forces the old method of Requests that utilize
	// a new Inbox and a new Subscription for each request.
	UseOldRequestStyle bool

	// NoCallbacksAfterClientClose allows preventing the invocation of
	// callbacks after Close() is called. Client won't receive notifications
	// when Close is invoked by user code. Default is to invoke the callbacks.
	NoCallbacksAfterClientClose bool

	// LameDuckModeHandler sets the callback to invoke when the server notifies
	// the connection that it entered lame duck mode, that is, going to
	// gradually disconnect all its connections before shutting down. This is
	// often used in deployments when upgrading NATS Servers.
	LameDuckModeHandler ConnHandler

	// RetryOnFailedConnect sets the connection in reconnecting state right
	// away if it can't connect to a server in the initial set. The
	// MaxReconnect and ReconnectWait options are used for this process,
	// similarly to when an established connection is disconnected.
	// If a ReconnectHandler is set, it will be invoked on the first
	// successful reconnect attempt (if the initial connect fails),
	// and if a ClosedHandler is set, it will be invoked if
	// it fails to connect (after exhausting the MaxReconnect attempts).
	RetryOnFailedConnect bool

	// For websocket connections, indicates to the server that the connection
	// supports compression. If the server does too, then data will be compressed.
	Compression bool

	// For websocket connections, adds a path to connections url.
	// This is useful when connecting to NATS behind a proxy.
	ProxyPath string

	// InboxPrefix allows the default _INBOX prefix to be customized
	InboxPrefix string

	// IgnoreAuthErrorAbort - if set to true, client opts out of the default connect behavior of aborting
	// subsequent reconnect attempts if server returns the same auth error twice (regardless of reconnect policy).
	IgnoreAuthErrorAbort bool

	// SkipHostLookup skips the DNS lookup for the server hostname.
	SkipHostLookup bool

	// PermissionErrOnSubscribe - if set to true, the client will return ErrPermissionViolation
	// from SubscribeSync if the server returns a permissions error for a subscription.
	// Defaults to false.
	PermissionErrOnSubscribe bool

	// WebSocketConnectionHeaders is an optional http request headers to be sent with the WebSocket request.
	WebSocketConnectionHeaders http.Header

	// WebSocketConnectionHeadersHandler is an optional callback handler for generating token used for WebSocket connections.
	WebSocketConnectionHeadersHandler WebSocketHeadersHandler

	// SkipSubjectValidation will disable publish subject validation.
	// NOTE: This is not recommended in general, as the performance gain is minimal
	// and may lead to breaking protocol.
	SkipSubjectValidation bool

	// IgnoreDiscoveredServers will disable adding advertised server URLs
	// from INFO messages to the server pool.
	IgnoreDiscoveredServers bool

	// WriteBufferSize is an advanced option that sets the flush threshold
	// of the write buffer used to batch outgoing data before writing to
	// the underlying connection. In most cases, the default value should
	// not be changed. A smaller buffer reduces the amount of data that
	// can be lost on blocked writes but may significantly reduce throughput.
	// Defaults to 32768 bytes (32KB).
	WriteBufferSize int
}

const (
	// Scratch storage for assembling protocol headers
	scratchSize = 512

	// The size of the bufio reader/writer on top of the socket.
	defaultBufSize = 32768

	// The buffered size of the flush "kick" channel
	flushChanSize = 1

	// Default server pool size
	srvPoolSize = 4

	// NUID size
	nuidSize = 22

	// Default ports used if none is specified in given URL(s)
	defaultWSPortString  = "80"
	defaultWSSPortString = "443"
	defaultPortString    = "4222"
)

// A Conn represents a bare connection to a nats-server.
// It can send and receive []byte payloads.
// The connection is safe to use in multiple Go routines concurrently.
type Conn struct {
	// Keep all members for which we use atomic at the beginning of the
	// struct and make sure they are all 64bits (or use padding if necessary).
	// atomic.* functions crash on 32bit machines if operand is not aligned
	// at 64bit. See https://github.com/golang/go/issues/599
	Statistics
	mu sync.RWMutex
	// Opts holds the configuration of the Conn.
	// Modifying the configuration of a running Conn is a race.
	Opts          Options
	wg            sync.WaitGroup
	srvPool       []*Server
	current       *Server
	urls          map[string]struct{} // Keep track of all known URLs (used by processInfo)
	conn          net.Conn
	bw            *natsWriter
	br            *natsReader
	fch           chan struct{}
	info          ServerInfo
	ssid          int64
	subsMu        sync.RWMutex
	subs          map[int64]*Subscription
	ach           *asyncCallbacksHandler
	pongs         []chan struct{}
	scratch       [scratchSize]byte
	status        Status
	statListeners map[Status]map[chan Status]struct{}
	initc         bool // true if the connection is performing the initial connect
	err           error
	ps            *parseState
	ptmr          *time.Timer
	pout          int
	ar            bool // abort reconnect
	rqch          chan struct{}
	ws            bool // true if a websocket connection

	// New style response handler
	respSub       string               // The wildcard subject
	respSubPrefix string               // the wildcard prefix including trailing .
	respSubLen    int                  // the length of the wildcard prefix excluding trailing .
	respMux       *Subscription        // A single response subscription
	respMap       map[string]chan *Msg // Request map for the response msg channels
	respRand      *rand.Rand           // Used for generating suffix

	// Msg filters for testing.
	// Protected by subsMu
	filters map[string]msgFilter
}

type natsReader struct {
	r   io.Reader
	buf []byte
	off int
	n   int
}

type natsWriter struct {
	w       io.Writer
	bufs    []byte
	limit   int
	pending *bytes.Buffer
	plimit  int
}

// Subscription represents interest in a given subject.
type Subscription struct {
	mu  sync.Mutex
	sid int64

	// Subject that represents this subscription. This can be different
	// than the received subject inside a Msg if this is a wildcard.
	Subject string

	// Optional queue group name. If present, all subscriptions with the
	// same name will form a distributed queue, and each message will
	// only be processed by one member of the group.
	Queue string

	// For holding information about a JetStream consumer.
	jsi *jsSub

	delivered      uint64
	max            uint64
	conn           *Conn
	mcb            MsgHandler
	mch            chan *Msg
	errCh          chan (error)
	closed         bool
	sc             bool
	connClosed     bool
	draining       bool
	status         SubStatus
	statListeners  map[chan SubStatus][]SubStatus
	permissionsErr error

	// Type of Subscription
	typ SubscriptionType

	// Async linked list
	pHead *Msg
	pTail *Msg
	pCond *sync.Cond
	pDone func(subject string)

	// Pending stats, async subscriptions, high-speed etc.
	pMsgs       int
	pBytes      int
	pMsgsMax    int
	pBytesMax   int
	pMsgsLimit  int
	pBytesLimit int
	dropped     int
}

// Status represents the state of the connection.
type SubStatus int

const (
	SubscriptionActive = SubStatus(iota)
	SubscriptionDraining
	SubscriptionClosed
	SubscriptionSlowConsumer
)

func (s SubStatus) String() string {
	switch s {
	case SubscriptionActive:
		return "Active"
	case SubscriptionDraining:
		return "Draining"
	case SubscriptionClosed:
		return "Closed"
	case SubscriptionSlowConsumer:
		return "SlowConsumer"
	}
	return "unknown status"
}

// Msg represents a message delivered by NATS. This structure is used
// by Subscribers and PublishMsg().
//
// # Types of Acknowledgements
//
// In case using JetStream, there are multiple ways to ack a Msg:
//
//	// Acknowledgement that a message has been processed.
//	msg.Ack()
//
//	// Negatively acknowledges a message.
//	msg.Nak()
//
//	// Terminate a message so that it is not redelivered further.
//	msg.Term()
//
//	// Signal the server that the message is being worked on and reset redelivery timer.
//	msg.InProgress()
type Msg struct {
	Subject string
	Reply   string
	Header  Header
	Data    []byte
	Sub     *Subscription
	// Internal
	next    *Msg
	wsz     int
	barrier *barrierInfo
	ackd    uint32
}

// Compares two msgs, ignores sub but checks all other public fields.
func (m *Msg) Equal(msg *Msg) bool {
	if m == msg {
		return true
	}
	if m == nil || msg == nil {
		return false
	}
	if m.Subject != msg.Subject || m.Reply != msg.Reply {
		return false
	}
	if !bytes.Equal(m.Data, msg.Data) {
		return false
	}
	if len(m.Header) != len(msg.Header) {
		return false
	}
	for k, v := range m.Header {
		val, ok := msg.Header[k]
		if !ok || len(v) != len(val) {
			return false
		}
		for i, hdr := range v {
			if hdr != val[i] {
				return false
			}
		}
	}
	return true
}

// Size returns a message size in bytes.
func (m *Msg) Size() int {
	if m.wsz != 0 {
		return m.wsz
	}
	hdr, _ := m.headerBytes()
	return len(m.Subject) + len(m.Reply) + len(hdr) + len(m.Data)
}

func (m *Msg) headerBytes() ([]byte, error) {
	var hdr []byte
	if len(m.Header) == 0 {
		return hdr, nil
	}

	// Validate the keys and calculate an upper bound for the total encoded
	// size.
	//
	// The size calculation may not be exact if any values contain
	// leading/trailing whitespace (as those will be trimmed later), but a
	// slight over-estimation here is more preferable than an under-estimation
	// (which could lead to unnecessary memory allocations).
	//
	// We don't perform any special validation on header values because we
	// consider all strings to be valid values (including Unicode strings,
	// special characters, etc.).
	hdrSize := len(hdrLine) + len(crlf)
	for k, vs := range m.Header {
		if !isHeaderKeyValid(k) {
			return nil, ErrBadHeaderMsg
		}
		for _, v := range vs {
			hdrSize += len(k) + len(v) + 4 // 4 is for the colon, space, CR, and LF
		}
	}

	// NOTE: bytes.Buffer.WriteString() returns an error because bytes.Buffer
	// is intended to be a valid implementation of the io.StringWriter interface.
	// This implementation cannot actually fail in practice, so it is safe to
	// ignore write errors here.
	var b bytes.Buffer
	b.Grow(hdrSize)
	_, _ = b.WriteString(hdrLine)
	for k, vs := range m.Header {
		for _, v := range vs {
			_, _ = b.WriteString(k)
			_, _ = b.WriteString(": ")
			writeHeaderValue(&b, v)
			_, _ = b.WriteString(crlf)
		}
	}
	_, _ = b.WriteString(crlf)

	return b.Bytes(), nil
}

// asciiSet is a 256-byte lookup table for fast ASCII character membership testing.
// This implementation mirrors the one Go uses internally for many string operations,
// including strings.Contains().
//
// References:
//   - https://github.com/golang/go/blob/master/src/strings/strings.go
type asciiSet [256]bool

func (as *asciiSet) MatchesString(s string) bool {

	// Implementation note: Our high level strategy is to compare each byte in
	// 's' with the corresponding entry in 'as'. If the entry is 'true', the
	// character is valid, so we can move forward with the next byte in 's'. If
	// the entry in 'as' is 'false', the character is invalid, so we can bail
	// out.
	//
	// Benchmarking shows there to be some benefit to manually unrolling the
	// loop (up to ~35% faster in some cases), so that's what we've done here.

	i := 0

	// Process the string in batches of 8 bytes
	for ; i <= len(s)-8; i += 8 {
		if !as[s[i]] {
			return false
		}
		if !as[s[i+1]] {
			return false
		}
		if !as[s[i+2]] {
			return false
		}
		if !as[s[i+3]] {
			return false
		}
		if !as[s[i+4]] {
			return false
		}
		if !as[s[i+5]] {
			return false
		}
		if !as[s[i+6]] {
			return false
		}
		if !as[s[i+7]] {
			return false
		}
	}

	// Handle any remaining bytes at the end of the string
	for ; i < len(s); i++ {
		if !as[s[i]] {
			return false
		}
	}
	return true
}

// The asciiSet that represents valid header keys can be precalculated and
// cached since it will never change during the lifetime of the application.
var validHeaderKeyChars = makeValidHeaderKeyAsciiSet()

func makeValidHeaderKeyAsciiSet() asciiSet {

	// ADR-4 specifies that header keys must be printable ASCII characters
	// (e.g. those in the range [33, 126]), except for colons. This
	// implementation is very slightly stricter, as it disallows several
	// additional special characters.
	const forbiddenChars = "\"()/,:;<=>?@[\\]{}"

	var as asciiSet
	for i := rune(33); i <= rune(126); i++ {
		if !strings.ContainsRune(forbiddenChars, i) {
			as[i] = true
		}
	}
	return as
}

func isHeaderKeyValid(k string) bool {
	return len(k) > 0 && validHeaderKeyChars.MatchesString(k)
}

var headerValueNewlineReplacer = strings.NewReplacer("\r", " ", "\n", " ")

func writeHeaderValue(buffer *bytes.Buffer, value string) {

	// ADR-4 specifies that header values must be ASCII characters (except for
	// '\r' or '\n'), however the Go implementation is slightly more lenient
	// to maintain backwards compatibility with older versions of the library.
	// We allow arbitrary UTF-8 strings to be used as values and deliberately
	// sanitize '\r' and '\n' characters by replacing them with spaces.

	// NOTE: It is safe to ignore the error returned by WriteString because
	// we're writing to a bytes.Buffer object, and that implementation never
	// returns an error.
	_, _ = headerValueNewlineReplacer.WriteString(buffer, textproto.TrimString(value))
}

type barrierInfo struct {
	refs int64
	f    func()
}

// Tracks various stats received and sent on this connection,
// including counts for messages and bytes.
type Statistics struct {
	InMsgs     uint64
	OutMsgs    uint64
	InBytes    uint64
	OutBytes   uint64
	Reconnects uint64
}

// Server represents a server in the pool of servers that the client can connect to.
type Server struct {
	URL        *url.URL
	Reconnects int
	didConnect bool
	lastErr    error
	isImplicit bool
	tlsName    string
}

func (s Server) clone() Server {
	c := s
	if s.URL != nil {
		u := *s.URL
		c.URL = &u
	}
	return c
}

// ServerInfo represents the information about the server that is sent in the INFO protocol message.
type ServerInfo struct {
	ID           string   `json:"server_id"`
	Name         string   `json:"server_name"`
	Proto        int      `json:"proto"`
	Version      string   `json:"version"`
	Host         string   `json:"host"`
	Port         int      `json:"port"`
	Headers      bool     `json:"headers"`
	AuthRequired bool     `json:"auth_required,omitempty"`
	TLSRequired  bool     `json:"tls_required,omitempty"`
	TLSAvailable bool     `json:"tls_available,omitempty"`
	MaxPayload   int64    `json:"max_payload"`
	CID          uint64   `json:"client_id,omitempty"`
	ClientIP     string   `json:"client_ip,omitempty"`
	Nonce        string   `json:"nonce,omitempty"`
	Cluster      string   `json:"cluster,omitempty"`
	ConnectURLs  []string `json:"connect_urls,omitempty"`
	LameDuckMode bool     `json:"ldm,omitempty"`
	// JetStream indicates whether the server has JetStream enabled.
	JetStream bool `json:"jetstream,omitempty"`
	// IsSystemAccount indicates whether the connected client's account
	// is the system account.
	IsSystemAccount bool `json:"acc_is_sys,omitempty"`
	// JSApiLevel is the JetStream API level advertised by the server.
	// Requires nats-server v2.12.0 or later; older servers will report 0.
	JSApiLevel int `json:"api_lvl,omitempty"`
}

const (
	// clientProtoZero is the original client protocol from 2009.
	// http://nats.io/documentation/internals/nats-protocol/
	/* clientProtoZero */ _ = iota
	// clientProtoInfo signals a client can receive more then the original INFO block.
	// This can be used to update clients on other cluster members, etc.
	clientProtoInfo
)

type connectInfo struct {
	Verbose      bool   `json:"verbose"`
	Pedantic     bool   `json:"pedantic"`
	UserJWT      string `json:"jwt,omitempty"`
	Nkey         string `json:"nkey,omitempty"`
	Signature    string `json:"sig,omitempty"`
	User         string `json:"user,omitempty"`
	Pass         string `json:"pass,omitempty"`
	Token        string `json:"auth_token,omitempty"`
	TLS          bool   `json:"tls_required"`
	Name         string `json:"name"`
	Lang         string `json:"lang"`
	Version      string `json:"version"`
	Protocol     int    `json:"protocol"`
	Echo         bool   `json:"echo"`
	Headers      bool   `json:"headers"`
	NoResponders bool   `json:"no_responders"`
}

// MsgHandler is a callback function that processes messages delivered to
// asynchronous subscribers.
type MsgHandler func(msg *Msg)

// Connect will attempt to connect to the NATS system.
// The url can contain username/password semantics. e.g. nats://derek:pass@localhost:4222
// Comma separated arrays are also supported, e.g. urlA, urlB.
// Options start with the defaults but can be overridden.
// To connect to a NATS Server's websocket port, use the `ws` or `wss` scheme, such as
// `ws://localhost:8080`. Note that websocket schemes cannot be mixed with others (nats/tls).
func Connect(url string, options ...Option) (*Conn, error) {
	opts := GetDefaultOptions()
	opts.Servers = processUrlString(url)
	for _, opt := range options {
		if opt != nil {
			if err := opt(&opts); err != nil {
				return nil, err
			}
		}
	}
	return opts.Connect()
}

// Options that can be passed to Connect.

// Name is an Option to set the client name.
func Name(name string) Option {
	return func(o *Options) error {
		o.Name = name
		return nil
	}
}

// InProcessServer is an Option that will try to establish a connection to a NATS server
// running within the process instead of dialing via TCP.
func InProcessServer(server InProcessConnProvider) Option {
	return func(o *Options) error {
		o.InProcessServer = server
		return nil
	}
}

// Secure is an Option to enable TLS secure connections that skip server verification by default.
// Pass a TLS Configuration for proper TLS.
// A TLS Configuration using InsecureSkipVerify should NOT be used in a production setting.
func Secure(tls ...*tls.Config) Option {
	return func(o *Options) error {
		o.Secure = true
		// Use of variadic just simplifies testing scenarios. We only take the first one.
		if len(tls) > 1 {
			return ErrMultipleTLSConfigs
		}
		if len(tls) == 1 {
			o.TLSConfig = tls[0]
		}
		return nil
	}
}

// ClientTLSConfig is an Option to set the TLS configuration for secure
// connections. It can be used to e.g. set TLS config with cert and root CAs
// from memory. For simple use case of loading cert and CAs from file,
// ClientCert and RootCAs options are more convenient.
// If Secure is not already set this will set it as well.
func ClientTLSConfig(certCB TLSCertHandler, rootCAsCB RootCAsHandler) Option {
	return func(o *Options) error {
		o.Secure = true

		if certCB == nil && rootCAsCB == nil {
			return ErrClientCertOrRootCAsRequired
		}

		// Smoke test the callbacks to fail early
		// if they are not valid.
		if certCB != nil {
			if _, err := certCB(); err != nil {
				return err
			}
		}
		if rootCAsCB != nil {
			if _, err := rootCAsCB(); err != nil {
				return err
			}
		}
		if o.TLSConfig == nil {
			o.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
		}
		o.TLSCertCB = certCB
		o.RootCAsCB = rootCAsCB
		return nil
	}
}

// RootCAs is a helper option to provide the RootCAs pool from a list of filenames.
// If Secure is not already set this will set it as well.
func RootCAs(file ...string) Option {
	return func(o *Options) error {
		rootCAsCB := func() (*x509.CertPool, error) {
			pool := x509.NewCertPool()
			for _, f := range file {
				rootPEM, err := os.ReadFile(f)
				if err != nil || rootPEM == nil {
					return nil, fmt.Errorf("nats: error loading or parsing rootCA file: %w", err)
				}
				ok := pool.AppendCertsFromPEM(rootPEM)
				if !ok {
					return nil, fmt.Errorf("nats: failed to parse root certificate from %q", f)
				}
			}
			return pool, nil
		}
		if o.TLSConfig == nil {
			o.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
		}
		if _, err := rootCAsCB(); err != nil {
			return err
		}
		o.RootCAsCB = rootCAsCB
		o.Secure = true
		return nil
	}
}

// ClientCert is a helper option to provide the client certificate from a file.
// If Secure is not already set this will set it as well.
func ClientCert(certFile, keyFile string) Option {
	return func(o *Options) error {
		tlsCertCB := func() (tls.Certificate, error) {
			cert, err := tls.LoadX509KeyPair(certFile, keyFile)
			if err != nil {
				return tls.Certificate{}, fmt.Errorf("nats: error loading client certificate: %w", err)
			}
			cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
			if err != nil {
				return tls.Certificate{}, fmt.Errorf("nats: error parsing client certificate: %w", err)
			}
			return cert, nil
		}
		if o.TLSConfig == nil {
			o.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
		}
		if _, err := tlsCertCB(); err != nil {
			return err
		}
		o.TLSCertCB = tlsCertCB
		o.Secure = true
		return nil
	}
}

// NoReconnect is an Option to turn off reconnect behavior.
func NoReconnect() Option {
	return func(o *Options) error {
		o.AllowReconnect = false
		return nil
	}
}

// DontRandomize is an Option to turn off randomizing the server pool.
func DontRandomize() Option {
	return func(o *Options) error {
		o.NoRandomize = true
		return nil
	}
}

// NoEcho is an Option to turn off messages echoing back from a server.
// Note this is supported on servers >= version 1.2. Proto 1 or greater.
func NoEcho() Option {
	return func(o *Options) error {
		o.NoEcho = true
		return nil
	}
}

// ReconnectWait is an Option to set the wait time between reconnect attempts.
// Defaults to 2s.
func ReconnectWait(t time.Duration) Option {
	return func(o *Options) error {
		o.ReconnectWait = t
		return nil
	}
}

// MaxReconnects is an Option to set the maximum number of reconnect attempts.
// If negative, it will never stop trying to reconnect.
// Defaults to 60.
func MaxReconnects(max int) Option {
	return func(o *Options) error {
		o.MaxReconnect = max
		return nil
	}
}

// ReconnectJitter is an Option to set the upper bound of a random delay added ReconnectWait.
// Defaults to 100ms and 1s, respectively.
func ReconnectJitter(jitter, jitterForTLS time.Duration) Option {
	return func(o *Options) error {
		o.ReconnectJitter = jitter
		o.ReconnectJitterTLS = jitterForTLS
		return nil
	}
}

// CustomReconnectDelay is an Option to set the CustomReconnectDelayCB option.
// See CustomReconnectDelayCB Option for more details.
func CustomReconnectDelay(cb ReconnectDelayHandler) Option {
	return func(o *Options) error {
		o.CustomReconnectDelayCB = cb
		return nil
	}
}

// ReconnectToServer is an Option to set a custom server selection callback
// for reconnection attempts. The callback receives a snapshot of available
// servers and must return a server from the pool to connect to and a delay
// duration before attempting the connection.
//
// See ReconnectToServerHandler for detailed documentation and usage examples.
func ReconnectToServer(cb ReconnectToServerHandler) Option {
	return func(o *Options) error {
		o.ReconnectToServerCB = cb
		return nil
	}
}

// PingInterval is an Option to set the period for client ping commands.
// Defaults to 2m.
func PingInterval(t time.Duration) Option {
	return func(o *Options) error {
		o.PingInterval = t
		return nil
	}
}

// MaxPingsOutstanding is an Option to set the maximum number of ping requests
// that can go unanswered by the server before closing the connection.
// Defaults to 2.
func MaxPingsOutstanding(max int) Option {
	return func(o *Options) error {
		o.MaxPingsOut = max
		return nil
	}
}

// ReconnectBufSize sets the buffer size of messages kept while busy reconnecting.
// Defaults to 8388608 bytes (8MB).  It can be disabled by setting it to -1.
func ReconnectBufSize(size int) Option {
	return func(o *Options) error {
		o.ReconnectBufSize = size
		return nil
	}
}

// WriteBufferSize is an advanced option that sets the flush threshold
// of the write buffer used to batch outgoing data before writing to
// the underlying connection. In most cases, the default value should
// not be changed. A smaller buffer reduces the amount of data that
// can be lost on blocked writes but may significantly reduce throughput.
// Defaults to 32768 bytes (32KB).
func WriteBufferSize(size int) Option {
	return func(o *Options) error {
		o.WriteBufferSize = size
		return nil
	}
}

// Timeout is an Option to set the timeout for Dial on a connection.
// Defaults to 2s.
func Timeout(t time.Duration) Option {
	return func(o *Options) error {
		o.Timeout = t
		return nil
	}
}

// FlusherTimeout is an Option to set the write (and flush) timeout on a connection.
func FlusherTimeout(t time.Duration) Option {
	return func(o *Options) error {
		o.FlusherTimeout = t
		return nil
	}
}

// ReconnectOnFlusherError is an Option to automatically trigger a
// reconnect when the background flusher hits any write error. See
// [Options.ReconnectOnFlusherError] for details. This is an
// advanced option and is usually not required.
func ReconnectOnFlusherError() Option {
	return func(o *Options) error {
		o.ReconnectOnFlusherError = true
		return nil
	}
}

// DrainTimeout is an Option to set the timeout for draining a connection.
// Defaults to 30s.
func DrainTimeout(t time.Duration) Option {
	return func(o *Options) error {
		o.DrainTimeout = t
		return nil
	}
}

// DisconnectErrHandler is an Option to set the disconnected error handler.
func DisconnectErrHandler(cb ConnErrHandler) Option {
	return func(o *Options) error {
		o.DisconnectedErrCB = cb
		return nil
	}
}

// DisconnectHandler is an Option to set the disconnected handler.
// Deprecated: Use DisconnectErrHandler.
func DisconnectHandler(cb ConnHandler) Option {
	return func(o *Options) error {
		o.DisconnectedCB = cb
		return nil
	}
}

// ConnectHandler is an Option to set the connected handler.
func ConnectHandler(cb ConnHandler) Option {
	return func(o *Options) error {
		o.ConnectedCB = cb
		return nil
	}
}

// ReconnectHandler is an Option to set the reconnected handler.
func ReconnectHandler(cb ConnHandler) Option {
	return func(o *Options) error {
		o.ReconnectedCB = cb
		return nil
	}
}

// ReconnectErrHandler is an Option to set the reconnect error handler.
func ReconnectErrHandler(cb ConnErrHandler) Option {
	return func(o *Options) error {
		o.ReconnectErrCB = cb
		return nil
	}
}

// ClosedHandler is an Option to set the closed handler.
func ClosedHandler(cb ConnHandler) Option {
	return func(o *Options) error {
		o.ClosedCB = cb
		return nil
	}
}

// DiscoveredServersHandler is an Option to set the new servers handler.
func DiscoveredServersHandler(cb ConnHandler) Option {
	return func(o *Options) error {
		o.DiscoveredServersCB = cb
		return nil
	}
}

// ErrorHandler is an Option to set the async error handler.
func ErrorHandler(cb ErrHandler) Option {
	return func(o *Options) error {
		o.AsyncErrorCB = cb
		return nil
	}
}

// UserInfo is an Option to set the username and password to
// use when not included directly in the URLs.
func UserInfo(user, password string) Option {
	return func(o *Options) error {
		o.User = user
		o.Password = password
		return nil
	}
}

func UserInfoHandler(cb UserInfoCB) Option {
	return func(o *Options) error {
		o.UserInfo = cb
		return nil
	}
}

// Token is an Option to set the token to use
// when a token is not included directly in the URLs
// and when a token handler is not provided.
func Token(token string) Option {
	return func(o *Options) error {
		if o.TokenHandler != nil {
			return ErrTokenAlreadySet
		}
		o.Token = token
		return nil
	}
}

// TokenHandler is an Option to set the token handler to use
// when a token is not included directly in the URLs
// and when a token is not set.
func TokenHandler(cb AuthTokenHandler) Option {
	return func(o *Options) error {
		if o.Token != "" {
			return ErrTokenAlreadySet
		}
		o.TokenHandler = cb
		return nil
	}
}

// UserCredentials is a convenience function that takes a filename
// for a user's JWT and a filename for the user's private Nkey seed.
func UserCredentials(userOrChainedFile string, seedFiles ...string) Option {
	userCB := func() (string, error) {
		return userFromFile(userOrChainedFile)
	}
	var keyFile string
	if len(seedFiles) > 0 {
		keyFile = seedFiles[0]
	} else {
		keyFile = userOrChainedFile
	}
	sigCB := func(nonce []byte) ([]byte, error) {
		return sigHandler(nonce, keyFile)
	}
	return UserJWT(userCB, sigCB)
}

// UserCredentialBytes is a convenience function that takes the JWT and seed
// values as byte slices. This allows passing credentials directly from memory
// or environment variables without needing to write them to disk.
func UserCredentialBytes(userOrChainedFileBytes []byte, seedFiles ...[]byte) Option {
	userCB := func() (string, error) {
		return nkeys.ParseDecoratedJWT(userOrChainedFileBytes)
	}

	var seedBytes []byte
	if len(seedFiles) > 0 {
		seedBytes = seedFiles[0]
	} else {
		seedBytes = userOrChainedFileBytes
	}

	sigCB := func(nonce []byte) ([]byte, error) {
		kp, err := nkeys.ParseDecoratedNKey(seedBytes)
		if err != nil {
			return nil, fmt.Errorf("unable to extract key pair from bytes: %w", err)
		}
		defer kp.Wipe()

		sig, _ := kp.Sign(nonce)
		return sig, nil
	}

	return UserJWT(userCB, sigCB)
}

// UserJWTAndSeed is a convenience function that takes the JWT and seed
// values as strings.
func UserJWTAndSeed(jwt string, seed string) Option {
	userCB := func() (string, error) {
		return jwt, nil
	}

	sigCB := func(nonce []byte) ([]byte, error) {
		kp, err := nkeys.FromSeed([]byte(seed))
		if err != nil {
			return nil, fmt.Errorf("unable to extract key pair from seed: %w", err)
		}
		// Wipe our key on exit.
		defer kp.Wipe()

		sig, _ := kp.Sign(nonce)
		return sig, nil
	}

	return UserJWT(userCB, sigCB)
}

// UserJWT will set the callbacks to retrieve the user's JWT and
// the signature callback to sign the server nonce. This an the Nkey
// option are mutually exclusive.
func UserJWT(userCB UserJWTHandler, sigCB SignatureHandler) Option {
	return func(o *Options) error {
		if userCB == nil {
			return ErrNoUserCB
		}
		if sigCB == nil {
			return ErrUserButNoSigCB
		}
		// Smoke test the user callback to ensure it is setup properly
		// when processing options.
		if _, err := userCB(); err != nil {
			return err
		}

		o.UserJWT = userCB
		o.SignatureCB = sigCB
		return nil
	}
}

// Nkey will set the public Nkey and the signature callback to
// sign the server nonce.
func Nkey(pubKey string, sigCB SignatureHandler) Option {
	return func(o *Options) error {
		o.Nkey = pubKey
		o.SignatureCB = sigCB
		if pubKey != "" && sigCB == nil {
			return ErrNkeyButNoSigCB
		}
		return nil
	}
}

// SyncQueueLen will set the maximum queue len for the internal
// channel used for SubscribeSync().
// Defaults to 65536.
func SyncQueueLen(max int) Option {
	return func(o *Options) error {
		o.SubChanLen = max
		return nil
	}
}

// Dialer is an Option to set the dialer which will be used when
// attempting to establish a connection.
// Deprecated: Should use CustomDialer instead.
func Dialer(dialer *net.Dialer) Option {
	return func(o *Options) error {
		o.Dialer = dialer
		return nil
	}
}

// SetCustomDialer is an Option to set a custom dialer which will be
// used when attempting to establish a connection. If both Dialer
// and CustomDialer are specified, CustomDialer takes precedence.
func SetCustomDialer(dialer CustomDialer) Option {
	return func(o *Options) error {
		o.CustomDialer = dialer
		return nil
	}
}

// UseOldRequestStyle is an Option to force usage of the old Request style.
func UseOldRequestStyle() Option {
	return func(o *Options) error {
		o.UseOldRequestStyle = true
		return nil
	}
}

// NoCallbacksAfterClientClose is an Option to disable callbacks when user code
// calls Close(). If close is initiated by any other condition, callbacks
// if any will be invoked.
func NoCallbacksAfterClientClose() Option {
	return func(o *Options) error {
		o.NoCallbacksAfterClientClose = true
		return nil
	}
}

// LameDuckModeHandler sets the callback to invoke when the server notifies
// the connection that it entered lame duck mode, that is, going to
// gradually disconnect all its connections before shutting down. This is
// often used in deployments when upgrading NATS Servers.
func LameDuckModeHandler(cb ConnHandler) Option {
	return func(o *Options) error {
		o.LameDuckModeHandler = cb
		return nil
	}
}

// RetryOnFailedConnect sets the connection in reconnecting state right away
// if it can't connect to a server in the initial set.
// See RetryOnFailedConnect option for more details.
func RetryOnFailedConnect(retry bool) Option {
	return func(o *Options) error {
		o.RetryOnFailedConnect = retry
		return nil
	}
}

// Compression is an Option to indicate if this connection supports
// compression. Currently only supported for Websocket connections.
func Compression(enabled bool) Option {
	return func(o *Options) error {
		o.Compression = enabled
		return nil
	}
}

// ProxyPath sets a path added to every WebSocket connection URL. This is
// useful when NATS is behind a reverse proxy that routes by path. Unlike a
// path in the connect URL, ProxyPath is also applied to server-discovered
// URLs (which are bare host:port). Use ProxyPath for clustered setups
// behind a proxy, so the client can connect to discovered servers on
// failover. When set, it overrides any path in the connection URL.
func ProxyPath(path string) Option {
	return func(o *Options) error {
		o.ProxyPath = path
		return nil
	}
}

// CustomInboxPrefix configures the request + reply inbox prefix
func CustomInboxPrefix(p string) Option {
	return func(o *Options) error {
		if p == "" || strings.Contains(p, ">") || strings.Contains(p, "*") || strings.HasSuffix(p, ".") {
			return errors.New("nats: invalid custom prefix")
		}
		o.InboxPrefix = p
		return nil
	}
}

// IgnoreAuthErrorAbort opts out of the default connect behavior of aborting
// subsequent reconnect attempts if server returns the same auth error twice.
func IgnoreAuthErrorAbort() Option {
	return func(o *Options) error {
		o.IgnoreAuthErrorAbort = true
		return nil
	}
}

// SkipHostLookup is an Option to skip the host lookup when connecting to a server.
func SkipHostLookup() Option {
	return func(o *Options) error {
		o.SkipHostLookup = true
		return nil
	}
}

func PermissionErrOnSubscribe(enabled bool) Option {
	return func(o *Options) error {
		o.PermissionErrOnSubscribe = enabled
		return nil
	}
}

// TLSHandshakeFirst is an Option to perform the TLS handshake first, that is
// before receiving the INFO protocol. This requires the server to also be
// configured with such option, otherwise the connection will fail.
func TLSHandshakeFirst() Option {
	return func(o *Options) error {
		o.TLSHandshakeFirst = true
		o.Secure = true
		return nil
	}
}

// WebSocketConnectionHeaders sets a fixed set of HTTP headers that will be
// sent during the WebSocket connection handshake.
// This option is mutually exclusive with WebSocketConnectionHeadersHandler;
// if a headers handler has already been configured, it returns
// ErrWebSocketHeadersAlreadySet.
func WebSocketConnectionHeaders(headers http.Header) Option {
	return func(o *Options) error {
		if o.WebSocketConnectionHeadersHandler != nil {
			return ErrWebSocketHeadersAlreadySet
		}
		o.WebSocketConnectionHeaders = headers
		return nil
	}
}

// WebSocketConnectionHeadersHandler registers a callback used to supply HTTP
// headers for the WebSocket connection handshake.
// This option is mutually exclusive with WebSocketConnectionHeaders; if
// non-empty static headers have already been configured, it returns
// ErrWebSocketHeadersAlreadySet.
func WebSocketConnectionHeadersHandler(cb WebSocketHeadersHandler) Option {
	return func(o *Options) error {
		if len(o.WebSocketConnectionHeaders) != 0 {
			return ErrWebSocketHeadersAlreadySet
		}
		o.WebSocketConnectionHeadersHandler = cb
		return nil
	}
}

// SkipSubjectValidation is an Option to skip subject validation when
// publishing messages.
// By default, subject validation is performed to ensure that subjects
// are valid according to NATS subject syntax (no spaces newlines and tabs).
// NOTE: It is not recommended to use this option as the performance gain
// is minimal and disabling subject validation can lead to breaking protocol
// rules.
func SkipSubjectValidation() Option {
	return func(o *Options) error {
		o.SkipSubjectValidation = true
		return nil
	}
}

// IgnoreDiscoveredServers is an Option to disable adding advertised
// server URLs from INFO messages to the server pool.
func IgnoreDiscoveredServers() Option {
	return func(o *Options) error {
		o.IgnoreDiscoveredServers = true
		return nil
	}
}

// Handler processing

// SetDisconnectHandler will set the disconnect event handler.
// Deprecated: Use SetDisconnectErrHandler
func (nc *Conn) SetDisconnectHandler(dcb ConnHandler) {
	if nc == nil {
		return
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.Opts.DisconnectedCB = dcb
}

// SetDisconnectErrHandler will set the disconnect event handler.
func (nc *Conn) SetDisconnectErrHandler(dcb ConnErrHandler) {
	if nc == nil {
		return
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.Opts.DisconnectedErrCB = dcb
}

// DisconnectErrHandler will return the disconnect event handler.
func (nc *Conn) DisconnectErrHandler() ConnErrHandler {
	if nc == nil {
		return nil
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	return nc.Opts.DisconnectedErrCB
}

// SetReconnectHandler will set the reconnect event handler.
func (nc *Conn) SetReconnectHandler(rcb ConnHandler) {
	if nc == nil {
		return
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.Opts.ReconnectedCB = rcb
}

// ReconnectHandler will return the reconnect event handler.
func (nc *Conn) ReconnectHandler() ConnHandler {
	if nc == nil {
		return nil
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	return nc.Opts.ReconnectedCB
}

// SetDiscoveredServersHandler will set the discovered servers handler.
func (nc *Conn) SetDiscoveredServersHandler(dscb ConnHandler) {
	if nc == nil {
		return
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.Opts.DiscoveredServersCB = dscb
}

// DiscoveredServersHandler will return the discovered servers handler.
func (nc *Conn) DiscoveredServersHandler() ConnHandler {
	if nc == nil {
		return nil
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	return nc.Opts.DiscoveredServersCB
}

// SetClosedHandler will set the closed event handler.
func (nc *Conn) SetClosedHandler(cb ConnHandler) {
	if nc == nil {
		return
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.Opts.ClosedCB = cb
}

// ClosedHandler will return the closed event handler.
func (nc *Conn) ClosedHandler() ConnHandler {
	if nc == nil {
		return nil
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	return nc.Opts.ClosedCB
}

// SetErrorHandler will set the async error handler.
func (nc *Conn) SetErrorHandler(cb ErrHandler) {
	if nc == nil {
		return
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.Opts.AsyncErrorCB = cb
}

// ErrorHandler will return the async error handler.
func (nc *Conn) ErrorHandler() ErrHandler {
	if nc == nil {
		return nil
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	return nc.Opts.AsyncErrorCB
}

// Process the url string argument to Connect.
// Return an array of urls, even if only one.
func processUrlString(url string) []string {
	urls := strings.Split(url, ",")
	var j int
	for _, s := range urls {
		u := strings.TrimSuffix(strings.TrimSpace(s), "/")
		if len(u) > 0 {
			urls[j] = u
			j++
		}
	}
	return urls[:j]
}

// Connect will attempt to connect to a NATS server with multiple options.
func (o Options) Connect() (*Conn, error) {
	nc := &Conn{Opts: o}

	// Some default options processing.
	if nc.Opts.MaxPingsOut == 0 {
		nc.Opts.MaxPingsOut = DefaultMaxPingOut
	}
	// Allow old default for channel length to work correctly.
	if nc.Opts.SubChanLen == 0 {
		nc.Opts.SubChanLen = DefaultMaxChanLen
	}
	// Default ReconnectBufSize
	if nc.Opts.ReconnectBufSize == 0 {
		nc.Opts.ReconnectBufSize = DefaultReconnectBufSize
	}
	// Default WriteBufferSize
	if nc.Opts.WriteBufferSize <= 0 {
		nc.Opts.WriteBufferSize = DefaultWriteBufSize
	}
	// Ensure that Timeout is not 0
	if nc.Opts.Timeout == 0 {
		nc.Opts.Timeout = DefaultTimeout
	}

	// Check first for user jwt callback being defined and nkey.
	if nc.Opts.UserJWT != nil && nc.Opts.Nkey != "" {
		return nil, ErrNkeyAndUser
	}

	// Check if we have an nkey but no signature callback defined.
	if nc.Opts.Nkey != "" && nc.Opts.SignatureCB == nil {
		return nil, ErrNkeyButNoSigCB
	}

	// Allow custom Dialer for connecting using a timeout by default
	if nc.Opts.Dialer == nil {
		nc.Opts.Dialer = &net.Dialer{
			Timeout: nc.Opts.Timeout,
		}
	}

	// If the TLSHandshakeFirst option is specified, make sure that
	// the Secure boolean is true.
	if nc.Opts.TLSHandshakeFirst {
		nc.Opts.Secure = true
	}

	if err := nc.setupServerPool(); err != nil {
		return nil, err
	}

	// Create the async callback handler.
	nc.ach = &asyncCallbacksHandler{}
	nc.ach.cond = sync.NewCond(&nc.ach.mu)

	// Set a default error handler that will print to stderr.
	if nc.Opts.AsyncErrorCB == nil {
		nc.Opts.AsyncErrorCB = defaultErrHandler
	}

	// Create reader/writer
	nc.newReaderWriter()

	// Spin up the async cb dispatcher before connect so it's ready
	// to handle callbacks, especially when RetryOnFailedConnect is used
	// and initial connection fails.
	go nc.ach.asyncCBDispatcher()

	connectionEstablished, err := nc.connect()
	if err != nil {
		nc.ach.close()
		return nil, err
	}

	if connectionEstablished {
		if connectedCB := nc.Opts.ConnectedCB; connectedCB != nil {
			nc.ach.push(func() { connectedCB(nc) })
		}
	}

	return nc, nil
}

func defaultErrHandler(nc *Conn, sub *Subscription, err error) {
	var cid uint64
	if nc != nil {
		nc.mu.RLock()
		cid = nc.info.CID
		nc.mu.RUnlock()
	}
	var errStr string
	if sub != nil {
		var subject string
		sub.mu.Lock()
		if sub.jsi != nil {
			subject = sub.jsi.psubj
		} else {
			subject = sub.Subject
		}
		sub.mu.Unlock()
		errStr = fmt.Sprintf("%s on connection [%d] for subscription on %q\n", err.Error(), cid, subject)
	} else {
		errStr = fmt.Sprintf("%s on connection [%d]\n", err.Error(), cid)
	}
	os.Stderr.WriteString(errStr)
}

const (
	_CRLF_   = "\r\n"
	_EMPTY_  = ""
	_SPC_    = " "
	_PUB_P_  = "PUB "
	_HPUB_P_ = "HPUB "
)

var _CRLF_BYTES_ = []byte(_CRLF_)

const (
	_OK_OP_   = "+OK"
	_ERR_OP_  = "-ERR"
	_PONG_OP_ = "PONG"
	_INFO_OP_ = "INFO"
)

const (
	connectProto = "CONNECT %s" + _CRLF_
	pingProto    = "PING" + _CRLF_
	pongProto    = "PONG" + _CRLF_
	subProto     = "SUB %s %s %d" + _CRLF_
	unsubProto   = "UNSUB %d %s" + _CRLF_
	okProto      = _OK_OP_ + _CRLF_
)

// Return the currently selected server
func (nc *Conn) currentServer() (int, *Server) {
	for i, s := range nc.srvPool {
		if s == nil {
			continue
		}
		if s == nc.current {
			return i, s
		}
	}
	return -1, nil
}

// Pop the current server and put onto the end of the list. Select head of list as long
// as number of reconnect attempts under MaxReconnect.
func (nc *Conn) selectNextServer() (*Server, error) {
	i, s := nc.currentServer()
	if i < 0 {
		return nil, ErrNoServers
	}
	sp := nc.srvPool
	num := len(sp)
	copy(sp[i:num-1], sp[i+1:num])
	maxReconnect := nc.Opts.MaxReconnect
	if maxReconnect < 0 || s.Reconnects < maxReconnect {
		nc.srvPool[num-1] = s
	} else {
		nc.srvPool = sp[0 : num-1]
	}
	if len(nc.srvPool) <= 0 {
		nc.current = nil
		return nil, ErrNoServers
	}
	nc.current = nc.srvPool[0]
	return nc.srvPool[0], nil
}

// Will assign the correct server to nc.current
func (nc *Conn) pickServer() error {
	nc.current = nil
	if len(nc.srvPool) <= 0 {
		return ErrNoServers
	}

	for _, s := range nc.srvPool {
		if s != nil {
			nc.current = s
			return nil
		}
	}
	return ErrNoServers
}

const tlsScheme = "tls"

// Create the server pool using the options given.
// We will place a Url option first, followed by any
// Server Options. We will randomize the server pool unless
// the NoRandomize flag is set.
func (nc *Conn) setupServerPool() error {
	nc.srvPool = make([]*Server, 0, srvPoolSize)
	nc.urls = make(map[string]struct{}, srvPoolSize)

	// Create srv objects from each url string in nc.Opts.Servers
	// and add them to the pool.
	for _, urlString := range nc.Opts.Servers {
		if err := nc.addURLToPool(urlString, false, false); err != nil {
			return err
		}
	}

	// Randomize if allowed to
	if !nc.Opts.NoRandomize {
		nc.shufflePool(0)
	}

	// Normally, if this one is set, Options.Servers should not be,
	// but we always allowed that, so continue to do so.
	if nc.Opts.Url != _EMPTY_ {
		// Add to the end of the array
		if err := nc.addURLToPool(nc.Opts.Url, false, false); err != nil {
			return err
		}
		// Then swap it with first to guarantee that Options.Url is tried first.
		last := len(nc.srvPool) - 1
		if last > 0 {
			nc.srvPool[0], nc.srvPool[last] = nc.srvPool[last], nc.srvPool[0]
		}
	} else if len(nc.srvPool) <= 0 {
		// Place default URL if pool is empty.
		if err := nc.addURLToPool(DefaultURL, false, false); err != nil {
			return err
		}
	}

	// Check for Scheme hint to move to TLS mode.
	for _, srv := range nc.srvPool {
		if srv.URL.Scheme == tlsScheme || srv.URL.Scheme == wsSchemeTLS {
			// FIXME(dlc), this is for all in the pool, should be case by case.
			nc.Opts.Secure = true
			if nc.Opts.TLSConfig == nil {
				nc.Opts.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
			}
		}
	}

	return nc.pickServer()
}

// Helper function to return scheme
func (nc *Conn) connScheme() string {
	if nc.ws {
		if nc.Opts.Secure {
			return wsSchemeTLS
		}
		return wsScheme
	}
	if nc.Opts.Secure {
		return tlsScheme
	}
	return "nats"
}

// Return true iff u.Hostname() is an IP address.
func hostIsIP(u *url.URL) bool {
	return net.ParseIP(u.Hostname()) != nil
}

// parseServerURL parses a server URL string into a Server struct.
// It handles scheme defaults and port defaults. Does not validate websocket consistency.
// Returns the parsed Server and whether it's a websocket URL.
func (nc *Conn) parseServerURL(sURL string, implicit, saveTLSName bool) (*Server, error) {
	if !strings.Contains(sURL, "://") {
		sURL = fmt.Sprintf("%s://%s", nc.connScheme(), sURL)
	}
	var (
		u   *url.URL
		err error
	)
	for i := 0; i < 2; i++ {
		u, err = url.Parse(sURL)
		if err != nil {
			return nil, err
		}
		if u.Port() != "" {
			break
		}
		// In case given URL is of the form "localhost:", just add
		// the port number at the end, otherwise, add ":4222".
		if sURL[len(sURL)-1] != ':' {
			sURL += ":"
		}
		switch u.Scheme {
		case wsScheme:
			sURL += defaultWSPortString
		case wsSchemeTLS:
			sURL += defaultWSSPortString
		default:
			sURL += defaultPortString
		}
	}

	var tlsName string
	if implicit {
		curl := nc.current.URL
		// Check to see if we do not have a url.User but current connected
		// url does. If so copy over.
		if u.User == nil && curl.User != nil {
			u.User = curl.User
		}
		// We are checking to see if we have a secure connection and are
		// adding an implicit server that just has an IP. If so we will remember
		// the current hostname we are connected to.
		if saveTLSName && hostIsIP(u) {
			tlsName = curl.Hostname()
		}
	}

	s := &Server{URL: u, isImplicit: implicit, tlsName: tlsName}
	return s, nil
}

// addURLToPool adds an entry to the server pool
func (nc *Conn) addURLToPool(sURL string, implicit, saveTLSName bool) error {
	s, err := nc.parseServerURL(sURL, implicit, saveTLSName)
	if err != nil {
		return err
	}

	// We don't support mix and match of websocket and non websocket URLs.
	// If this is the first URL, then we accept and switch the global state
	// to websocket. After that, we will know how to reject mixed URLs.
	isWS := isWebsocketScheme(s.URL)
	if len(nc.srvPool) == 0 {
		nc.ws = isWS
	} else if isWS != nc.ws {
		return ErrMixingWebsocketSchemes
	}

	nc.srvPool = append(nc.srvPool, s)
	nc.urls[s.URL.Host] = struct{}{}
	return nil
}

// shufflePool swaps randomly elements in the server pool
// The `offset` value indicates that the shuffling should start at
// this offset and leave the elements from [0..offset) intact.
func (nc *Conn) shufflePool(offset int) {
	if len(nc.srvPool) <= offset+1 {
		return
	}
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)
	for i := offset; i < len(nc.srvPool); i++ {
		j := offset + r.Intn(i+1-offset)
		nc.srvPool[i], nc.srvPool[j] = nc.srvPool[j], nc.srvPool[i]
	}
}

func (nc *Conn) newReaderWriter() {
	nc.br = &natsReader{
		buf: make([]byte, defaultBufSize),
		off: -1,
	}
	nc.bw = &natsWriter{
		limit:  nc.Opts.WriteBufferSize,
		plimit: nc.Opts.ReconnectBufSize,
	}
}

func (nc *Conn) bindToNewConn() {
	bw := nc.bw
	bw.w, bw.bufs = nc.newWriter(), nil
	br := nc.br
	br.r, br.n, br.off = nc.conn, 0, -1
}

func (nc *Conn) newWriter() io.Writer {
	var w io.Writer = nc.conn
	if nc.Opts.FlusherTimeout > 0 {
		w = &timeoutWriter{conn: nc.conn, timeout: nc.Opts.FlusherTimeout}
	}
	return w
}

func (w *natsWriter) appendString(str string) error {
	return w.appendBufs([]byte(str))
}

func (w *natsWriter) appendBufs(bufs ...[]byte) error {
	for _, buf := range bufs {
		if len(buf) == 0 {
			continue
		}
		if w.pending != nil {
			w.pending.Write(buf)
		} else {
			w.bufs = append(w.bufs, buf...)
		}
	}
	if w.pending == nil && len(w.bufs) >= w.limit {
		return w.flush()
	}
	return nil
}

func (w *natsWriter) writeDirect(strs ...string) error {
	for _, str := range strs {
		if _, err := w.w.Write([]byte(str)); err != nil {
			return err
		}
	}
	return nil
}

func (w *natsWriter) flush() error {
	// If a pending buffer is set, we don't flush. Code that needs to
	// write directly to the socket, by-passing buffers during (re)connect,
	// will use the writeDirect() API.
	if w.pending != nil {
		return nil
	}
	// Do not skip calling w.w.Write() here if len(w.bufs) is 0 because
	// the actual writer (if websocket for instance) may have things
	// to do such as sending control frames, etc..
	_, err := w.w.Write(w.bufs)
	w.bufs = w.bufs[:0]
	return err
}

func (w *natsWriter) buffered() int {
	if w.pending != nil {
		return w.pending.Len()
	}
	return len(w.bufs)
}

func (w *natsWriter) switchToPending() {
	w.pending = new(bytes.Buffer)
}

func (w *natsWriter) flushPendingBuffer() error {
	if w.pending == nil || w.pending.Len() == 0 {
		return nil
	}
	_, err := w.w.Write(w.pending.Bytes())
	// Reset the pending buffer at this point because we don't want
	// to take the risk of sending duplicates or partials.
	w.pending.Reset()
	return err
}

func (w *natsWriter) atLimitIfUsingPending() bool {
	if w.pending == nil {
		return false
	}
	return w.pending.Len() >= w.plimit
}

func (w *natsWriter) doneWithPending() {
	w.pending = nil
}

// Notify the reader that we are done with the connect, where "read" operations
// happen synchronously and under the connection lock. After this point, "read"
// will be happening from the read loop, without the connection lock.
//
// Note: this runs under the connection lock.
func (r *natsReader) doneWithConnect() {
	if wsr, ok := r.r.(*websocketReader); ok {
		wsr.doneWithConnect()
	}
}

func (r *natsReader) Read() ([]byte, error) {
	if r.off >= 0 {
		off := r.off
		r.off = -1
		return r.buf[off:r.n], nil
	}
	var err error
	r.n, err = r.r.Read(r.buf)
	return r.buf[:r.n], err
}

func (r *natsReader) ReadString(delim byte) (string, error) {
	var s string
build_string:
	// First look if we have something in the buffer
	if r.off >= 0 {
		i := bytes.IndexByte(r.buf[r.off:r.n], delim)
		if i >= 0 {
			end := r.off + i + 1
			s += string(r.buf[r.off:end])
			r.off = end
			if r.off >= r.n {
				r.off = -1
			}
			return s, nil
		}
		// We did not find the delim, so will have to read more.
		s += string(r.buf[r.off:r.n])
		r.off = -1
	}
	if _, err := r.Read(); err != nil {
		return s, err
	}
	r.off = 0
	goto build_string
}

// createConn will connect to the server and wrap the appropriate
// bufio structures. It will do the right thing when an existing
// connection is in place.
func (nc *Conn) createConn() (err error) {
	if nc.Opts.Timeout < 0 {
		return ErrBadTimeout
	}
	if _, cur := nc.currentServer(); cur == nil {
		return ErrNoServers
	}

	// If we have a reference to an in-process server then establish a
	// connection using that.
	if nc.Opts.InProcessServer != nil {
		conn, err := nc.Opts.InProcessServer.InProcessConn()
		if err != nil {
			return fmt.Errorf("failed to get in-process connection: %w", err)
		}
		nc.conn = conn
		nc.bindToNewConn()
		return nil
	}

	// We will auto-expand host names if they resolve to multiple IPs
	hosts := []string{}
	u := nc.current.URL

	if !nc.Opts.SkipHostLookup && net.ParseIP(u.Hostname()) == nil {
		addrs, _ := net.LookupHost(u.Hostname())
		for _, addr := range addrs {
			hosts = append(hosts, net.JoinHostPort(addr, u.Port()))
		}
	}
	// Fall back to what we were given.
	if len(hosts) == 0 {
		hosts = append(hosts, u.Host)
	}

	// CustomDialer takes precedence. If not set, use Opts.Dialer which
	// is set to a default *net.Dialer (in Connect()) if not explicitly
	// set by the user.
	dialer := nc.Opts.CustomDialer
	if dialer == nil {
		// We will copy and shorten the timeout if we have multiple hosts to try.
		copyDialer := *nc.Opts.Dialer
		copyDialer.Timeout = copyDialer.Timeout / time.Duration(len(hosts))
		dialer = &copyDialer
	}

	if len(hosts) > 1 && !nc.Opts.NoRandomize {
		rand.Shuffle(len(hosts), func(i, j int) {
			hosts[i], hosts[j] = hosts[j], hosts[i]
		})
	}
	for _, host := range hosts {
		nc.conn, err = dialer.Dial("tcp", host)
		if err == nil {
			break
		}
	}
	if err != nil {
		return err
	}

	// If scheme starts with "ws" then branch out to websocket code.
	if isWebsocketScheme(u) {
		return nc.wsInitHandshake(u)
	}

	// Reset reader/writer to this new TCP connection
	nc.bindToNewConn()
	return nil
}

type skipTLSDialer interface {
	SkipTLSHandshake() bool
}

// makeTLSConn will wrap an existing Conn using TLS
func (nc *Conn) makeTLSConn() error {
	if nc.Opts.CustomDialer != nil {
		// we do nothing when asked to skip the TLS wrapper
		sd, ok := nc.Opts.CustomDialer.(skipTLSDialer)
		if ok && sd.SkipTLSHandshake() {
			return nil
		}
	}
	// Allow the user to configure their own tls.Config structure.
	tlsCopy := &tls.Config{}
	if nc.Opts.TLSConfig != nil {
		tlsCopy = util.CloneTLSConfig(nc.Opts.TLSConfig)
	}
	if nc.Opts.TLSCertCB != nil {
		cert, err := nc.Opts.TLSCertCB()
		if err != nil {
			return err
		}
		tlsCopy.Certificates = []tls.Certificate{cert}
	}
	if nc.Opts.RootCAsCB != nil {
		rootCAs, err := nc.Opts.RootCAsCB()
		if err != nil {
			return err
		}
		tlsCopy.RootCAs = rootCAs
	}
	// If its blank we will override it with the current host
	if tlsCopy.ServerName == _EMPTY_ {
		if nc.current.tlsName != _EMPTY_ {
			tlsCopy.ServerName = nc.current.tlsName
		} else {
			h, _, _ := net.SplitHostPort(nc.current.URL.Host)
			tlsCopy.ServerName = h
		}
	}
	nc.conn = tls.Client(nc.conn, tlsCopy)
	conn := nc.conn.(*tls.Conn)
	if err := conn.Handshake(); err != nil {
		return fmt.Errorf("%w: %w", ErrTLS, err)
	}
	nc.bindToNewConn()
	return nil
}

// TLSConnectionState retrieves the state of the TLS connection to the server
func (nc *Conn) TLSConnectionState() (tls.ConnectionState, error) {
	if !nc.isConnected() {
		return tls.ConnectionState{}, ErrDisconnected
	}

	nc.mu.RLock()
	conn := nc.conn
	nc.mu.RUnlock()

	tc, ok := conn.(*tls.Conn)
	if !ok {
		return tls.ConnectionState{}, ErrConnectionNotTLS
	}

	return tc.ConnectionState(), nil
}

// waitForExits will wait for all socket watcher Go routines to
// be shutdown before proceeding.
func (nc *Conn) waitForExits() {
	// Kick old flusher forcefully.
	select {
	case nc.fch <- struct{}{}:
	default:
	}

	// Wait for any previous go routines.
	nc.wg.Wait()
}

// ForceReconnect forces a reconnect attempt to the server.
// This is a non-blocking call and will start the reconnect
// process without waiting for it to complete.
//
// If the connection is already in the process of reconnecting,
// this call will force an immediate reconnect attempt (bypassing
// the current reconnect delay).
func (nc *Conn) ForceReconnect() error {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	if nc.isClosed() {
		return ErrConnectionClosed
	}
	if nc.isReconnecting() {
		// if we're already reconnecting, force a reconnect attempt
		// even if we're in the middle of a backoff
		if nc.rqch != nil {
			close(nc.rqch)
			nc.rqch = nil
		}
		return nil
	}

	// Clear any queued pongs
	nc.clearPendingFlushCalls()

	// Clear any queued and blocking requests.
	nc.clearPendingRequestCalls()

	// Stop ping timer if set.
	nc.stopPingTimer()

	// flush any pending data and switch to pending mode to buffer new outgoing
	// data until we reconnect and can flush it.
	nc.bw.flush()
	nc.bw.switchToPending()
	nc.conn.Close()

	nc.changeConnStatus(RECONNECTING)
	go nc.doReconnect(nil, true)
	return nil
}

// ConnectedUrl reports the connected server's URL
func (nc *Conn) ConnectedUrl() string {
	if nc == nil {
		return _EMPTY_
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return _EMPTY_
	}
	return nc.current.URL.String()
}

// ConnectedUrlRedacted reports the connected server's URL with passwords redacted
func (nc *Conn) ConnectedUrlRedacted() string {
	if nc == nil {
		return _EMPTY_
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return _EMPTY_
	}
	return nc.current.URL.Redacted()
}

// ConnectedAddr returns the connected server's IP
func (nc *Conn) ConnectedAddr() string {
	if nc == nil {
		return _EMPTY_
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return _EMPTY_
	}
	return nc.conn.RemoteAddr().String()
}

// LocalAddr returns the local network address of the connection
func (nc *Conn) LocalAddr() string {
	if nc == nil {
		return _EMPTY_
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return _EMPTY_
	}
	return nc.conn.LocalAddr().String()
}

// ConnectedServerId reports the connected server's Id
func (nc *Conn) ConnectedServerId() string {
	if nc == nil {
		return _EMPTY_
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return _EMPTY_
	}
	return nc.info.ID
}

// ConnectedServerName reports the connected server's name
func (nc *Conn) ConnectedServerName() string {
	if nc == nil {
		return _EMPTY_
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return _EMPTY_
	}
	return nc.info.Name
}

var semVerRe = regexp.MustCompile(`\Av?([0-9]+)\.?([0-9]+)?\.?([0-9]+)?`)

func versionComponents(version string) (major, minor, patch int, err error) {
	m := semVerRe.FindStringSubmatch(version)
	if m == nil {
		return 0, 0, 0, errors.New("invalid semver")
	}
	major, err = strconv.Atoi(m[1])
	if err != nil {
		return -1, -1, -1, err
	}
	minor, err = strconv.Atoi(m[2])
	if err != nil {
		return -1, -1, -1, err
	}
	patch, err = strconv.Atoi(m[3])
	if err != nil {
		return -1, -1, -1, err
	}
	return major, minor, patch, err
}

// Check for minimum server requirement.
func (nc *Conn) serverMinVersion(major, minor, patch int) bool {
	smajor, sminor, spatch, _ := versionComponents(nc.ConnectedServerVersion())
	if smajor < major || (smajor == major && sminor < minor) || (smajor == major && sminor == minor && spatch < patch) {
		return false
	}
	return true
}

// ConnectedServerVersion reports the connected server's version as a string
func (nc *Conn) ConnectedServerVersion() string {
	if nc == nil {
		return _EMPTY_
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return _EMPTY_
	}
	return nc.info.Version
}

// ConnectedClusterName reports the connected server's cluster name if any
func (nc *Conn) ConnectedClusterName() string {
	if nc == nil {
		return _EMPTY_
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return _EMPTY_
	}
	return nc.info.Cluster
}

// ConnectedServerJetStream reports whether the connected server has
// JetStream enabled and, if so, its API level. The API level is
// advertised by nats-server v2.12.0 or later; older servers will
// report 0 even when JetStream is enabled.
func (nc *Conn) ConnectedServerJetStream() (bool, int) {
	if nc == nil {
		return false, 0
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return false, 0
	}
	return nc.info.JetStream, nc.info.JSApiLevel
}

// IsSystemAccount reports whether the connected client's account
// is the system account.
func (nc *Conn) IsSystemAccount() bool {
	if nc == nil {
		return false
	}

	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.status != CONNECTED {
		return false
	}
	return nc.info.IsSystemAccount
}

// Low level setup for structs, etc
func (nc *Conn) setup() {
	nc.subs = make(map[int64]*Subscription)
	nc.pongs = make([]chan struct{}, 0, 8)

	nc.fch = make(chan struct{}, flushChanSize)
	nc.rqch = make(chan struct{})

	// Setup scratch outbound buffer for PUB/HPUB
	pub := nc.scratch[:len(_HPUB_P_)]
	copy(pub, _HPUB_P_)
}

// tlsHandshakeEOF wraps an error with context when it occurs right after
// a completed TLS handshake, which typically indicates the remote side
// rejected the client certificate (e.g. an mTLS proxy like nginx).
// Depending on timing, the error may be io.EOF (read from closed conn)
// or a "broken pipe"/"connection reset" (write to closed conn).
func (nc *Conn) tlsHandshakeEOF(err error) error {
	tlsConn, ok := nc.conn.(*tls.Conn)
	if !ok || !tlsConn.ConnectionState().HandshakeComplete {
		return err
	}
	if errors.Is(err, io.EOF) || isConnClosedError(err) {
		return fmt.Errorf("%w: connection closed by remote after TLS handshake: %w", ErrTLS, err)
	}
	return err
}

// isConnClosedError reports whether the error indicates the remote
// side closed the connection (broken pipe or connection reset).
// NOTE: On Windows, connection resets use WSAECONNRESET which may not
// match syscall.ECONNRESET. In that case, the error will not be
// detected here but will still be returned unwrapped by the caller.
func isConnClosedError(err error) bool {
	return errors.Is(err, syscall.EPIPE) || errors.Is(err, syscall.ECONNRESET)
}

// Process a connected connection and initialize properly.
func (nc *Conn) processConnectInit() error {
	// Set our deadline for the whole connect process
	nc.conn.SetDeadline(time.Now().Add(nc.Opts.Timeout))
	defer nc.conn.SetDeadline(time.Time{})

	// Set our status to connecting.
	nc.changeConnStatus(CONNECTING)

	// If we need to have a TLS connection and want the TLS handshake to occur
	// first, do it now.
	if nc.Opts.Secure && nc.Opts.TLSHandshakeFirst {
		if err := nc.makeTLSConn(); err != nil {
			return err
		}
	}

	// Process the INFO protocol received from the server
	err := nc.processExpectedInfo()
	if err != nil {
		return nc.tlsHandshakeEOF(err)
	}

	// Send the CONNECT protocol along with the initial PING protocol.
	// Wait for the PONG response (or any error that we get from the server).
	err = nc.sendConnect()
	if err != nil {
		return nc.tlsHandshakeEOF(err)
	}

	// Reset the number of PING sent out
	nc.pout = 0

	// Start or reset Timer
	if nc.Opts.PingInterval > 0 {
		if nc.ptmr == nil {
			nc.ptmr = time.AfterFunc(nc.Opts.PingInterval, nc.processPingTimer)
		} else {
			nc.ptmr.Reset(nc.Opts.PingInterval)
		}
	}

	// Start the readLoop and flusher go routines, we will wait on both on a reconnect event.
	nc.wg.Add(2)
	go nc.readLoop()
	go nc.flusher()

	// Notify the reader that we are done with the connect handshake, where
	// reads were done synchronously and under the connection lock.
	nc.br.doneWithConnect()

	return nil
}

// Main connect function. Will connect to the nats-server.
func (nc *Conn) connect() (bool, error) {
	var err error
	var connectionEstablished bool

	// Create actual socket connection
	// For first connect we walk all servers in the pool and try
	// to connect immediately.
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.initc = true
	// The pool may change inside the loop iteration due to INFO protocol.
	for i := 0; i < len(nc.srvPool); i++ {
		nc.current = nc.srvPool[i]

		if err = nc.createConn(); err == nil {
			// This was moved out of processConnectInit() because
			// that function is now invoked from doReconnect() too.
			nc.setup()

			err = nc.processConnectInit()

			if err == nil {
				nc.current.didConnect = true
				nc.current.Reconnects = 0
				nc.current.lastErr = nil
				break
			} else {
				nc.mu.Unlock()
				nc.close(DISCONNECTED, false, err)
				nc.mu.Lock()
				// Do not reset nc.current here since it would prevent
				// RetryOnFailedConnect to work should this be the last server
				// to try before starting doReconnect().
			}
		} else {
			// Cancel out default connection refused, will trigger the
			// No servers error conditional
			if strings.Contains(err.Error(), "connection refused") {
				err = nil
			}
		}
	}

	if err == nil && nc.status != CONNECTED {
		err = ErrNoServers
	}

	if err == nil {
		connectionEstablished = true
		nc.initc = false
	} else if nc.Opts.RetryOnFailedConnect {
		nc.setup()
		nc.changeConnStatus(RECONNECTING)
		nc.bw.switchToPending()
		go nc.doReconnect(err, false)
		err = nil
	} else {
		nc.current = nil
	}

	return connectionEstablished, err
}

// This will check to see if the connection should be
// secure. This can be dictated from either end and should
// only be called after the INIT protocol has been received.
func (nc *Conn) checkForSecure() error {
	// Check to see if we need to engage TLS
	o := nc.Opts

	// Check for mismatch in setups
	if o.Secure && !nc.info.TLSRequired && !nc.info.TLSAvailable {
		return ErrSecureConnWanted
	} else if nc.info.TLSRequired && !o.Secure {
		// Switch to Secure since server needs TLS.
		o.Secure = true
	}

	if o.Secure {
		// If TLS handshake first is true, we have already done
		// the handshake, so we are done here.
		if o.TLSHandshakeFirst {
			return nil
		}
		// Need to rewrap with bufio
		if err := nc.makeTLSConn(); err != nil {
			return err
		}
	}
	return nil
}

// processExpectedInfo will look for the expected first INFO message
// sent when a connection is established. The lock should be held entering.
func (nc *Conn) processExpectedInfo() error {
	c := &control{}

	// Read the protocol
	err := nc.readOp(c)
	if err != nil {
		return err
	}

	// The nats protocol should send INFO first always.
	if c.op != _INFO_OP_ {
		return ErrNoInfoReceived
	}

	// Parse the protocol
	if err := nc.processInfo(c.args); err != nil {
		return err
	}

	if nc.Opts.Nkey != "" && nc.info.Nonce == "" {
		return ErrNkeysNotSupported
	}

	// For websocket connections, we already switched to TLS if need be,
	// so we are done here.
	if nc.ws {
		return nil
	}

	return nc.checkForSecure()
}

// Sends a protocol control message by queuing into the bufio writer
// and kicking the flush Go routine.  These writes are protected.
func (nc *Conn) sendProto(proto string) {
	nc.mu.Lock()
	nc.bw.appendString(proto)
	nc.kickFlusher()
	nc.mu.Unlock()
}

// Generate a connect protocol message, issuing user/password if
// applicable. The lock is assumed to be held upon entering.
func (nc *Conn) connectProto() (string, error) {
	o := nc.Opts
	var nkey, sig, user, pass, token, ujwt string
	u := nc.current.URL.User
	if u != nil {
		// if no password, assume username is authToken
		if _, ok := u.Password(); !ok {
			token = u.Username()
		} else {
			user = u.Username()
			pass, _ = u.Password()
		}
	} else {
		// Take from options (possibly all empty strings)
		user = o.User
		pass = o.Password
		token = o.Token
		nkey = o.Nkey

		if nc.Opts.UserInfo != nil {
			if user != _EMPTY_ || pass != _EMPTY_ {
				return _EMPTY_, ErrUserInfoAlreadySet
			}
			user, pass = nc.Opts.UserInfo()
		}
	}

	// Look for user jwt.
	if o.UserJWT != nil {
		if jwt, err := o.UserJWT(); err != nil {
			return _EMPTY_, err
		} else {
			ujwt = jwt
		}
		if nkey != _EMPTY_ {
			return _EMPTY_, ErrNkeyAndUser
		}
	}

	if ujwt != _EMPTY_ || nkey != _EMPTY_ {
		if o.SignatureCB == nil {
			if ujwt == _EMPTY_ {
				return _EMPTY_, ErrNkeyButNoSigCB
			}
			return _EMPTY_, ErrUserButNoSigCB
		}
		sigraw, err := o.SignatureCB([]byte(nc.info.Nonce))
		if err != nil {
			return _EMPTY_, fmt.Errorf("error signing nonce: %w", err)
		}
		sig = base64.RawURLEncoding.EncodeToString(sigraw)
	}

	if nc.Opts.TokenHandler != nil {
		if token != _EMPTY_ {
			return _EMPTY_, ErrTokenAlreadySet
		}
		token = nc.Opts.TokenHandler()
	}

	// If our server does not support headers then we can't do them or no responders.
	hdrs := nc.info.Headers
	cinfo := connectInfo{
		o.Verbose, o.Pedantic, ujwt, nkey, sig, user, pass, token,
		o.Secure, o.Name, LangString, Version, clientProtoInfo, !o.NoEcho, hdrs, hdrs,
	}

	b, err := json.Marshal(cinfo)
	if err != nil {
		return _EMPTY_, ErrJsonParse
	}

	// Check if NoEcho is set and we have a server that supports it.
	if o.NoEcho && nc.info.Proto < 1 {
		return _EMPTY_, ErrNoEchoNotSupported
	}

	return fmt.Sprintf(connectProto, b), nil
}

// normalizeErr removes the prefix -ERR, trim spaces and remove the quotes.
func normalizeErr(line string) string {
	s := strings.TrimSpace(strings.TrimPrefix(line, _ERR_OP_))
	s = strings.TrimLeft(strings.TrimRight(s, "'"), "'")
	return s
}

// natsProtoErr represents an -ERR protocol message sent by the server.
type natsProtoErr struct {
	description string
}

func (nerr *natsProtoErr) Error() string {
	return fmt.Sprintf("nats: %s", nerr.description)
}

func (nerr *natsProtoErr) Is(err error) bool {
	return strings.ToLower(nerr.Error()) == err.Error()
}

// Send a connect protocol message to the server, issue user/password if
// applicable. Will wait for a flush to return from the server for error
// processing.
func (nc *Conn) sendConnect() error {
	// Construct the CONNECT protocol string
	cProto, err := nc.connectProto()
	if err != nil {
		if !nc.initc {
			if asyncErrorCB := nc.Opts.AsyncErrorCB; asyncErrorCB != nil {
				nc.ach.push(func() { asyncErrorCB(nc, nil, err) })
			}
		}
		return err
	}

	// Write the protocol and PING directly to the underlying writer.
	if err := nc.bw.writeDirect(cProto, pingProto); err != nil {
		return err
	}

	// We don't want to read more than we need here, otherwise
	// we would need to transfer the excess read data to the readLoop.
	// Since in normal situations we just are looking for a PONG\r\n,
	// reading byte-by-byte here is ok.
	proto, err := nc.readProto()
	if err != nil {
		if !nc.initc {
			if asyncErrorCB := nc.Opts.AsyncErrorCB; asyncErrorCB != nil {
				nc.ach.push(func() { asyncErrorCB(nc, nil, err) })
			}
		}
		return err
	}

	// If opts.Verbose is set, handle +OK
	if nc.Opts.Verbose && proto == okProto {
		// Read the rest now...
		proto, err = nc.readProto()
		if err != nil {
			if !nc.initc {
				if asyncErrorCB := nc.Opts.AsyncErrorCB; asyncErrorCB != nil {
					nc.ach.push(func() { asyncErrorCB(nc, nil, err) })
				}
			}
			return err
		}
	}

	// We expect a PONG
	if proto != pongProto {
		// But it could be something else, like -ERR

		// Since we no longer use ReadLine(), trim the trailing "\r\n"
		proto = strings.TrimRight(proto, "\r\n")

		// If it's a server error...
		if strings.HasPrefix(proto, _ERR_OP_) {
			// Remove -ERR, trim spaces and quotes, and convert to lower case.
			proto = normalizeErr(proto)

			// Check if this is an auth error
			if authErr := checkAuthError(strings.ToLower(proto)); authErr != nil {
				// This will schedule an async error if we are in reconnect,
				// and keep track of the auth error for the current server.
				// If we have got the same error twice, this sets nc.ar to true to
				// indicate that the reconnect should be aborted (will be checked
				// in doReconnect()).
				nc.processAuthError(authErr)
			}
			return &natsProtoErr{proto}
		}

		// Notify that we got an unexpected protocol.
		return fmt.Errorf("nats: expected '%s', got '%s'", _PONG_OP_, proto)
	}

	// This is where we are truly connected.
	nc.changeConnStatus(CONNECTED)

	return nil
}

// reads a protocol line.
func (nc *Conn) readProto() (string, error) {
	return nc.br.ReadString('\n')
}

// A control protocol line.
type control struct {
	op, args string
}

// Read a control line and process the intended op.
func (nc *Conn) readOp(c *control) error {
	line, err := nc.readProto()
	if err != nil {
		return err
	}
	parseControl(line, c)
	return nil
}

// Parse a control line from the server.
func parseControl(line string, c *control) {
	toks := strings.SplitN(line, _SPC_, 2)
	if len(toks) == 1 {
		c.op = strings.TrimSpace(toks[0])
		c.args = _EMPTY_
	} else if len(toks) == 2 {
		c.op, c.args = strings.TrimSpace(toks[0]), strings.TrimSpace(toks[1])
	} else {
		c.op = _EMPTY_
	}
}

// flushReconnectPendingItems will push the pending items that were
// gathered while we were in a RECONNECTING state to the socket.
func (nc *Conn) flushReconnectPendingItems() error {
	return nc.bw.flushPendingBuffer()
}

// Stops the ping timer if set.
// Connection lock is held on entry.
func (nc *Conn) stopPingTimer() {
	if nc.ptmr != nil {
		nc.ptmr.Stop()
	}
}

// Try to reconnect using the option parameters.
// This function assumes we are allowed to reconnect.
func (nc *Conn) doReconnect(err error, forceReconnect bool) {
	// We want to make sure we have the other watchers shutdown properly
	// here before we proceed past this point.
	nc.waitForExits()

	// FIXME(dlc) - We have an issue here if we have
	// outstanding flush points (pongs) and they were not
	// sent out, but are still in the pipe.

	// Hold the lock manually and release where needed below,
	// can't do defer here.
	nc.mu.Lock()

	// Clear any errors.
	nc.err = nil

	// Perform appropriate callback if needed for a disconnect.
	// DisconnectedErrCB has priority over deprecated DisconnectedCB
	if !nc.initc {
		if disconnectedErrCB := nc.Opts.DisconnectedErrCB; disconnectedErrCB != nil {
			nc.ach.push(func() { disconnectedErrCB(nc, err) })
		} else if disconnectedCB := nc.Opts.DisconnectedCB; disconnectedCB != nil {
			nc.ach.push(func() { disconnectedCB(nc) })
		}
	} else if nc.Opts.RetryOnFailedConnect && nc.initc && err != nil {
		// For initial connection failure with RetryOnFailedConnect,
		// report the error via ReconnectErrCB if available
		if nc.Opts.ReconnectErrCB != nil {
			nc.ach.push(func() { nc.Opts.ReconnectErrCB(nc, err) })
		}
	}

	// This is used to wait on go routines exit if we start them in the loop
	// but an error occurs after that.
	waitForGoRoutines := false
	var rt *time.Timer
	// Channel used to kick routine out of sleep when conn is closed.
	rqch := nc.rqch

	// if rqch is nil, we need to set it up to signal
	// the reconnect loop to reconnect immediately
	// this means that `ForceReconnect` was called
	// before entering doReconnect
	if rqch == nil {
		rqch = make(chan struct{})
		close(rqch)
	}

	// Counter that is increased when the whole list of servers has been tried.
	var wlf int

	var jitter time.Duration
	var rw time.Duration
	// If a custom reconnect delay handler is set, this takes precedence.
	crd := nc.Opts.CustomReconnectDelayCB
	if crd == nil {
		rw = nc.Opts.ReconnectWait
		// TODO: since we sleep only after the whole list has been tried, we can't
		// rely on individual *srv to know if it is a TLS or non-TLS url.
		// We have to pick which type of jitter to use, for now, we use these hints:
		jitter = nc.Opts.ReconnectJitter
		if nc.Opts.Secure || nc.Opts.TLSConfig != nil {
			jitter = nc.Opts.ReconnectJitterTLS
		}
	}

	for i := 0; len(nc.srvPool) > 0; {
		var err error
		var cur *Server
		var callbackDelay time.Duration
		var useCallbackDelay bool

		if nc.Opts.ReconnectToServerCB != nil {
			// Enforce MaxReconnect limits before calling the callback
			maxReconnect := nc.Opts.MaxReconnect
			if maxReconnect >= 0 {
				// Remove servers that have exceeded MaxReconnect attempts
				filtered := make([]*Server, 0, len(nc.srvPool))
				for _, srv := range nc.srvPool {
					if srv != nil && srv.Reconnects < maxReconnect {
						filtered = append(filtered, srv)
					}
				}
				nc.srvPool = filtered
			}

			// Check if we still have servers after filtering
			if len(nc.srvPool) == 0 {
				nc.err = ErrNoServers
				break
			}

			// Copy server values to avoid caller modifying internal state
			srvVals := make([]Server, len(nc.srvPool))
			for idx, srv := range nc.srvPool {
				if srv != nil {
					srvVals[idx] = srv.clone()
				}
			}

			var selectedSrv *Server
			selectedSrv, callbackDelay = nc.Opts.ReconnectToServerCB(srvVals, nc.info)
			if selectedSrv != nil {
				idx := slices.IndexFunc(nc.srvPool, func(srv *Server) bool {
					return srv != nil && srv.URL.String() == selectedSrv.URL.String()
				})
				if idx != -1 {
					cur = nc.srvPool[idx]
					nc.current = cur
					useCallbackDelay = true
				} else if reconnectErrCB := nc.Opts.ReconnectErrCB; reconnectErrCB != nil {
					nc.ach.push(func() { reconnectErrCB(nc, ErrServerNotInPool) })
				}
			}
		}

		var doSleep bool
		if cur == nil {
			cur, err = nc.selectNextServer()
			if err != nil {
				nc.err = err
				break
			}
			doSleep = i+1 >= len(nc.srvPool) && !forceReconnect
		}

		forceReconnect = false
		nc.mu.Unlock()

		if useCallbackDelay {
			if callbackDelay > 0 {
				i = 0
				if rt == nil {
					rt = time.NewTimer(callbackDelay)
				} else {
					rt.Reset(callbackDelay)
				}
				select {
				case <-rqch:
					rt.Stop()
					nc.mu.Lock()
					nc.rqch = make(chan struct{})
					nc.mu.Unlock()
				case <-rt.C:
				}
			} else {
				runtime.Gosched()
			}
		} else if !doSleep {
			i++
			// Release the lock to give a chance to a concurrent nc.Close() to break the loop.
			runtime.Gosched()
		} else {
			i = 0
			var st time.Duration
			if crd != nil {
				wlf++
				st = crd(wlf)
			} else {
				st = rw
				if jitter > 0 {
					st += time.Duration(rand.Int63n(int64(jitter)))
				}
			}
			if rt == nil {
				rt = time.NewTimer(st)
			} else {
				rt.Reset(st)
			}
			select {
			case <-rqch:
				rt.Stop()

				// we need to reset the rqch channel to avoid
				// closing a closed channel in the next iteration
				nc.mu.Lock()
				nc.rqch = make(chan struct{})
				nc.mu.Unlock()
			case <-rt.C:
			}
		}
		// If the readLoop, etc.. go routines were started, wait for them to complete.
		if waitForGoRoutines {
			nc.waitForExits()
			waitForGoRoutines = false
		}
		nc.mu.Lock()

		// Check if we have been closed first.
		if nc.isClosed() {
			break
		}

		// Mark that we tried a reconnect
		cur.Reconnects++

		// Try to create a new connection
		err = nc.createConn()
		// Not yet connected, retry...
		// Continue to hold the lock
		if err != nil {
			// Perform appropriate callback for a failed connection attempt.
			if reconnectErrCB := nc.Opts.ReconnectErrCB; reconnectErrCB != nil {
				nc.ach.push(func() { reconnectErrCB(nc, err) })
			}
			nc.err = nil
			continue
		}

		// We are reconnected
		nc.Reconnects++

		// Process connect logic
		if nc.err = nc.processConnectInit(); nc.err != nil {
			// Check if we should abort reconnect. If so, break out
			// of the loop and connection will be closed.
			if nc.ar {
				break
			}
			nc.changeConnStatus(RECONNECTING)
			continue
		}

		// Clear possible lastErr under the connection lock after
		// a successful processConnectInit().
		nc.current.lastErr = nil

		// Clear out server stats for the server we connected to..
		cur.didConnect = true
		cur.Reconnects = 0

		// Send existing subscription state
		nc.resendSubscriptions()

		// Now send off and clear pending buffer
		nc.err = nc.flushReconnectPendingItems()
		if nc.err != nil {
			nc.changeConnStatus(RECONNECTING)
			// Stop the ping timer (if set)
			nc.stopPingTimer()
			// Since processConnectInit() returned without error, the
			// go routines were started, so wait for them to return
			// on the next iteration (after releasing the lock).
			waitForGoRoutines = true
			continue
		}

		// Done with the pending buffer
		nc.bw.doneWithPending()

		// Queue up the correct callback. If we are in initial connect state
		// (using retry on failed connect), we will call the ConnectedCB,
		// otherwise the ReconnectedCB.
		if reconnectedCB := nc.Opts.ReconnectedCB; reconnectedCB != nil && !nc.initc {
			nc.ach.push(func() { reconnectedCB(nc) })
		} else if connectedCB := nc.Opts.ConnectedCB; connectedCB != nil && nc.initc {
			nc.ach.push(func() { connectedCB(nc) })
		}

		// If we are here with a retry on failed connect, indicate that the
		// initial connect is now complete.
		nc.initc = false

		// Release lock here, we will return below.
		nc.mu.Unlock()

		// Make sure to flush everything
		nc.Flush()

		return
	}

	// Call into close.. We have no servers left..
	if nc.err == nil {
		nc.err = ErrNoServers
	}
	nc.mu.Unlock()
	nc.close(CLOSED, true, nil)
}

// processOpErr handles errors from reading or parsing the protocol.
// The lock should not be held entering this function. If forceReconnect
// is true, the first reconnect attempt will bypass the configured
// ReconnectWait; subsequent attempts still obey the normal backoff.
func (nc *Conn) processOpErr(err error, forceReconnect bool) bool {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if nc.isConnecting() || nc.isClosed() || nc.isReconnecting() {
		return false
	}

	if nc.Opts.AllowReconnect && nc.status == CONNECTED {
		// Set our new status
		nc.changeConnStatus(RECONNECTING)
		// Stop ping timer if set
		nc.stopPingTimer()
		if nc.conn != nil {
			nc.conn.Close()
			nc.conn = nil
		}

		// Create pending buffer before reconnecting.
		nc.bw.switchToPending()

		// Clear any queued pongs, e.g. pending flush calls.
		nc.clearPendingFlushCalls()

		go nc.doReconnect(err, forceReconnect)
		return false
	}

	nc.changeConnStatus(DISCONNECTED)
	nc.err = err
	return true
}

// dispatch is responsible for calling any async callbacks
func (ac *asyncCallbacksHandler) asyncCBDispatcher() {
	for {
		ac.mu.Lock()
		// Protect for spurious wakeups. We should get out of the
		// wait only if there is an element to pop from the list.
		for ac.head == nil {
			ac.cond.Wait()
		}
		cur := ac.head
		ac.head = cur.next
		if cur == ac.tail {
			ac.tail = nil
		}
		ac.mu.Unlock()

		// This signals that the dispatcher has been closed and all
		// previous callbacks have been dispatched.
		if cur.f == nil {
			return
		}
		// Invoke callback outside of handler's lock
		cur.f()
	}
}

// Add the given function to the tail of the list and
// signals the dispatcher.
func (ac *asyncCallbacksHandler) push(f func()) {
	ac.pushOrClose(f, false)
}

// Signals that we are closing...
func (ac *asyncCallbacksHandler) close() {
	ac.pushOrClose(nil, true)
}

// Add the given function to the tail of the list and
// signals the dispatcher.
func (ac *asyncCallbacksHandler) pushOrClose(f func(), close bool) {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	// Make sure that library is not calling push with nil function,
	// since this is used to notify the dispatcher that it should stop.
	if !close && f == nil {
		panic("pushing a nil callback")
	}
	cb := &asyncCB{f: f}
	if ac.tail != nil {
		ac.tail.next = cb
	} else {
		ac.head = cb
	}
	ac.tail = cb
	if close {
		ac.cond.Broadcast()
	} else {
		ac.cond.Signal()
	}
}

// readLoop() will sit on the socket reading and processing the
// protocol from the server. It will dispatch appropriately based
// on the op type.
func (nc *Conn) readLoop() {
	// Release the wait group on exit
	defer nc.wg.Done()

	// Create a parseState if needed.
	nc.mu.Lock()
	if nc.ps == nil {
		nc.ps = &parseState{}
	}
	conn := nc.conn
	br := nc.br
	nc.mu.Unlock()

	if conn == nil {
		return
	}

	for {
		buf, err := br.Read()
		if err == nil {
			// With websocket, it is possible that there is no error but
			// also no buffer returned (either WS control message or read of a
			// partial compressed message). We could call parse(buf) which
			// would ignore an empty buffer, but simply go back to top of the loop.
			if len(buf) == 0 {
				continue
			}
			err = nc.parse(buf)
		}
		if err != nil {
			if shouldClose := nc.processOpErr(err, false); shouldClose {
				nc.close(CLOSED, true, nil)
			}
			break
		}
	}
	// Clear the parseState here..
	nc.mu.Lock()
	nc.ps = nil
	nc.mu.Unlock()
}

// waitForMsgs waits on the conditional shared with readLoop and processMsg.
// It is used to deliver messages to asynchronous subscribers.
func (nc *Conn) waitForMsgs(s *Subscription) {
	var closed bool
	var delivered, max uint64

	// Used to account for adjustments to sub.pBytes when we wrap back around.
	msgLen := -1

	for {
		s.mu.Lock()
		// Do accounting for last msg delivered here so we only lock once
		// and drain state trips after callback has returned.
		if msgLen >= 0 {
			s.pMsgs--
			s.pBytes -= msgLen
			msgLen = -1
		}

		if s.pHead == nil && !s.closed {
			s.pCond.Wait()
		}
		// Pop the msg off the list
		m := s.pHead
		if m != nil {
			s.pHead = m.next
			if s.pHead == nil {
				s.pTail = nil
			}
			if m.barrier != nil {
				s.mu.Unlock()
				if atomic.AddInt64(&m.barrier.refs, -1) == 0 {
					m.barrier.f()
				}
				continue
			}
			msgLen = len(m.Data)
		}
		mcb := s.mcb
		max = s.max
		closed = s.closed
		var fcReply string
		if !s.closed {
			s.delivered++
			delivered = s.delivered
			if s.jsi != nil {
				fcReply = s.checkForFlowControlResponse()
			}
		}
		s.mu.Unlock()

		// Respond to flow control if applicable
		if fcReply != _EMPTY_ {
			nc.Publish(fcReply, nil)
		}

		if closed {
			break
		}

		// Deliver the message.
		if m != nil && (max == 0 || delivered <= max) {
			mcb(m)
		}
		// If we have hit the max for delivered msgs, remove sub.
		if max > 0 && delivered >= max {
			nc.mu.Lock()
			nc.removeSub(s)
			nc.mu.Unlock()
			break
		}
	}
	// Check for barrier messages
	s.mu.Lock()
	for m := s.pHead; m != nil; m = s.pHead {
		if m.barrier != nil {
			s.mu.Unlock()
			if atomic.AddInt64(&m.barrier.refs, -1) == 0 {
				m.barrier.f()
			}
			s.mu.Lock()
		}
		s.pHead = m.next
	}
	// Now check for pDone
	done := s.pDone
	s.mu.Unlock()

	if done != nil {
		done(s.Subject)
	}
}

// Used for debugging and simulating loss for certain tests.
// Return what is to be used. If we return nil the message will be dropped.
type msgFilter func(m *Msg) *Msg

// processMsg is called by parse and will place the msg on the
// appropriate channel/pending queue for processing. If the channel is full,
// or the pending queue is over the pending limits, the connection is
// considered a slow consumer.
func (nc *Conn) processMsg(data []byte) {
	// Stats
	atomic.AddUint64(&nc.InMsgs, 1)
	atomic.AddUint64(&nc.InBytes, uint64(len(data)))

	// Don't lock the connection to avoid server cutting us off if the
	// flusher is holding the connection lock, trying to send to the server
	// that is itself trying to send data to us.
	nc.subsMu.RLock()
	sub := nc.subs[nc.ps.ma.sid]
	var mf msgFilter
	if nc.filters != nil {
		mf = nc.filters[string(nc.ps.ma.subject)]
	}
	nc.subsMu.RUnlock()

	if sub == nil {
		return
	}

	// Copy them into string
	subj := string(nc.ps.ma.subject)
	reply := string(nc.ps.ma.reply)

	// Doing message create outside of the sub's lock to reduce contention.
	// It's possible that we end-up not using the message, but that's ok.

	// FIXME(dlc): Need to copy, should/can do COW?
	msgPayload := data
	if !nc.ps.msgCopied {
		msgPayload = make([]byte, len(data))
		copy(msgPayload, data)
	}

	// Check if we have headers encoded here.
	var h Header
	var err error
	var ctrlMsg bool
	var ctrlType int
	var fcReply string

	if nc.ps.ma.hdr > 0 {
		hbuf := msgPayload[:nc.ps.ma.hdr]
		msgPayload = msgPayload[nc.ps.ma.hdr:]
		h, err = DecodeHeadersMsg(hbuf)
		if err != nil {
			// We will pass the message through but send async error.
			nc.mu.Lock()
			nc.err = ErrBadHeaderMsg
			if asyncErrorCB := nc.Opts.AsyncErrorCB; asyncErrorCB != nil {
				nc.ach.push(func() { asyncErrorCB(nc, sub, ErrBadHeaderMsg) })
			}
			nc.mu.Unlock()
		}
	}

	// FIXME(dlc): Should we recycle these containers?
	m := &Msg{
		Subject: subj,
		Reply:   reply,
		Header:  h,
		Data:    msgPayload,
		Sub:     sub,
		wsz:     len(data) + len(subj) + len(reply),
	}

	// Check for message filters.
	if mf != nil {
		if m = mf(m); m == nil {
			// Drop message.
			return
		}
	}

	sub.mu.Lock()

	// Check if closed.
	if sub.closed {
		sub.mu.Unlock()
		return
	}

	// Skip flow control messages in case of using a JetStream context.
	jsi := sub.jsi
	if jsi != nil {
		// There has to be a header for it to be a control message.
		if h != nil {
			ctrlMsg, ctrlType = isJSControlMessage(m)
			if ctrlMsg && ctrlType == jsCtrlHB {
				// Check if the heartbeat has a "Consumer Stalled" header, if
				// so, the value is the FC reply to send a nil message to.
				// We will send it at the end of this function.
				fcReply = m.Header.Get(consumerStalledHdr)
			}
		}
		// Check for ordered consumer here. If checkOrderedMsgs returns true that means it detected a gap.
		if !ctrlMsg && jsi.ordered && sub.checkOrderedMsgs(m) {
			sub.mu.Unlock()
			return
		}
	}

	// Skip processing if this is a control message and
	// if not a pull consumer heartbeat. For pull consumers,
	// heartbeats have to be handled on per request basis.
	if !ctrlMsg || (jsi != nil && jsi.pull) {
		var chanSubCheckFC bool
		// Subscription internal stats (applicable only for non ChanSubscription's)
		if sub.typ != ChanSubscription {
			sub.pMsgs++
			if sub.pMsgs > sub.pMsgsMax {
				sub.pMsgsMax = sub.pMsgs
			}
			sub.pBytes += len(m.Data)
			if sub.pBytes > sub.pBytesMax {
				sub.pBytesMax = sub.pBytes
			}

			// Check for a Slow Consumer
			if (sub.pMsgsLimit > 0 && sub.pMsgs > sub.pMsgsLimit) ||
				(sub.pBytesLimit > 0 && sub.pBytes > sub.pBytesLimit) {
				goto slowConsumer
			}
		} else if jsi != nil {
			chanSubCheckFC = true
		}

		// We have two modes of delivery. One is the channel, used by channel
		// subscribers and syncSubscribers, the other is a linked list for async.
		if sub.mch != nil {
			select {
			case sub.mch <- m:
				// For ChanSubscribe, track delivered count here
				if sub.typ == ChanSubscription {
					sub.delivered++
				}
			default:
				goto slowConsumer
			}
		} else {
			// Push onto the async pList
			if sub.pHead == nil {
				sub.pHead = m
				sub.pTail = m
				if sub.pCond != nil {
					sub.pCond.Signal()
				}
			} else {
				sub.pTail.next = m
				sub.pTail = m
			}
		}
		if jsi != nil {
			// Store the ACK metadata from the message to
			// compare later on with the received heartbeat.
			sub.trackSequences(m.Reply)
			if chanSubCheckFC {
				// For ChanSubscription, since we can't call this when a message
				// is "delivered" (since user is pull from their own channel),
				// we have a go routine that does this check, however, we do it
				// also here to make it much more responsive. The go routine is
				// really to avoid stalling when there is no new messages coming.
				fcReply = sub.checkForFlowControlResponse()
			}
		}
	} else if ctrlType == jsCtrlFC && m.Reply != _EMPTY_ {
		// This is a flow control message.
		// We will schedule the send of the FC reply once we have delivered the
		// DATA message that was received before this flow control message, which
		// has sequence `jsi.fciseq`. However, it is possible that this message
		// has already been delivered, in that case, we need to send the FC reply now.
		if sub.getJSDelivered() >= jsi.fciseq {
			fcReply = m.Reply
		} else {
			// Schedule a reply after the previous message is delivered.
			sub.scheduleFlowControlResponse(m.Reply)
		}
	}

	// Clear any SlowConsumer status.
	if sub.sc {
		sub.changeSubStatus(SubscriptionActive)
	}
	sub.sc = false
	sub.mu.Unlock()

	if fcReply != _EMPTY_ {
		nc.Publish(fcReply, nil)
	}

	// Handle control heartbeat messages.
	if ctrlMsg && ctrlType == jsCtrlHB && m.Reply == _EMPTY_ {
		nc.checkForSequenceMismatch(m, sub, jsi)
	}

	// Check if we need to auto-unsubscribe for chan subscriptions
	if sub.typ == ChanSubscription && sub.max > 0 && !ctrlMsg {
		sub.mu.Lock()
		if sub.delivered >= sub.max {
			sub.mu.Unlock()
			nc.mu.Lock()
			nc.removeSub(sub)
			nc.mu.Unlock()
		} else {
			sub.mu.Unlock()
		}
	}

	return

slowConsumer:
	sub.dropped++
	sc := !sub.sc
	sub.sc = true
	// Undo stats from above
	if sub.typ != ChanSubscription {
		sub.pMsgs--
		sub.pBytes -= len(m.Data)
	}
	if sc {
		sub.changeSubStatus(SubscriptionSlowConsumer)
		sub.mu.Unlock()
		// Now we need connection's lock and we may end-up in the situation
		// that we were trying to avoid, except that in this case, the client
		// is already experiencing client-side slow consumer situation.
		nc.mu.Lock()
		nc.err = ErrSlowConsumer
		if asyncErrorCB := nc.Opts.AsyncErrorCB; asyncErrorCB != nil {
			nc.ach.push(func() { asyncErrorCB(nc, sub, ErrSlowConsumer) })
		}
		nc.mu.Unlock()
	} else {
		sub.mu.Unlock()
	}
}

var (
	permissionsRe      = regexp.MustCompile(`Subscription to "(\S+)"`)
	permissionsQueueRe = regexp.MustCompile(`using queue "(\S+)"`)
)

// processTransientError is called when the server signals a non terminal error
// which does not close the connection or trigger a reconnect.
// This will trigger the async error callback if set.
// These errors include the following:
// - permissions violation on publish or subscribe
// - maximum subscriptions exceeded
func (nc *Conn) processTransientError(err error) {
	nc.mu.Lock()
	nc.err = err
	if errors.Is(err, ErrPermissionViolation) {
		matches := permissionsRe.FindStringSubmatch(err.Error())
		if len(matches) >= 2 {
			queueMatches := permissionsQueueRe.FindStringSubmatch(err.Error())
			var q string
			if len(queueMatches) >= 2 {
				q = queueMatches[1]
			}
			subject := matches[1]
			for _, sub := range nc.subs {
				if sub.Subject == subject && sub.Queue == q && sub.permissionsErr == nil {
					sub.mu.Lock()
					if sub.errCh != nil {
						sub.errCh <- err
					}
					sub.permissionsErr = err
					sub.mu.Unlock()
				}
			}
		}
	}
	if asyncErrorCB := nc.Opts.AsyncErrorCB; asyncErrorCB != nil {
		nc.ach.push(func() { asyncErrorCB(nc, nil, err) })
	}
	nc.mu.Unlock()
}

// processAuthError generally processing for auth errors. We want to do retries
// unless we get the same error again. This allows us for instance to swap credentials
// and have the app reconnect, but if nothing is changing we should bail.
// This function will return true if the connection should be closed, false otherwise.
// Connection lock is held on entry
func (nc *Conn) processAuthError(err error) bool {
	nc.err = err
	if !nc.initc {
		if asyncErrorCB := nc.Opts.AsyncErrorCB; asyncErrorCB != nil {
			nc.ach.push(func() { asyncErrorCB(nc, nil, err) })
		}
	}
	// We should give up if we tried twice on this server and got the
	// same error. This behavior can be modified using IgnoreAuthErrorAbort.
	if nc.current.lastErr == err && !nc.Opts.IgnoreAuthErrorAbort {
		nc.ar = true
	} else {
		nc.current.lastErr = err
	}
	return nc.ar
}

// flusher is a separate Go routine that will process flush requests for the write
// bufio. This allows coalescing of writes to the underlying socket.
func (nc *Conn) flusher() {
	// Release the wait group
	defer nc.wg.Done()

	// snapshot the bw and conn since they can change from underneath of us.
	nc.mu.Lock()
	bw := nc.bw
	conn := nc.conn
	fch := nc.fch
	nc.mu.Unlock()

	if conn == nil || bw == nil {
		return
	}

	for {
		if _, ok := <-fch; !ok {
			return
		}
		nc.mu.Lock()

		// Check to see if we should bail out.
		if !nc.isConnected() || nc.isConnecting() || conn != nc.conn {
			nc.mu.Unlock()
			return
		}
		if bw.buffered() > 0 {
			if err := bw.flush(); err != nil {
				if nc.err == nil {
					nc.err = err
				}
				if asyncErrorCB := nc.Opts.AsyncErrorCB; asyncErrorCB != nil {
					nc.ach.push(func() { asyncErrorCB(nc, nil, err) })
				}
				if nc.Opts.ReconnectOnFlusherError {
					nc.mu.Unlock()
					if shouldClose := nc.processOpErr(err, true); shouldClose {
						nc.close(CLOSED, true, nil)
					}
					return
				}
			}
		}
		nc.mu.Unlock()
	}
}

// processPing will send an immediate pong protocol response to the
// server. The server uses this mechanism to detect dead clients.
func (nc *Conn) processPing() {
	nc.sendProto(pongProto)
}

// processPong is used to process responses to the client's ping
// messages. We use pings for the flush mechanism as well.
func (nc *Conn) processPong() {
	var ch chan struct{}

	nc.mu.Lock()
	if len(nc.pongs) > 0 {
		ch = nc.pongs[0]
		nc.pongs = append(nc.pongs[:0], nc.pongs[1:]...)
	}
	nc.pout = 0
	nc.mu.Unlock()
	if ch != nil {
		ch <- struct{}{}
	}
}

// processOK is a placeholder for processing OK messages.
func (nc *Conn) processOK() {
	// do nothing
}

// processInfo is used to parse the info messages sent
// from the server.
// This function may update the server pool.
func (nc *Conn) processInfo(info string) error {
	if info == _EMPTY_ {
		return nil
	}
	var ncInfo ServerInfo
	if err := json.Unmarshal([]byte(info), &ncInfo); err != nil {
		return err
	}

	// Copy content into connection's info structure.
	nc.info = ncInfo
	// The array could be empty/not present on initial connect,
	// if advertise is disabled on that server, or servers that
	// did not include themselves in the async INFO protocol.
	// If empty, do not remove the implicit servers from the pool.
	if len(nc.info.ConnectURLs) == 0 || nc.Opts.IgnoreDiscoveredServers {
		if !nc.initc && ncInfo.LameDuckMode && nc.Opts.LameDuckModeHandler != nil {
			nc.ach.push(func() { nc.Opts.LameDuckModeHandler(nc) })
		}
		return nil
	}
	// Note about pool randomization: when the pool was first created,
	// it was randomized (if allowed). We keep the order the same (removing
	// implicit servers that are no longer sent to us). New URLs are sent
	// to us in no specific order so don't need extra randomization.
	hasNew := false
	// This is what we got from the server we are connected to.
	urls := nc.info.ConnectURLs
	// Transform that to a map for easy lookups
	tmp := make(map[string]struct{}, len(urls))
	for _, curl := range urls {
		tmp[curl] = struct{}{}
	}
	// Walk the pool and removed the implicit servers that are no longer in the
	// given array/map
	sp := nc.srvPool
	for i := 0; i < len(sp); i++ {
		srv := sp[i]
		curl := srv.URL.Host
		// Check if this URL is in the INFO protocol
		_, inInfo := tmp[curl]
		// Remove from the temp map so that at the end we are left with only
		// new (or restarted) servers that need to be added to the pool.
		delete(tmp, curl)
		// Keep servers that were set through Options, but also the one that
		// we are currently connected to (even if it is a discovered server).
		if !srv.isImplicit || srv.URL == nc.current.URL {
			continue
		}
		if !inInfo {
			// Remove from server pool. Keep current order.
			copy(sp[i:], sp[i+1:])
			nc.srvPool = sp[:len(sp)-1]
			sp = nc.srvPool
			i--
		}
	}
	// Figure out if we should save off the current non-IP hostname if we encounter a bare IP.
	saveTLS := nc.current != nil && !hostIsIP(nc.current.URL)

	// If there are any left in the tmp map, these are new (or restarted) servers
	// and need to be added to the pool.
	for curl := range tmp {
		// Before adding, check if this is a new (as in never seen) URL.
		// This is used to figure out if we invoke the DiscoveredServersCB
		if _, present := nc.urls[curl]; !present {
			hasNew = true
		}
		nc.addURLToPool(fmt.Sprintf("%s://%s", nc.connScheme(), curl), true, saveTLS)
	}
	if hasNew {
		// Randomize the pool if allowed but leave the first URL in place.
		if !nc.Opts.NoRandomize {
			nc.shufflePool(1)
		}
		if !nc.initc {
			if discoveredServersCB := nc.Opts.DiscoveredServersCB; discoveredServersCB != nil {
				nc.ach.push(func() { discoveredServersCB(nc) })
			}
		}
	}
	if !nc.initc && ncInfo.LameDuckMode {
		if lameDuckModeHandler := nc.Opts.LameDuckModeHandler; lameDuckModeHandler != nil {
			nc.ach.push(func() { lameDuckModeHandler(nc) })
		}
	}
	return nil
}

// processAsyncInfo does the same than processInfo, but is called
// from the parser. Calls processInfo under connection's lock
// protection.
func (nc *Conn) processAsyncInfo(info []byte) {
	nc.mu.Lock()
	// Ignore errors, we will simply not update the server pool...
	nc.processInfo(string(info))
	nc.mu.Unlock()
}

// LastError reports the last error encountered via the connection.
// It can be used reliably within ClosedCB in order to find out reason
// why connection was closed for example.
func (nc *Conn) LastError() error {
	if nc == nil {
		return ErrInvalidConnection
	}
	nc.mu.RLock()
	err := nc.err
	nc.mu.RUnlock()
	return err
}

// Check if the given error string is an auth error, and if so returns
// the corresponding ErrXXX error, nil otherwise
func checkAuthError(e string) error {
	if strings.HasPrefix(e, AUTHORIZATION_ERR) {
		return ErrAuthorization
	}
	if strings.HasPrefix(e, AUTHENTICATION_EXPIRED_ERR) {
		return ErrAuthExpired
	}
	if strings.HasPrefix(e, AUTHENTICATION_REVOKED_ERR) {
		return ErrAuthRevoked
	}
	if strings.HasPrefix(e, ACCOUNT_AUTHENTICATION_EXPIRED_ERR) {
		return ErrAccountAuthExpired
	}
	return nil
}

// processErr processes any error messages from the server and
// sets the connection's LastError.
func (nc *Conn) processErr(ie string) {
	// Trim, remove quotes
	ne := normalizeErr(ie)
	// convert to lower case.
	e := strings.ToLower(ne)

	var close bool

	// FIXME(dlc) - process Slow Consumer signals special.
	if e == STALE_CONNECTION {
		close = nc.processOpErr(ErrStaleConnection, false)
	} else if e == MAX_CONNECTIONS_ERR {
		close = nc.processOpErr(ErrMaxConnectionsExceeded, false)
	} else if e == MAX_ACCOUNT_CONNECTIONS_ERR {
		close = nc.processOpErr(ErrMaxAccountConnectionsExceeded, false)
	} else if strings.HasPrefix(e, PERMISSIONS_ERR) {
		nc.processTransientError(fmt.Errorf("%w: %s", ErrPermissionViolation, ne))
	} else if strings.HasPrefix(e, MAX_SUBSCRIPTIONS_ERR) {
		nc.processTransientError(ErrMaxSubscriptionsExceeded)
	} else if authErr := checkAuthError(e); authErr != nil {
		nc.mu.Lock()
		close = nc.processAuthError(authErr)
		nc.mu.Unlock()
	} else {
		close = true
		nc.mu.Lock()
		nc.err = errors.New("nats: " + ne)
		nc.mu.Unlock()
	}
	if close {
		nc.close(CLOSED, true, nil)
	}
}

// kickFlusher will send a bool on a channel to kick the
// flush Go routine to flush data to the server.
func (nc *Conn) kickFlusher() {
	if nc.bw != nil {
		select {
		case nc.fch <- struct{}{}:
		default:
		}
	}
}

// Publish publishes the data argument to the given subject. The data
// argument is left untouched and needs to be correctly interpreted on
// the receiver.
func (nc *Conn) Publish(subj string, data []byte) error {
	return nc.publish(subj, _EMPTY_, false, nil, data)
}

// Header represents the optional Header for a NATS message,
// based on the implementation of http.Header.
type Header map[string][]string

// Add adds the key, value pair to the header. It is case-sensitive
// and appends to any existing values associated with key.
func (h Header) Add(key, value string) {
	h[key] = append(h[key], value)
}

// Set sets the header entries associated with key to the single
// element value. It is case-sensitive and replaces any existing
// values associated with key.
func (h Header) Set(key, value string) {
	h[key] = []string{value}
}

// Get gets the first value associated with the given key.
// It is case-sensitive.
func (h Header) Get(key string) string {
	if h == nil {
		return _EMPTY_
	}
	if v := h[key]; v != nil {
		return v[0]
	}
	return _EMPTY_
}

// Values returns all values associated with the given key.
// It is case-sensitive.
func (h Header) Values(key string) []string {
	return h[key]
}

// Del deletes the values associated with a key.
// It is case-sensitive.
func (h Header) Del(key string) {
	delete(h, key)
}

// NewMsg creates a message for publishing that will use headers.
func NewMsg(subject string) *Msg {
	return &Msg{
		Subject: subject,
		Header:  make(Header),
	}
}

const (
	hdrLine            = "NATS/1.0\r\n"
	crlf               = "\r\n"
	hdrPreEnd          = len(hdrLine) - len(crlf)
	statusHdr          = "Status"
	descrHdr           = "Description"
	lastConsumerSeqHdr = "Nats-Last-Consumer"
	lastStreamSeqHdr   = "Nats-Last-Stream"
	consumerStalledHdr = "Nats-Consumer-Stalled"
	noResponders       = "503"
	noMessagesSts      = "404"
	reqTimeoutSts      = "408"
	jetStream409Sts    = "409"
	controlMsg         = "100"
	statusLen          = 3 // e.g. 20x, 40x, 50x
)

// DecodeHeadersMsg will decode and headers.
func DecodeHeadersMsg(data []byte) (Header, error) {
	br := bufio.NewReaderSize(bytes.NewReader(data), 128)
	tp := textproto.NewReader(br)
	l, err := tp.ReadLine()
	if err != nil || len(l) < hdrPreEnd || l[:hdrPreEnd] != hdrLine[:hdrPreEnd] {
		return nil, ErrBadHeaderMsg
	}

	mh, err := readMIMEHeader(tp)
	if err != nil {
		return nil, err
	}

	// Check if we have an inlined status.
	if len(l) > hdrPreEnd {
		var description string
		status := strings.TrimSpace(l[hdrPreEnd:])
		if len(status) != statusLen {
			description = strings.TrimSpace(status[statusLen:])
			status = status[:statusLen]
		}
		mh.Add(statusHdr, status)
		if len(description) > 0 {
			mh.Add(descrHdr, description)
		}
	}
	return Header(mh), nil
}

// readMIMEHeader returns a MIMEHeader that preserves the
// original case of the MIME header, based on the implementation
// of textproto.ReadMIMEHeader.
//
// https://golang.org/pkg/net/textproto/#Reader.ReadMIMEHeader
func readMIMEHeader(tp *textproto.Reader) (textproto.MIMEHeader, error) {
	m := make(textproto.MIMEHeader)
	for {
		kv, err := tp.ReadLine()
		if len(kv) == 0 {
			return m, err
		}

		// Process key fetching original case.
		i := strings.IndexByte(kv, ':')
		if i < 0 {
			return nil, ErrBadHeaderMsg
		}
		key := kv[:i]
		if key == "" {
			// Skip empty keys.
			continue
		}
		i++
		for i < len(kv) && (kv[i] == ' ' || kv[i] == '\t') {
			i++
		}
		m[key] = append(m[key], kv[i:])
		if err != nil {
			return m, err
		}
	}
}

// PublishMsg publishes the Msg structure, which includes the
// Subject, an optional Reply and an optional Data field.
func (nc *Conn) PublishMsg(m *Msg) error {
	if m == nil {
		return ErrInvalidMsg
	}
	hdr, err := m.headerBytes()
	if err != nil {
		return err
	}
	validateReply := m.Reply != _EMPTY_
	return nc.publish(m.Subject, m.Reply, validateReply, hdr, m.Data)
}

// PublishRequest will perform a Publish() expecting a response on the
// reply subject. Use Request() for automatically waiting for a response
// inline.
func (nc *Conn) PublishRequest(subj, reply string, data []byte) error {
	return nc.publish(subj, reply, true, nil, data)
}

// Used for handrolled Itoa
const digits = "0123456789"

// validateSubject checks if the subject contains characters that break the NATS protocol.
// Uses an adaptive algorithm: manual loop for short subjects (< 16 chars) and
// SIMD-optimized strings.IndexByte for longer subjects.
func validateSubject(subj string) error {
	if subj == "" {
		return ErrBadSubject
	}

	// Adaptive threshold based on benchmark data showing crossover at ~15-20 characters.
	const lengthThreshold = 16

	if len(subj) < lengthThreshold {
		// Fast path for short subjects (< 16 chars)
		// Short-circuit on non-control characters.
		for i := range len(subj) {
			c := subj[i]
			if c <= ' ' && (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
				return ErrBadSubject
			}
		}
		return nil
	}

	// Optimized path for long subjects (>= 16 chars)
	// Uses SIMD-optimized strings.IndexByte (processes 16+ bytes per instruction)
	if strings.IndexByte(subj, ' ') >= 0 ||
		strings.IndexByte(subj, '\t') >= 0 ||
		strings.IndexByte(subj, '\r') >= 0 ||
		strings.IndexByte(subj, '\n') >= 0 {
		return ErrBadSubject
	}
	return nil
}

// publish is the internal function to publish messages to a nats-server.
// Sends a protocol data message by queuing into the bufio writer
// and kicking the flush go routine. These writes should be protected.
func (nc *Conn) publish(subj, reply string, validateReply bool, hdr, data []byte) error {
	if nc == nil {
		return ErrInvalidConnection
	}
	if !nc.Opts.SkipSubjectValidation {
		if err := validateSubject(subj); err != nil {
			return err
		}
		if validateReply {
			if err := validateSubject(reply); err != nil {
				return ErrBadSubject
			}
		}
	} else if subj == _EMPTY_ {
		return ErrBadSubject
	}
	nc.mu.Lock()

	// Check if headers attempted to be sent to server that does not support them.
	if len(hdr) > 0 && !nc.info.Headers {
		nc.mu.Unlock()
		return ErrHeadersNotSupported
	}

	if nc.isClosed() {
		nc.mu.Unlock()
		return ErrConnectionClosed
	}

	if nc.isDrainingPubs() {
		nc.mu.Unlock()
		return ErrConnectionDraining
	}

	// Proactively reject payloads over the threshold set by server.
	msgSize := int64(len(data) + len(hdr))
	// Skip this check if we are not yet connected (RetryOnFailedConnect)
	if !nc.initc && msgSize > nc.info.MaxPayload {
		nc.mu.Unlock()
		return ErrMaxPayload
	}

	// Check if we are reconnecting, and if so check if
	// we have exceeded our reconnect outbound buffer limits.
	if nc.bw.atLimitIfUsingPending() {
		nc.mu.Unlock()
		return ErrReconnectBufExceeded
	}

	var mh []byte
	if hdr != nil {
		mh = nc.scratch[:len(_HPUB_P_)]
	} else {
		mh = nc.scratch[1:len(_HPUB_P_)]
	}
	mh = append(mh, subj...)
	mh = append(mh, ' ')
	if reply != "" {
		mh = append(mh, reply...)
		mh = append(mh, ' ')
	}

	// We could be smarter here, but simple loop is ok,
	// just avoid strconv in fast path.
	// FIXME(dlc) - Find a better way here.
	// msgh = strconv.AppendInt(msgh, int64(len(data)), 10)
	// go 1.14 some values strconv faster, may be able to switch over.

	var b [12]byte
	i := len(b)

	if hdr != nil {
		if len(hdr) > 0 {
			for l := len(hdr); l > 0; l /= 10 {
				i--
				b[i] = digits[l%10]
			}
		} else {
			i--
			b[i] = digits[0]
		}
		mh = append(mh, b[i:]...)
		mh = append(mh, ' ')
		// reset for below.
		i = len(b)
	}

	if msgSize > 0 {
		for l := msgSize; l > 0; l /= 10 {
			i--
			b[i] = digits[l%10]
		}
	} else {
		i--
		b[i] = digits[0]
	}

	mh = append(mh, b[i:]...)
	mh = append(mh, _CRLF_...)

	if err := nc.bw.appendBufs(mh, hdr, data, _CRLF_BYTES_); err != nil {
		nc.mu.Unlock()
		return err
	}

	nc.OutMsgs++
	nc.OutBytes += uint64(len(data) + len(hdr))

	if len(nc.fch) == 0 {
		nc.kickFlusher()
	}
	nc.mu.Unlock()
	return nil
}

// respHandler is the global response handler. It will look up
// the appropriate channel based on the last token and place
// the message on the channel if possible.
func (nc *Conn) respHandler(m *Msg) {
	nc.mu.Lock()

	// Just return if closed.
	if nc.isClosed() {
		nc.mu.Unlock()
		return
	}

	var mch chan *Msg

	// Grab mch
	rt := nc.respToken(m.Subject)
	if rt != _EMPTY_ {
		mch = nc.respMap[rt]
		// Delete the key regardless, one response only.
		delete(nc.respMap, rt)
	} else if len(nc.respMap) == 1 {
		// If the server has rewritten the subject, the response token (rt)
		// will not match (could be the case with JetStream). If that is the
		// case and there is a single entry, use that.
		for k, v := range nc.respMap {
			mch = v
			delete(nc.respMap, k)
			break
		}
	}
	nc.mu.Unlock()

	// Don't block, let Request timeout instead, mch is
	// buffered and we should delete the key before a
	// second response is processed.
	select {
	case mch <- m:
	default:
		return
	}
}

// Helper to setup and send new request style requests. Return the chan to receive the response.
func (nc *Conn) createNewRequestAndSend(subj string, hdr, data []byte) (chan *Msg, string, error) {
	nc.mu.Lock()
	// Create new literal Inbox and map to a chan msg.
	mch := make(chan *Msg, RequestChanLen)
	respInbox := nc.newRespInbox()
	token := respInbox[nc.respSubLen:]

	nc.respMap[token] = mch
	if nc.respMux == nil {
		// Create the response subscription we will use for all new style responses.
		// This will be on an _INBOX with an additional terminal token. The subscription
		// will be on a wildcard.
		s, err := nc.subscribeLocked(nc.respSub, _EMPTY_, nc.respHandler, nil, nil, false, nil)
		if err != nil {
			nc.mu.Unlock()
			return nil, token, err
		}
		nc.respMux = s
	}
	nc.mu.Unlock()

	if err := nc.publish(subj, respInbox, false, hdr, data); err != nil {
		return nil, token, err
	}

	return mch, token, nil
}

// RequestMsg will send a request payload including optional headers and deliver
// the response message, or an error, including a timeout if no message was received properly.
func (nc *Conn) RequestMsg(msg *Msg, timeout time.Duration) (*Msg, error) {
	if msg == nil {
		return nil, ErrInvalidMsg
	}
	hdr, err := msg.headerBytes()
	if err != nil {
		return nil, err
	}

	return nc.request(msg.Subject, hdr, msg.Data, timeout)
}

// Request will send a request payload and deliver the response message,
// or an error, including a timeout if no message was received properly.
func (nc *Conn) Request(subj string, data []byte, timeout time.Duration) (*Msg, error) {
	return nc.request(subj, nil, data, timeout)
}

func (nc *Conn) useOldRequestStyle() bool {
	nc.mu.RLock()
	r := nc.Opts.UseOldRequestStyle
	nc.mu.RUnlock()
	return r
}

func (nc *Conn) request(subj string, hdr, data []byte, timeout time.Duration) (*Msg, error) {
	if nc == nil {
		return nil, ErrInvalidConnection
	}

	var m *Msg
	var err error

	if nc.useOldRequestStyle() {
		m, err = nc.oldRequest(subj, hdr, data, timeout)
	} else {
		m, err = nc.newRequest(subj, hdr, data, timeout)
	}

	// Check for no responder status.
	if err == nil && len(m.Data) == 0 && m.Header.Get(statusHdr) == noResponders {
		m, err = nil, ErrNoResponders
	}
	return m, err
}

func (nc *Conn) newRequest(subj string, hdr, data []byte, timeout time.Duration) (*Msg, error) {
	mch, token, err := nc.createNewRequestAndSend(subj, hdr, data)
	if err != nil {
		return nil, err
	}

	t := globalTimerPool.Get(timeout)
	defer globalTimerPool.Put(t)

	var ok bool
	var msg *Msg

	select {
	case msg, ok = <-mch:
		if !ok {
			return nil, ErrConnectionClosed
		}
	case <-t.C:
		nc.mu.Lock()
		delete(nc.respMap, token)
		nc.mu.Unlock()
		return nil, ErrTimeout
	}

	return msg, nil
}

// oldRequest will create an Inbox and perform a Request() call
// with the Inbox reply and return the first reply received.
// This is optimized for the case of multiple responses.
func (nc *Conn) oldRequest(subj string, hdr, data []byte, timeout time.Duration) (*Msg, error) {
	inbox := nc.NewInbox()
	ch := make(chan *Msg, RequestChanLen)

	s, err := nc.subscribe(inbox, _EMPTY_, nil, ch, nil, true, nil)
	if err != nil {
		return nil, err
	}
	s.AutoUnsubscribe(1)
	defer s.Unsubscribe()

	err = nc.publish(subj, inbox, false, hdr, data)
	if err != nil {
		return nil, err
	}

	return s.NextMsg(timeout)
}

// InboxPrefix is the prefix for all inbox subjects.
const (
	InboxPrefix    = "_INBOX."
	inboxPrefixLen = len(InboxPrefix)
	replySuffixLen = 8 // Gives us 62^8
	rdigits        = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	base           = 62
)

// NewInbox will return an inbox string which can be used for directed replies from
// subscribers. These are guaranteed to be unique, but can be shared and subscribed
// to by others.
func NewInbox() string {
	var b [inboxPrefixLen + nuidSize]byte
	pres := b[:inboxPrefixLen]
	copy(pres, InboxPrefix)
	ns := b[inboxPrefixLen:]
	copy(ns, nuid.Next())
	return string(b[:])
}

// Create a new inbox that is prefix aware.
func (nc *Conn) NewInbox() string {
	if nc.Opts.InboxPrefix == _EMPTY_ {
		return NewInbox()
	}

	var sb strings.Builder
	sb.WriteString(nc.Opts.InboxPrefix)
	sb.WriteByte('.')
	sb.WriteString(nuid.Next())
	return sb.String()
}

// Function to init new response structures.
func (nc *Conn) initNewResp() {
	nc.respSubPrefix = fmt.Sprintf("%s.", nc.NewInbox())
	nc.respSubLen = len(nc.respSubPrefix)
	nc.respSub = fmt.Sprintf("%s*", nc.respSubPrefix)
	nc.respMap = make(map[string]chan *Msg)
	nc.respRand = rand.New(rand.NewSource(time.Now().UnixNano()))
}

// newRespInbox creates a new literal response subject
// that will trigger the mux subscription handler.
// Lock should be held.
func (nc *Conn) newRespInbox() string {
	if nc.respMap == nil {
		nc.initNewResp()
	}

	var sb strings.Builder
	sb.WriteString(nc.respSubPrefix)

	rn := nc.respRand.Int63()
	for i := 0; i < replySuffixLen; i++ {
		sb.WriteByte(rdigits[rn%base])
		rn /= base
	}

	return sb.String()
}

// NewRespInbox is the new format used for _INBOX.
func (nc *Conn) NewRespInbox() string {
	nc.mu.Lock()
	s := nc.newRespInbox()
	nc.mu.Unlock()
	return s
}

// respToken will return the last token of a literal response inbox
// which we use for the message channel lookup. This needs to verify the subject
// prefix matches to protect itself against the server changing the subject.
// Lock should be held.
func (nc *Conn) respToken(respInbox string) string {
	if token, found := strings.CutPrefix(respInbox, nc.respSubPrefix); found {
		return token
	}
	return ""
}

// Subscribe will express interest in the given subject. The subject
// can have wildcards.
// There are two type of wildcards: * for partial, and > for full.
// A subscription on subject time.*.east would receive messages sent to time.us.east and time.eu.east.
// A subscription on subject time.us.> would receive messages sent to
// time.us.east and time.us.east.atlanta, while time.us.* would only match time.us.east
// since it can't match more than one token.
// Messages will be delivered to the associated MsgHandler.
func (nc *Conn) Subscribe(subj string, cb MsgHandler) (*Subscription, error) {
	return nc.subscribe(subj, _EMPTY_, cb, nil, nil, false, nil)
}

// ChanSubscribe will express interest in the given subject and place
// all messages received on the channel.
// You should not close the channel until sub.Unsubscribe() has been called.
func (nc *Conn) ChanSubscribe(subj string, ch chan *Msg) (*Subscription, error) {
	return nc.subscribe(subj, _EMPTY_, nil, ch, nil, false, nil)
}

// ChanQueueSubscribe will express interest in the given subject.
// All subscribers with the same queue name will form the queue group
// and only one member of the group will be selected to receive any given message,
// which will be placed on the channel.
// You should not close the channel until sub.Unsubscribe() has been called.
// Note: This is the same than QueueSubscribeSyncWithChan.
func (nc *Conn) ChanQueueSubscribe(subj, group string, ch chan *Msg) (*Subscription, error) {
	return nc.subscribe(subj, group, nil, ch, nil, false, nil)
}

// SubscribeSync will express interest on the given subject. Messages will
// be received synchronously using Subscription.NextMsg().
func (nc *Conn) SubscribeSync(subj string) (*Subscription, error) {
	if nc == nil {
		return nil, ErrInvalidConnection
	}
	mch := make(chan *Msg, nc.Opts.SubChanLen)
	var errCh chan error
	if nc.Opts.PermissionErrOnSubscribe {
		errCh = make(chan error, 100)
	}
	return nc.subscribe(subj, _EMPTY_, nil, mch, errCh, true, nil)
}

// QueueSubscribe creates an asynchronous queue subscriber on the given subject.
// All subscribers with the same queue name will form the queue group and
// only one member of the group will be selected to receive any given
// message asynchronously.
func (nc *Conn) QueueSubscribe(subj, queue string, cb MsgHandler) (*Subscription, error) {
	return nc.subscribe(subj, queue, cb, nil, nil, false, nil)
}

// QueueSubscribeSync creates a synchronous queue subscriber on the given
// subject. All subscribers with the same queue name will form the queue
// group and only one member of the group will be selected to receive any
// given message synchronously using Subscription.NextMsg().
func (nc *Conn) QueueSubscribeSync(subj, queue string) (*Subscription, error) {
	mch := make(chan *Msg, nc.Opts.SubChanLen)
	var errCh chan error
	if nc.Opts.PermissionErrOnSubscribe {
		errCh = make(chan error, 100)
	}
	return nc.subscribe(subj, queue, nil, mch, errCh, true, nil)
}

// QueueSubscribeSyncWithChan will express interest in the given subject.
// All subscribers with the same queue name will form the queue group
// and only one member of the group will be selected to receive any given message,
// which will be placed on the channel.
// You should not close the channel until sub.Unsubscribe() has been called.
// Note: This is the same than ChanQueueSubscribe.
func (nc *Conn) QueueSubscribeSyncWithChan(subj, queue string, ch chan *Msg) (*Subscription, error) {
	return nc.subscribe(subj, queue, nil, ch, nil, false, nil)
}

// badSubject will do quick test on whether a subject is acceptable.
// Spaces are not allowed and all tokens should be > 0 in len.
func badSubject(subj string) bool {
	if strings.ContainsAny(subj, " \t\r\n") {
		return true
	}
	tokens := strings.Split(subj, ".")
	for _, t := range tokens {
		if len(t) == 0 {
			return true
		}
	}
	return false
}

// badQueue will check a queue name for whitespace.
func badQueue(qname string) bool {
	return strings.ContainsAny(qname, " \t\r\n")
}

// subscribe is the internal subscribe function that indicates interest in a subject.
func (nc *Conn) subscribe(subj, queue string, cb MsgHandler, ch chan *Msg, errCh chan (error), isSync bool, js *jsSub) (*Subscription, error) {
	if nc == nil {
		return nil, ErrInvalidConnection
	}
	nc.mu.Lock()
	defer nc.mu.Unlock()
	return nc.subscribeLocked(subj, queue, cb, ch, errCh, isSync, js)
}

func (nc *Conn) subscribeLocked(subj, queue string, cb MsgHandler, ch chan *Msg, errCh chan (error), isSync bool, js *jsSub) (*Subscription, error) {
	if nc == nil {
		return nil, ErrInvalidConnection
	}
	if badSubject(subj) {
		return nil, ErrBadSubject
	}
	if queue != _EMPTY_ && badQueue(queue) {
		return nil, ErrBadQueueName
	}

	// Check for some error conditions.
	if nc.isClosed() {
		return nil, ErrConnectionClosed
	}
	if nc.isDraining() {
		return nil, ErrConnectionDraining
	}

	if cb == nil && ch == nil {
		return nil, ErrBadSubscription
	}

	sub := &Subscription{
		Subject: subj,
		Queue:   queue,
		mcb:     cb,
		conn:    nc,
		jsi:     js,
	}
	// Set pending limits.
	if ch != nil {
		sub.pMsgsLimit = cap(ch)
	} else {
		sub.pMsgsLimit = DefaultSubPendingMsgsLimit
	}
	sub.pBytesLimit = DefaultSubPendingBytesLimit

	// If we have an async callback, start up a sub specific
	// Go routine to deliver the messages.
	var sr bool
	if cb != nil {
		sub.typ = AsyncSubscription
		sub.pCond = sync.NewCond(&sub.mu)
		sr = true
	} else if !isSync {
		sub.typ = ChanSubscription
		sub.mch = ch
	} else { // Sync Subscription
		sub.typ = SyncSubscription
		sub.mch = ch
		sub.errCh = errCh
	}

	nc.subsMu.Lock()
	nc.ssid++
	sub.sid = nc.ssid
	nc.subs[sub.sid] = sub
	nc.subsMu.Unlock()

	// Let's start the go routine now that it is fully setup and registered.
	if sr {
		go nc.waitForMsgs(sub)
	}

	// We will send these for all subs when we reconnect
	// so that we can suppress here if reconnecting.
	if !nc.isReconnecting() {
		nc.bw.appendString(fmt.Sprintf(subProto, subj, queue, sub.sid))
		nc.kickFlusher()
	}

	sub.changeSubStatus(SubscriptionActive)
	return sub, nil
}

// NumSubscriptions returns active number of subscriptions.
func (nc *Conn) NumSubscriptions() int {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return len(nc.subs)
}

// Lock for nc should be held here upon entry
func (nc *Conn) removeSub(s *Subscription) {
	nc.subsMu.Lock()
	delete(nc.subs, s.sid)
	nc.subsMu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	// Release callers on NextMsg for SyncSubscription only
	if s.mch != nil && s.typ == SyncSubscription {
		close(s.mch)
	}
	s.mch = nil

	// If JS subscription then stop HB timer.
	if jsi := s.jsi; jsi != nil {
		if jsi.hbc != nil {
			jsi.hbc.Stop()
			jsi.hbc = nil
		}
		if jsi.csfct != nil {
			jsi.csfct.Stop()
			jsi.csfct = nil
		}
	}

	if s.typ != AsyncSubscription {
		done := s.pDone
		if done != nil {
			done(s.Subject)
		}
	}
	// Mark as invalid
	s.closed = true
	s.changeSubStatus(SubscriptionClosed)
	if s.pCond != nil {
		s.pCond.Broadcast()
	}
}

// SubscriptionType is the type of the Subscription.
type SubscriptionType int

// The different types of subscription types.
const (
	AsyncSubscription = SubscriptionType(iota)
	SyncSubscription
	ChanSubscription
	NilSubscription
	PullSubscription
)

// Type returns the type of Subscription.
func (s *Subscription) Type() SubscriptionType {
	if s == nil {
		return NilSubscription
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	// Pull subscriptions are really a SyncSubscription and we want this
	// type to be set internally for all delivered messages management, etc..
	// So check when to return PullSubscription to the user.
	if s.jsi != nil && s.jsi.pull {
		return PullSubscription
	}
	return s.typ
}

// IsValid returns a boolean indicating whether the subscription
// is still active. This will return false if the subscription has
// already been closed.
func (s *Subscription) IsValid() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.closed
}

// Drain will remove interest but continue callbacks until all messages
// have been processed.
//
// For a JetStream subscription, if the library has created the JetStream
// consumer, the library will send a DeleteConsumer request to the server
// when the drain operation completes. If a failure occurs when deleting
// the JetStream consumer, an error will be reported to the asynchronous
// error callback.
// If you do not wish the JetStream consumer to be automatically deleted,
// ensure that the consumer is not created by the library, which means
// create the consumer with AddConsumer and bind to this consumer.
func (s *Subscription) Drain() error {
	if s == nil {
		return ErrBadSubscription
	}
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return ErrBadSubscription
	}
	return conn.unsubscribe(s, 0, true)
}

// IsDraining returns a boolean indicating whether the subscription
// is being drained.
// This will return false if the subscription has already been closed.
func (s *Subscription) IsDraining() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.draining
}

// StatusChanged returns a channel on which given list of subscription status
// changes will be sent. If no status is provided, all status changes will be sent.
// Available statuses are SubscriptionActive, SubscriptionDraining, SubscriptionClosed,
// and SubscriptionSlowConsumer.
// The returned channel will be closed when the subscription is closed.
func (s *Subscription) StatusChanged(statuses ...SubStatus) <-chan SubStatus {
	if len(statuses) == 0 {
		statuses = []SubStatus{SubscriptionActive, SubscriptionDraining, SubscriptionClosed, SubscriptionSlowConsumer}
	}
	ch := make(chan SubStatus, 10)
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.status == SubscriptionClosed {
		if slices.Contains(statuses, SubscriptionClosed) {
			ch <- SubscriptionClosed
		}
		close(ch)
		return ch
	}

	for _, status := range statuses {
		s.registerStatusChangeListener(status, ch)
		// initial status
		if status == s.status {
			ch <- status
		}
	}
	return ch
}

// registerStatusChangeListener registers a channel waiting for a specific status change event.
// Status change events are non-blocking - if no receiver is waiting for the status change,
// it will not be sent on the channel. Closed channels are ignored.
// Lock should be held entering.
func (s *Subscription) registerStatusChangeListener(status SubStatus, ch chan SubStatus) {
	if s.statListeners == nil {
		s.statListeners = make(map[chan SubStatus][]SubStatus)
	}
	if _, ok := s.statListeners[ch]; !ok {
		s.statListeners[ch] = make([]SubStatus, 0)
	}
	s.statListeners[ch] = append(s.statListeners[ch], status)
}

// sendStatusEvent sends subscription status event to all channels.
// If there is no listener, sendStatusEvent
// will not block. Lock should be held entering.
func (s *Subscription) sendStatusEvent(status SubStatus) {
	for ch, statuses := range s.statListeners {
		if !slices.Contains(statuses, status) {
			continue
		}
		// only send event if someone's listening
		select {
		case ch <- status:
		default:
		}
	}
	// After sending SubscriptionClosed status to all listeners,
	// close all channels and clear the map to prevent future
	// sends to closed channels that could cause panics
	if status == SubscriptionClosed {
		for ch := range s.statListeners {
			close(ch)
		}
		s.statListeners = nil
	}
}

// changeSubStatus changes subscription status and sends events
// to all listeners. Lock should be held entering.
func (s *Subscription) changeSubStatus(status SubStatus) {
	if s == nil {
		return
	}
	s.sendStatusEvent(status)
	s.status = status
}

// Unsubscribe will remove interest in the given subject.
//
// For a JetStream subscription, if the library has created the JetStream
// consumer, it will send a DeleteConsumer request to the server (if the
// unsubscribe itself was successful). If the delete operation fails, the
// error will be returned.
// If you do not wish the JetStream consumer to be automatically deleted,
// ensure that the consumer is not created by the library, which means
// create the consumer with AddConsumer and bind to this consumer (using
// the nats.Bind() option).
func (s *Subscription) Unsubscribe() error {
	if s == nil {
		return ErrBadSubscription
	}
	s.mu.Lock()
	conn := s.conn
	closed := s.closed
	dc := s.jsi != nil && s.jsi.dc
	s.mu.Unlock()
	if conn == nil || conn.IsClosed() {
		return ErrConnectionClosed
	}
	if closed {
		return ErrBadSubscription
	}
	if conn.IsDraining() {
		return ErrConnectionDraining
	}
	err := conn.unsubscribe(s, 0, false)
	if err == nil && dc {
		err = s.deleteConsumer()
	}
	return err
}

// checkDrained will watch for a subscription to be fully drained
// and then remove it.
func (nc *Conn) checkDrained(sub *Subscription) {
	if nc == nil || sub == nil {
		return
	}
	defer func() {
		sub.mu.Lock()
		defer sub.mu.Unlock()
		sub.draining = false
	}()

	// This allows us to know that whatever we have in the client pending
	// is correct and the server will not send additional information.
	nc.Flush()

	sub.mu.Lock()
	// For JS subscriptions, check if we are going to delete the
	// JS consumer when drain completes.
	dc := sub.jsi != nil && sub.jsi.dc
	sub.mu.Unlock()

	// Once we are here we just wait for Pending to reach 0 or
	// any other state to exit this go routine.
	for {
		// check connection is still valid.
		if nc.IsClosed() {
			return
		}

		// Check subscription state
		sub.mu.Lock()
		conn := sub.conn
		closed := sub.closed
		pMsgs := sub.pMsgs
		sub.mu.Unlock()

		if conn == nil || closed || pMsgs == 0 {
			nc.mu.Lock()
			nc.removeSub(sub)
			nc.mu.Unlock()
			if dc {
				if err := sub.deleteConsumer(); err != nil {
					nc.mu.Lock()
					if errCB := nc.Opts.AsyncErrorCB; errCB != nil {
						nc.ach.push(func() { errCB(nc, sub, err) })
					}
					nc.mu.Unlock()
				}
			}
			return
		}

		time.Sleep(100 * time.Millisecond)
	}
}

// AutoUnsubscribe will issue an automatic Unsubscribe that is
// processed by the server when max messages have been received.
// This can be useful when sending a request to an unknown number
// of subscribers.
func (s *Subscription) AutoUnsubscribe(max int) error {
	if s == nil {
		return ErrBadSubscription
	}
	s.mu.Lock()
	conn := s.conn
	closed := s.closed
	s.mu.Unlock()
	if conn == nil || closed {
		return ErrBadSubscription
	}
	return conn.unsubscribe(s, max, false)
}

// SetClosedHandler will set the closed handler for when a subscription
// is closed (either unsubscribed or drained).
func (s *Subscription) SetClosedHandler(handler func(subject string)) {
	s.mu.Lock()
	s.pDone = handler
	s.mu.Unlock()
}

// unsubscribe performs the low level unsubscribe to the server.
// Use Subscription.Unsubscribe()
func (nc *Conn) unsubscribe(sub *Subscription, max int, drainMode bool) error {
	var maxStr string
	if max > 0 {
		sub.mu.Lock()
		sub.max = uint64(max)
		if sub.delivered < sub.max {
			maxStr = strconv.Itoa(max)
		}
		sub.mu.Unlock()
	}

	nc.mu.Lock()
	// ok here, but defer is expensive
	defer nc.mu.Unlock()

	if nc.isClosed() {
		return ErrConnectionClosed
	}

	nc.subsMu.RLock()
	s := nc.subs[sub.sid]
	nc.subsMu.RUnlock()
	// Already unsubscribed
	if s == nil {
		return nil
	}

	if maxStr == _EMPTY_ && !drainMode {
		nc.removeSub(s)
	}

	if drainMode {
		s.mu.Lock()
		s.draining = true
		sub.changeSubStatus(SubscriptionDraining)
		s.mu.Unlock()
		go nc.checkDrained(sub)
	}

	// We will send these for all subs when we reconnect
	// so that we can suppress here.
	if !nc.isReconnecting() {
		nc.bw.appendString(fmt.Sprintf(unsubProto, s.sid, maxStr))
		nc.kickFlusher()
	}

	// For JetStream subscriptions cancel the attached context if there is any.
	var cancel func()
	sub.mu.Lock()
	jsi := sub.jsi
	if jsi != nil {
		cancel = jsi.cancel
		jsi.cancel = nil
	}
	sub.mu.Unlock()
	if cancel != nil {
		cancel()
	}

	return nil
}

// NextMsg will return the next message available to a synchronous subscriber
// or block until one is available. An error is returned if the subscription is invalid (ErrBadSubscription),
// the connection is closed (ErrConnectionClosed), the timeout is reached (ErrTimeout),
// or if there were no responders (ErrNoResponders) when used in the context of a request/reply.
func (s *Subscription) NextMsg(timeout time.Duration) (*Msg, error) {
	if s == nil {
		return nil, ErrBadSubscription
	}

	s.mu.Lock()
	err := s.validateNextMsgState(false)
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}

	// snapshot
	mch := s.mch
	s.mu.Unlock()

	var ok bool
	var msg *Msg

	// If something is available right away, let's optimize that case.
	select {
	case msg, ok = <-mch:
		if !ok {
			return nil, s.getNextMsgErr()
		}
		if err := s.processNextMsgDelivered(msg); err != nil {
			return nil, err
		} else {
			return msg, nil
		}
	default:
	}

	// If we are here a message was not immediately available, so lets loop
	// with a timeout.

	t := globalTimerPool.Get(timeout)
	defer globalTimerPool.Put(t)

	if s.errCh != nil {
		select {
		case msg, ok = <-mch:
			if !ok {
				return nil, s.getNextMsgErr()
			}
			if err := s.processNextMsgDelivered(msg); err != nil {
				return nil, err
			}
		case err := <-s.errCh:
			return nil, err
		case <-t.C:
			return nil, ErrTimeout
		}
	} else {
		select {
		case msg, ok = <-mch:
			if !ok {
				return nil, s.getNextMsgErr()
			}
			if err := s.processNextMsgDelivered(msg); err != nil {
				return nil, err
			}
		case <-t.C:
			return nil, ErrTimeout
		}
	}

	return msg, nil
}

// nextMsgNoTimeout works similarly to Subscription.NextMsg() but will not
// time out. It is only used internally for non-timeout subscription iterator.
func (s *Subscription) nextMsgNoTimeout() (*Msg, error) {
	if s == nil {
		return nil, ErrBadSubscription
	}

	s.mu.Lock()
	err := s.validateNextMsgState(false)
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}

	// snapshot
	mch := s.mch
	s.mu.Unlock()

	var ok bool
	var msg *Msg

	// If something is available right away, let's optimize that case.
	select {
	case msg, ok = <-mch:
		if !ok {
			return nil, s.getNextMsgErr()
		}
		if err := s.processNextMsgDelivered(msg); err != nil {
			return nil, err
		} else {
			return msg, nil
		}
	default:
	}

	if s.errCh != nil {
		select {
		case msg, ok = <-mch:
			if !ok {
				return nil, s.getNextMsgErr()
			}
			if err := s.processNextMsgDelivered(msg); err != nil {
				return nil, err
			}
		case err := <-s.errCh:
			return nil, err
		}
	} else {
		msg, ok = <-mch
		if !ok {
			return nil, s.getNextMsgErr()
		}
		if err := s.processNextMsgDelivered(msg); err != nil {
			return nil, err
		}
	}

	return msg, nil
}

// validateNextMsgState checks whether the subscription is in a valid
// state to call NextMsg and be delivered another message synchronously.
// This should be called while holding the lock.
func (s *Subscription) validateNextMsgState(pullSubInternal bool) error {
	if s.connClosed {
		return ErrConnectionClosed
	}
	if s.mch == nil {
		if s.max > 0 && s.delivered >= s.max {
			return ErrMaxMessages
		} else if s.closed {
			return ErrBadSubscription
		}
	}
	if s.mcb != nil {
		return ErrSyncSubRequired
	}
	// if this subscription previously had a permissions error
	// and no reconnect has been attempted, return the permissions error
	// since the subscription does not exist on the server
	if s.conn.Opts.PermissionErrOnSubscribe && s.permissionsErr != nil {
		return s.permissionsErr
	}
	if s.sc {
		s.changeSubStatus(SubscriptionActive)
		s.sc = false
		return ErrSlowConsumer
	}
	// Unless this is from an internal call, reject use of this API.
	// Users should use Fetch() instead.
	if !pullSubInternal && s.jsi != nil && s.jsi.pull {
		return ErrTypeSubscription
	}
	return nil
}

// This is called when the sync channel has been closed.
// The error returned will be either connection or subscription
// closed depending on what caused NextMsg() to fail.
func (s *Subscription) getNextMsgErr() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.connClosed {
		return ErrConnectionClosed
	}
	return ErrBadSubscription
}

// processNextMsgDelivered takes a message and applies the needed
// accounting to the stats from the subscription, returning an
// error in case we have the maximum number of messages have been
// delivered already. It should not be called while holding the lock.
func (s *Subscription) processNextMsgDelivered(msg *Msg) error {
	s.mu.Lock()
	nc := s.conn
	max := s.max

	var fcReply string
	// Update some stats.
	s.delivered++
	delivered := s.delivered
	if s.jsi != nil {
		fcReply = s.checkForFlowControlResponse()
	}

	if s.typ == SyncSubscription {
		s.pMsgs--
		s.pBytes -= len(msg.Data)
	}
	s.mu.Unlock()

	if fcReply != _EMPTY_ {
		nc.Publish(fcReply, nil)
	}

	if max > 0 {
		if delivered > max {
			return ErrMaxMessages
		}
		// Remove subscription if we have reached max.
		if delivered == max {
			nc.mu.Lock()
			nc.removeSub(s)
			nc.mu.Unlock()
		}
	}
	if len(msg.Data) == 0 && msg.Header.Get(statusHdr) == noResponders {
		return ErrNoResponders
	}

	return nil
}

// Queued returns the number of queued messages in the client for this subscription.
//
// Deprecated: Use Pending()
func (s *Subscription) QueuedMsgs() (int, error) {
	m, _, err := s.Pending()
	return int(m), err
}

// Pending returns the number of queued messages and queued bytes in the client for this subscription.
func (s *Subscription) Pending() (int, int, error) {
	if s == nil {
		return -1, -1, ErrBadSubscription
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil || s.closed {
		return -1, -1, ErrBadSubscription
	}
	if s.typ == ChanSubscription {
		return -1, -1, ErrTypeSubscription
	}
	return s.pMsgs, s.pBytes, nil
}

// MaxPending returns the maximum number of queued messages and queued bytes seen so far.
func (s *Subscription) MaxPending() (int, int, error) {
	if s == nil {
		return -1, -1, ErrBadSubscription
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil || s.closed {
		return -1, -1, ErrBadSubscription
	}
	if s.typ == ChanSubscription {
		return -1, -1, ErrTypeSubscription
	}
	return s.pMsgsMax, s.pBytesMax, nil
}

// ClearMaxPending resets the maximums seen so far.
func (s *Subscription) ClearMaxPending() error {
	if s == nil {
		return ErrBadSubscription
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil || s.closed {
		return ErrBadSubscription
	}
	if s.typ == ChanSubscription {
		return ErrTypeSubscription
	}
	s.pMsgsMax, s.pBytesMax = 0, 0
	return nil
}

// Pending Limits
const (
	// DefaultSubPendingMsgsLimit will be 500k msgs.
	DefaultSubPendingMsgsLimit = 500_000
	// DefaultSubPendingBytesLimit is 64MB
	DefaultSubPendingBytesLimit = 64 * 1024 * 1024
)

// PendingLimits returns the current limits for this subscription.
// If no error is returned, a negative value indicates that the
// given metric is not limited.
func (s *Subscription) PendingLimits() (int, int, error) {
	if s == nil {
		return -1, -1, ErrBadSubscription
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil || s.closed {
		return -1, -1, ErrBadSubscription
	}
	if s.typ == ChanSubscription {
		return -1, -1, ErrTypeSubscription
	}
	return s.pMsgsLimit, s.pBytesLimit, nil
}

// SetPendingLimits sets the limits for pending msgs and bytes for this subscription.
// Zero is not allowed. Any negative value means that the given metric is not limited.
func (s *Subscription) SetPendingLimits(msgLimit, bytesLimit int) error {
	if s == nil {
		return ErrBadSubscription
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil || s.closed {
		return ErrBadSubscription
	}
	if s.typ == ChanSubscription {
		return ErrTypeSubscription
	}
	if msgLimit == 0 || bytesLimit == 0 {
		return ErrInvalidArg
	}
	s.pMsgsLimit, s.pBytesLimit = msgLimit, bytesLimit
	return nil
}

// Delivered returns the number of delivered messages for this subscription.
func (s *Subscription) Delivered() (int64, error) {
	if s == nil {
		return -1, ErrBadSubscription
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil || s.closed {
		return -1, ErrBadSubscription
	}
	return int64(s.delivered), nil
}

// Dropped returns the number of known dropped messages for this subscription.
// This will correspond to messages dropped by violations of PendingLimits. If
// the server declares the connection a SlowConsumer, this number may not be
// valid.
func (s *Subscription) Dropped() (int, error) {
	if s == nil {
		return -1, ErrBadSubscription
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil || s.closed {
		return -1, ErrBadSubscription
	}
	return s.dropped, nil
}

// Respond allows a convenient way to respond to requests in service based subscriptions.
func (m *Msg) Respond(data []byte) error {
	if m == nil || m.Sub == nil {
		return ErrMsgNotBound
	}
	if m.Reply == "" {
		return ErrMsgNoReply
	}
	m.Sub.mu.Lock()
	nc := m.Sub.conn
	m.Sub.mu.Unlock()
	// No need to check the connection here since the call to publish will do all the checking.
	return nc.Publish(m.Reply, data)
}

// RespondMsg allows a convenient way to respond to requests in service based subscriptions that might include headers
func (m *Msg) RespondMsg(msg *Msg) error {
	if m == nil || m.Sub == nil {
		return ErrMsgNotBound
	}
	if m.Reply == "" {
		return ErrMsgNoReply
	}
	msg.Subject = m.Reply
	m.Sub.mu.Lock()
	nc := m.Sub.conn
	m.Sub.mu.Unlock()
	// No need to check the connection here since the call to publish will do all the checking.
	return nc.PublishMsg(msg)
}

// FIXME: This is a hack
// removeFlushEntry is needed when we need to discard queued up responses
// for our pings as part of a flush call. This happens when we have a flush
// call outstanding and we call close.
func (nc *Conn) removeFlushEntry(ch chan struct{}) bool {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if nc.pongs == nil {
		return false
	}
	for i, c := range nc.pongs {
		if c == ch {
			nc.pongs[i] = nil
			return true
		}
	}
	return false
}

// The lock must be held entering this function.
func (nc *Conn) sendPing(ch chan struct{}) {
	nc.pongs = append(nc.pongs, ch)
	nc.bw.appendString(pingProto)
	// Flush in place.
	nc.bw.flush()
}

// This will fire periodically and send a client origin
// ping to the server. Will also check that we have received
// responses from the server.
func (nc *Conn) processPingTimer() {
	nc.mu.Lock()

	if nc.status != CONNECTED {
		nc.mu.Unlock()
		return
	}

	// Check for violation
	nc.pout++
	if nc.pout > nc.Opts.MaxPingsOut {
		nc.mu.Unlock()
		if shouldClose := nc.processOpErr(ErrStaleConnection, false); shouldClose {
			nc.close(CLOSED, true, nil)
		}
		return
	}

	nc.sendPing(nil)
	nc.ptmr.Reset(nc.Opts.PingInterval)
	nc.mu.Unlock()
}

// FlushTimeout allows a Flush operation to have an associated timeout.
func (nc *Conn) FlushTimeout(timeout time.Duration) (err error) {
	if nc == nil {
		return ErrInvalidConnection
	}
	if timeout <= 0 {
		return ErrBadTimeout
	}

	nc.mu.Lock()
	if nc.isClosed() {
		nc.mu.Unlock()
		return ErrConnectionClosed
	}
	t := globalTimerPool.Get(timeout)
	defer globalTimerPool.Put(t)

	// Create a buffered channel to prevent chan send to block
	// in processPong() if this code here times out just when
	// PONG was received.
	ch := make(chan struct{}, 1)
	nc.sendPing(ch)
	nc.mu.Unlock()

	select {
	case _, ok := <-ch:
		if !ok {
			err = ErrConnectionClosed
		} else {
			close(ch)
		}
	case <-t.C:
		err = ErrTimeout
	}

	if err != nil {
		nc.removeFlushEntry(ch)
	}
	return
}

// RTT calculates the round trip time between this client and the server.
func (nc *Conn) RTT() (time.Duration, error) {
	if nc.IsClosed() {
		return 0, ErrConnectionClosed
	}
	if nc.IsReconnecting() {
		return 0, ErrDisconnected
	}
	start := time.Now()
	if err := nc.FlushTimeout(10 * time.Second); err != nil {
		return 0, err
	}
	return time.Since(start), nil
}

// Flush will perform a round trip to the server and return when it
// receives the internal reply.
func (nc *Conn) Flush() error {
	return nc.FlushTimeout(10 * time.Second)
}

// Buffered will return the number of bytes buffered to be sent to the server.
// FIXME(dlc) take into account disconnected state.
func (nc *Conn) Buffered() (int, error) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	if nc.isClosed() || nc.bw == nil {
		return -1, ErrConnectionClosed
	}
	return nc.bw.buffered(), nil
}

// resendSubscriptions will send our subscription state back to the
// server. Used in reconnects
func (nc *Conn) resendSubscriptions() {
	// Since we are going to send protocols to the server, we don't want to
	// be holding the subsMu lock (which is used in processMsg). So copy
	// the subscriptions in a temporary array.
	nc.subsMu.RLock()
	subs := make([]*Subscription, 0, len(nc.subs))
	for _, s := range nc.subs {
		subs = append(subs, s)
	}
	nc.subsMu.RUnlock()
	for _, s := range subs {
		adjustedMax := uint64(0)
		s.mu.Lock()
		// when resending subscriptions, the permissions error should be cleared
		// since the user may have fixed the permissions issue
		s.permissionsErr = nil
		if s.max > 0 {
			if s.delivered < s.max {
				adjustedMax = s.max - s.delivered
			}
			// adjustedMax could be 0 here if the number of delivered msgs
			// reached the max, if so unsubscribe.
			if adjustedMax == 0 {
				s.mu.Unlock()
				nc.bw.writeDirect(fmt.Sprintf(unsubProto, s.sid, _EMPTY_))
				continue
			}
		}
		subj, queue, sid := s.Subject, s.Queue, s.sid
		s.mu.Unlock()

		nc.bw.writeDirect(fmt.Sprintf(subProto, subj, queue, sid))
		if adjustedMax > 0 {
			maxStr := strconv.Itoa(int(adjustedMax))
			nc.bw.writeDirect(fmt.Sprintf(unsubProto, sid, maxStr))
		}
	}
}

// This will clear any pending flush calls and release pending calls.
// Lock is assumed to be held by the caller.
func (nc *Conn) clearPendingFlushCalls() {
	// Clear any queued pongs, e.g. pending flush calls.
	for _, ch := range nc.pongs {
		if ch != nil {
			close(ch)
		}
	}
	nc.pongs = nil
}

// This will clear any pending Request calls.
// Lock is assumed to be held by the caller.
func (nc *Conn) clearPendingRequestCalls() {
	for key, ch := range nc.respMap {
		if ch != nil {
			close(ch)
			delete(nc.respMap, key)
		}
	}
}

// Low level close call that will do correct cleanup and set
// desired status. Also controls whether user defined callbacks
// will be triggered. The lock should not be held entering this
// function. This function will handle the locking manually.
func (nc *Conn) close(status Status, doCBs bool, err error) {
	nc.mu.Lock()
	if nc.isClosed() {
		nc.status = status
		nc.mu.Unlock()
		return
	}
	nc.status = CLOSED

	// Kick the Go routines so they fall out.
	nc.kickFlusher()

	// If the reconnect timer is waiting between a reconnect attempt,
	// this will kick it out.
	if nc.rqch != nil {
		close(nc.rqch)
		nc.rqch = nil
	}

	// Clear any queued pongs, e.g. pending flush calls.
	nc.clearPendingFlushCalls()

	// Clear any queued and blocking Requests.
	nc.clearPendingRequestCalls()

	// Stop ping timer if set.
	nc.stopPingTimer()
	nc.ptmr = nil

	// Need to close and set TCP conn to nil if reconnect loop has stopped,
	// otherwise we would incorrectly invoke Disconnect handler (if set)
	// down below.
	if nc.ar && nc.conn != nil {
		nc.conn.Close()
		nc.conn = nil
	} else if nc.conn != nil {
		// Go ahead and make sure we have flushed the outbound
		nc.bw.flush()
		defer nc.conn.Close()
	}

	// Close sync subscriber channels and release any
	// pending NextMsg() calls.
	nc.subsMu.Lock()
	for _, s := range nc.subs {
		s.mu.Lock()

		// Release callers on NextMsg for SyncSubscription only
		if s.mch != nil && s.typ == SyncSubscription {
			close(s.mch)
		}
		s.mch = nil

		// Call closed handler for non-AsyncSubscription types (AsyncSubscription handlers
		// are called by waitForMsgs when it exits)
		var done func(string)
		if s.typ != AsyncSubscription && s.pDone != nil {
			done = s.pDone
		}

		// Mark as invalid, for signaling to waitForMsgs
		s.closed = true
		// Mark connection closed in subscription
		s.connClosed = true
		// If we have an async subscription, signals it to exit
		if s.typ == AsyncSubscription && s.pCond != nil {
			s.pCond.Signal()
		}

		s.mu.Unlock()

		// Call the closed handler outside the lock to avoid potential deadlocks
		if done != nil {
			done(s.Subject)
		}
	}
	nc.subs = nil
	nc.subsMu.Unlock()

	nc.changeConnStatus(status)

	// Perform appropriate callback if needed for a disconnect.
	if doCBs {
		if nc.conn != nil {
			if disconnectedErrCB := nc.Opts.DisconnectedErrCB; disconnectedErrCB != nil {
				nc.ach.push(func() { disconnectedErrCB(nc, err) })
			} else if disconnectedCB := nc.Opts.DisconnectedCB; disconnectedCB != nil {
				nc.ach.push(func() { disconnectedCB(nc) })
			}
		}
		if closedCB := nc.Opts.ClosedCB; closedCB != nil {
			nc.ach.push(func() { closedCB(nc) })
		}
	}
	// If this is terminal, then we have to notify the asyncCB handler that
	// it can exit once all async callbacks have been dispatched.
	if status == CLOSED {
		nc.ach.close()
	}
	nc.mu.Unlock()
}

// Close will close the connection to the server. This call will release
// all blocking calls, such as Flush() and NextMsg()
func (nc *Conn) Close() {
	if nc != nil {
		// This will be a no-op if the connection was not websocket.
		// We do this here as opposed to inside close() because we want
		// to do this only for the final user-driven close of the client.
		// Otherwise, we would need to change close() to pass a boolean
		// indicating that this is the case.
		nc.wsClose()
		nc.close(CLOSED, !nc.Opts.NoCallbacksAfterClientClose, nil)
	}
}

// IsClosed tests if a Conn has been closed.
func (nc *Conn) IsClosed() bool {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.isClosed()
}

// IsReconnecting tests if a Conn is reconnecting.
func (nc *Conn) IsReconnecting() bool {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.isReconnecting()
}

// IsConnected tests if a Conn is connected.
func (nc *Conn) IsConnected() bool {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.isConnected()
}

// drainConnection will run in a separate Go routine and will
// flush all publishes and drain all active subscriptions.
func (nc *Conn) drainConnection() {
	// Snapshot subs list.
	nc.mu.Lock()

	// Check again here if we are in a state to not process.
	if nc.isClosed() {
		nc.mu.Unlock()
		return
	}
	if nc.isConnecting() || nc.isReconnecting() {
		nc.mu.Unlock()
		// Move to closed state.
		nc.Close()
		return
	}

	subs := make([]*Subscription, 0, len(nc.subs))
	for _, s := range nc.subs {
		if s == nc.respMux {
			// Skip since might be in use while messages
			// are being processed (can miss responses).
			continue
		}
		subs = append(subs, s)
	}
	errCB := nc.Opts.AsyncErrorCB
	drainWait := nc.Opts.DrainTimeout
	respMux := nc.respMux
	nc.mu.Unlock()

	// for pushing errors with context.
	pushErr := func(err error) {
		nc.mu.Lock()
		nc.err = err
		if errCB != nil {
			nc.ach.push(func() { errCB(nc, nil, err) })
		}
		nc.mu.Unlock()
	}

	// Do subs first, skip request handler if present.
	for _, s := range subs {
		if err := s.Drain(); err != nil {
			// We will notify about these but continue.
			pushErr(err)
		}
	}

	// Wait for the subscriptions to drop to zero.
	timeout := time.Now().Add(drainWait)
	var min int
	if respMux != nil {
		min = 1
	} else {
		min = 0
	}
	for time.Now().Before(timeout) {
		if nc.NumSubscriptions() == min {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	// In case there was a request/response handler
	// then need to call drain at the end.
	if respMux != nil {
		if err := respMux.Drain(); err != nil {
			// We will notify about these but continue.
			pushErr(err)
		}
		for time.Now().Before(timeout) {
			if nc.NumSubscriptions() == 0 {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
	}

	// Check if we timed out.
	if nc.NumSubscriptions() != 0 {
		pushErr(ErrDrainTimeout)
	}

	// Flip State
	nc.mu.Lock()
	nc.changeConnStatus(DRAINING_PUBS)
	nc.mu.Unlock()

	// Do publish drain via Flush() call.
	err := nc.FlushTimeout(5 * time.Second)
	if err != nil {
		pushErr(err)
	}

	// Move to closed state.
	nc.Close()
}

// Drain will put a connection into a drain state. All subscriptions will
// immediately be put into a drain state. Upon completion, the publishers
// will be drained and can not publish any additional messages. Upon draining
// of the publishers, the connection will be closed. Use the ClosedCB
// option to know when the connection has moved from draining to closed.
//
// See note in Subscription.Drain for JetStream subscriptions.
func (nc *Conn) Drain() error {
	nc.mu.Lock()
	if nc.isClosed() {
		nc.mu.Unlock()
		return ErrConnectionClosed
	}
	if nc.isConnecting() || nc.isReconnecting() {
		nc.mu.Unlock()
		nc.Close()
		return ErrConnectionReconnecting
	}
	if nc.isDraining() {
		nc.mu.Unlock()
		return nil
	}
	nc.changeConnStatus(DRAINING_SUBS)
	go nc.drainConnection()
	nc.mu.Unlock()

	return nil
}

// IsDraining tests if a Conn is in the draining state.
func (nc *Conn) IsDraining() bool {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.isDraining()
}

// caller must lock
func (nc *Conn) getServers(implicitOnly bool) []string {
	poolSize := len(nc.srvPool)
	servers := make([]string, 0)
	for i := 0; i < poolSize; i++ {
		if implicitOnly && !nc.srvPool[i].isImplicit {
			continue
		}
		url := nc.srvPool[i].URL
		servers = append(servers, fmt.Sprintf("%s://%s", url.Scheme, url.Host))
	}
	return servers
}

// Servers returns the list of known server urls, including additional
// servers discovered after a connection has been established.  If
// authentication is enabled, use UserInfo or Token when connecting with
// these urls.
func (nc *Conn) Servers() []string {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.getServers(false)
}

// DiscoveredServers returns only the server urls that have been discovered
// after a connection has been established. If authentication is enabled,
// use UserInfo or Token when connecting with these urls.
func (nc *Conn) DiscoveredServers() []string {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.getServers(true)
}

// Status returns the current state of the connection.
func (nc *Conn) Status() Status {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.status
}

// Test if Conn has been closed Lock is assumed held.
func (nc *Conn) isClosed() bool {
	return nc.status == CLOSED
}

// Test if Conn is in the process of connecting
func (nc *Conn) isConnecting() bool {
	return nc.status == CONNECTING
}

// Test if Conn is being reconnected.
func (nc *Conn) isReconnecting() bool {
	return nc.status == RECONNECTING
}

// Test if Conn is connected or connecting.
func (nc *Conn) isConnected() bool {
	return nc.status == CONNECTED || nc.isDraining()
}

// Test if Conn is in the draining state.
func (nc *Conn) isDraining() bool {
	return nc.status == DRAINING_SUBS || nc.status == DRAINING_PUBS
}

// Test if Conn is in the draining state for pubs.
func (nc *Conn) isDrainingPubs() bool {
	return nc.status == DRAINING_PUBS
}

// Stats will return a race safe copy of the Statistics section for the connection.
func (nc *Conn) Stats() Statistics {
	// Stats are updated either under connection's mu or with atomic operations
	// for inbound stats in processMsg().
	nc.mu.Lock()
	stats := Statistics{
		InMsgs:     atomic.LoadUint64(&nc.InMsgs),
		InBytes:    atomic.LoadUint64(&nc.InBytes),
		OutMsgs:    nc.OutMsgs,
		OutBytes:   nc.OutBytes,
		Reconnects: nc.Reconnects,
	}
	nc.mu.Unlock()
	return stats
}

// MaxPayload returns the size limit that a message payload can have.
// This is set by the server configuration and delivered to the client
// upon connect.
func (nc *Conn) MaxPayload() int64 {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.info.MaxPayload
}

// HeadersSupported will return if the server supports headers
func (nc *Conn) HeadersSupported() bool {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.info.Headers
}

// AuthRequired will return if the connected server requires authorization.
func (nc *Conn) AuthRequired() bool {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.info.AuthRequired
}

// TLSRequired will return if the connected server requires TLS connections.
func (nc *Conn) TLSRequired() bool {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.info.TLSRequired
}

// Barrier schedules the given function `f` to all registered asynchronous
// subscriptions.
// Only the last subscription to see this barrier will invoke the function.
// If no subscription is registered at the time of this call, `f()` is invoked
// right away.
// ErrConnectionClosed is returned if the connection is closed prior to
// the call.
func (nc *Conn) Barrier(f func()) error {
	nc.mu.Lock()
	if nc.isClosed() {
		nc.mu.Unlock()
		return ErrConnectionClosed
	}
	nc.subsMu.Lock()
	// Need to figure out how many non chan subscriptions there are
	numSubs := 0
	for _, sub := range nc.subs {
		if sub.typ == AsyncSubscription {
			numSubs++
		}
	}
	if numSubs == 0 {
		nc.subsMu.Unlock()
		nc.mu.Unlock()
		f()
		return nil
	}
	barrier := &barrierInfo{refs: int64(numSubs), f: f}
	for _, sub := range nc.subs {
		sub.mu.Lock()
		if sub.mch == nil {
			msg := &Msg{barrier: barrier}
			// Push onto the async pList
			if sub.pTail != nil {
				sub.pTail.next = msg
			} else {
				sub.pHead = msg
				sub.pCond.Signal()
			}
			sub.pTail = msg
		}
		sub.mu.Unlock()
	}
	nc.subsMu.Unlock()
	nc.mu.Unlock()
	return nil
}

// ServerPool returns a copy of the current server pool for the connection.
//
// This function should not be called from within connection callbacks to avoid
// potential deadlocks.
func (nc *Conn) ServerPool() []Server {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	servers := make([]Server, len(nc.srvPool))
	for i, srv := range nc.srvPool {
		if srv != nil {
			// Return a copy to avoid exposing internal state
			servers[i] = srv.clone()
		}
	}
	return servers
}

// SetServerPool allows updating the server pool for the connection. This
// replaces the existing pool with the provided list of server URLs. If the
// current server is not in the new pool, the client will switch to a server in
// the new pool on the next reconnect attempt. This function is thread-safe and
// can be called while the connection is active. It will return an error if the
// connection is closed or if any of the provided URLs are invalid.
//
// This function does not trigger an immediate reconnect. The new server
// pool will be used on the next reconnect attempt.
// If you want to trigger an immediate reconnect to apply the new server pool,
// you can call [Conn.ForceReconnect] after this function.
//
// Unless [IgnoreDiscoveredServers] is used true, the client will continue to
// discover and add new servers to the pool as it receives INFO messages from
// the server. If you want to prevent this behavior and only use the servers
// provided in SetServerPool, use [IgnoreDiscoveredServers].
//
// This function should not be called from within connection callbacks to avoid
// potential deadlocks.
func (nc *Conn) SetServerPool(servers []string) error {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if nc.isClosed() {
		return ErrConnectionClosed
	}

	// Parse and validate all URLs first (without modifying state)
	newPool := make([]*Server, 0, len(servers))
	newURLs := make(map[string]struct{})

	for _, addr := range servers {
		s, err := nc.parseServerURL(addr, false, false)
		if err != nil {
			return err
		}
		if isWebsocketScheme(s.URL) != nc.ws {
			return ErrMixingWebsocketSchemes
		}

		newPool = append(newPool, s)
		newURLs[s.URL.Host] = struct{}{}
	}

	// Preserve state from existing pool entries
	for _, newSrv := range newPool {
		if idx := slices.IndexFunc(nc.srvPool, func(oldSrv *Server) bool {
			return oldSrv != nil && oldSrv.URL.String() == newSrv.URL.String()
		}); idx != -1 {
			newSrv.Reconnects = nc.srvPool[idx].Reconnects
			newSrv.didConnect = nc.srvPool[idx].didConnect
			newSrv.lastErr = nc.srvPool[idx].lastErr
		}
	}

	nc.srvPool = newPool
	nc.urls = newURLs

	// Update nc.current to point to the corresponding server in the new pool
	// This is important because currentServer() uses pointer equality
	if nc.current != nil {
		currentURL := nc.current.URL.String()
		found := false
		for _, s := range newPool {
			if s.URL.String() == currentURL {
				// Update nc.current to point to the server instance in the new pool
				nc.current = s
				found = true
				break
			}
		}
		if !found && len(newPool) > 0 {
			// Current server not in new pool - point to first server in new pool
			// This ensures selectNextServer() can find it and properly rotate
			nc.current = newPool[0]
		}
	}

	return nil
}

// GetClientIP returns the client IP as known by the server.
// Supported as of server version 2.1.6.
func (nc *Conn) GetClientIP() (net.IP, error) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	if nc.isClosed() {
		return nil, ErrConnectionClosed
	}
	if nc.info.ClientIP == "" {
		return nil, ErrClientIPNotSupported
	}
	ip := net.ParseIP(nc.info.ClientIP)
	return ip, nil
}

// GetClientID returns the client ID assigned by the server to which
// the client is currently connected to. Note that the value may change if
// the client reconnects.
// This function returns ErrClientIDNotSupported if the server is of a
// version prior to 1.2.0.
func (nc *Conn) GetClientID() (uint64, error) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	if nc.isClosed() {
		return 0, ErrConnectionClosed
	}
	if nc.info.CID == 0 {
		return 0, ErrClientIDNotSupported
	}
	return nc.info.CID, nil
}

// StatusChanged returns a channel on which given list of connection status changes will be reported.
// If no statuses are provided, defaults will be used: CONNECTED, RECONNECTING, DISCONNECTED, CLOSED.
func (nc *Conn) StatusChanged(statuses ...Status) chan Status {
	if len(statuses) == 0 {
		statuses = []Status{CONNECTED, RECONNECTING, DISCONNECTED, CLOSED}
	}
	ch := make(chan Status, 10)
	nc.mu.Lock()
	defer nc.mu.Unlock()
	for _, s := range statuses {
		nc.registerStatusChangeListener(s, ch)
	}
	return ch
}

// RemoveStatusListener removes a status change listener.
// If the channel is not closed, it will be closed.
// Listeners will be removed automatically on status change
// as well, but this is a way to remove them manually.
func (nc *Conn) RemoveStatusListener(ch chan (Status)) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	select {
	case <-ch:
	default:
		close(ch)
	}

	for _, listeners := range nc.statListeners {
		for range listeners {
			delete(listeners, ch)
		}
	}
}

// registerStatusChangeListener registers a channel waiting for a specific status change event.
// Status change events are non-blocking - if no receiver is waiting for the status change,
// it will not be sent on the channel. Closed channels are ignored.
// The lock should be held entering.
func (nc *Conn) registerStatusChangeListener(status Status, ch chan Status) {
	if nc.statListeners == nil {
		nc.statListeners = make(map[Status]map[chan Status]struct{})
	}
	if _, ok := nc.statListeners[status]; !ok {
		nc.statListeners[status] = make(map[chan Status]struct{})
	}
	nc.statListeners[status][ch] = struct{}{}
}

// sendStatusEvent sends connection status event to all channels.
// If channel is closed, or there is no listener, sendStatusEvent
// will not block. Lock should be held entering.
func (nc *Conn) sendStatusEvent(s Status) {
Loop:
	for ch := range nc.statListeners[s] {
		// make sure channel is not closed
		select {
		case <-ch:
			// if chan is closed, remove it
			delete(nc.statListeners[s], ch)
			continue Loop
		default:
		}
		// only send event if someone's listening
		select {
		case ch <- s:
		default:
		}
	}
}

// changeConnStatus changes connections status and sends events
// to all listeners. Lock should be held entering.
func (nc *Conn) changeConnStatus(status Status) {
	if nc == nil {
		return
	}
	nc.sendStatusEvent(status)
	nc.status = status
}

// NkeyOptionFromSeed will load an nkey pair from a seed file.
// It will return the NKey Option and will handle
// signing of nonce challenges from the server. It will take
// care to not hold keys in memory and to wipe memory.
func NkeyOptionFromSeed(seedFile string) (Option, error) {
	kp, err := nkeyPairFromSeedFile(seedFile)
	if err != nil {
		return nil, err
	}
	// Wipe our key on exit.
	defer kp.Wipe()

	pub, err := kp.PublicKey()
	if err != nil {
		return nil, err
	}
	if !nkeys.IsValidPublicUserKey(pub) {
		return nil, errors.New("nats: Not a valid nkey user seed")
	}
	sigCB := func(nonce []byte) ([]byte, error) {
		return sigHandler(nonce, seedFile)
	}
	return Nkey(string(pub), sigCB), nil
}

// Just wipe slice with 'x', for clearing contents of creds or nkey seed file.
func wipeSlice(buf []byte) {
	for i := range buf {
		buf[i] = 'x'
	}
}

func userFromFile(userFile string) (string, error) {
	path, err := expandPath(userFile)
	if err != nil {
		return _EMPTY_, fmt.Errorf("nats: %w", err)
	}

	contents, err := os.ReadFile(path)
	if err != nil {
		return _EMPTY_, fmt.Errorf("nats: %w", err)
	}
	defer wipeSlice(contents)
	return nkeys.ParseDecoratedJWT(contents)
}

func homeDir() (string, error) {
	if runtime.GOOS == "windows" {
		homeDrive, homePath := os.Getenv("HOMEDRIVE"), os.Getenv("HOMEPATH")
		userProfile := os.Getenv("USERPROFILE")

		var home string
		if homeDrive == "" || homePath == "" {
			if userProfile == "" {
				return _EMPTY_, errors.New("nats: failed to get home dir, require %HOMEDRIVE% and %HOMEPATH% or %USERPROFILE%")
			}
			home = userProfile
		} else {
			home = filepath.Join(homeDrive, homePath)
		}

		return home, nil
	}

	home := os.Getenv("HOME")
	if home == "" {
		return _EMPTY_, errors.New("nats: failed to get home dir, require $HOME")
	}
	return home, nil
}

func expandPath(p string) (string, error) {
	p = os.ExpandEnv(p)

	if !strings.HasPrefix(p, "~") {
		return p, nil
	}

	home, err := homeDir()
	if err != nil {
		return _EMPTY_, err
	}

	return filepath.Join(home, p[1:]), nil
}

func nkeyPairFromSeedFile(seedFile string) (nkeys.KeyPair, error) {
	contents, err := os.ReadFile(seedFile)
	if err != nil {
		return nil, fmt.Errorf("nats: %w", err)
	}
	defer wipeSlice(contents)
	return nkeys.ParseDecoratedNKey(contents)
}

// Sign authentication challenges from the server.
// Do not keep private seed in memory.
func sigHandler(nonce []byte, seedFile string) ([]byte, error) {
	kp, err := nkeyPairFromSeedFile(seedFile)
	if err != nil {
		return nil, fmt.Errorf("unable to extract key pair from file %q: %w", seedFile, err)
	}
	// Wipe our key on exit.
	defer kp.Wipe()

	sig, _ := kp.Sign(nonce)
	return sig, nil
}

type timeoutWriter struct {
	timeout time.Duration
	conn    net.Conn
}

// Write implements the io.Writer interface.
func (tw *timeoutWriter) Write(p []byte) (int, error) {
	tw.conn.SetWriteDeadline(time.Now().Add(tw.timeout))
	n, err := tw.conn.Write(p)
	tw.conn.SetWriteDeadline(time.Time{})
	return n, err
}
