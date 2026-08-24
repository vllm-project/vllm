package redis

import (
	"context"
	"errors"
	"io"
	"net"
	"strings"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/internal/proto"
)

// ErrClosed performs any operation on the closed client will return this error.
var ErrClosed = pool.ErrClosed

// ErrPoolExhausted is returned from a pool connection method
// when the maximum number of database connections in the pool has been reached.
var ErrPoolExhausted = pool.ErrPoolExhausted

// ErrPoolTimeout timed out waiting to get a connection from the connection pool.
var ErrPoolTimeout = pool.ErrPoolTimeout

// ErrCrossSlot is returned when keys are used in the same Redis command and
// the keys are not in the same hash slot. This error is returned by Redis
// Cluster and will be returned by the client when TxPipeline or TxPipelined
// is used on a ClusterClient with keys in different slots.
var ErrCrossSlot = proto.RedisError("CROSSSLOT Keys in request don't hash to the same slot")

// ErrNoScript is returned when EVALSHA is requested for a script digest that
// is not available in the script cache. Note that this error text is reproduced
// literally from that used by Redis.
var ErrNoScript = proto.RedisError("NOSCRIPT No matching script. Please use EVAL.")

// HasErrorPrefix checks if the err is a Redis error and the message contains a prefix.
func HasErrorPrefix(err error, prefix string) bool {
	var rErr Error
	if !errors.As(err, &rErr) {
		return false
	}
	msg := rErr.Error()
	msg = strings.TrimPrefix(msg, "ERR ") // KVRocks adds such prefix
	return strings.HasPrefix(msg, prefix)
}

type Error interface {
	error

	// RedisError is a no-op function but
	// serves to distinguish types that are Redis
	// errors from ordinary errors: a type is a
	// Redis error if it has a RedisError method.
	RedisError()
}

var _ Error = proto.RedisError("")

func isContextError(err error) bool {
	// Check for wrapped context errors using errors.Is
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// isTimeoutError checks if an error is a timeout error, even if wrapped.
// Returns (isTimeout, shouldRetryOnTimeout) where:
// - isTimeout: true if the error is any kind of timeout error
// - shouldRetryOnTimeout: true if Timeout() method returns true
func isTimeoutError(err error) (isTimeout bool, hasTimeoutFlag bool) {
	// Check for timeoutError interface (works with wrapped errors)
	var te timeoutError
	if errors.As(err, &te) {
		return true, te.Timeout()
	}

	// Check for net.Error specifically (common case for network timeouts)
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true, netErr.Timeout()
	}

	return false, false
}

func shouldRetry(err error, retryTimeout bool) bool {
	if err == nil {
		return false
	}

	// Check for EOF errors (works with wrapped errors)
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}

	// Dial errors mean TCP connection was never established — safe to retry even
	// when wrapped inside context.DeadlineExceeded (from DialTimeout context).
	// Must be checked before the context error check below.
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op == "dial" {
		return true
	}

	// Check for context errors (works with wrapped errors)
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}

	// Check for pool timeout (works with wrapped errors)
	if errors.Is(err, pool.ErrPoolTimeout) {
		// connection pool timeout, increase retries. #3289
		return true
	}

	// Check for timeout errors (works with wrapped errors)
	if isTimeout, hasTimeoutFlag := isTimeoutError(err); isTimeout {
		if hasTimeoutFlag {
			return retryTimeout
		}
		return true
	}

	// Check for typed Redis errors using errors.As (works with wrapped errors)
	if proto.IsMaxClientsError(err) {
		return true
	}
	if proto.IsLoadingError(err) {
		return true
	}
	if proto.IsReadOnlyError(err) {
		return true
	}
	if proto.IsMasterDownError(err) {
		return true
	}
	if proto.IsClusterDownError(err) {
		return true
	}
	if proto.IsTryAgainError(err) {
		return true
	}
	if proto.IsNoReplicasError(err) {
		return true
	}

	// Fallback to string checking for backward compatibility with plain errors
	s := err.Error()
	if strings.HasPrefix(s, "ERR max number of clients reached") {
		return true
	}
	if strings.HasPrefix(s, "LOADING ") {
		return true
	}
	if strings.HasPrefix(s, "READONLY ") {
		return true
	}
	if strings.Contains(s, "-READONLY You can't write against a read only replica") {
		return true
	}
	if strings.HasPrefix(s, "CLUSTERDOWN ") {
		return true
	}
	if strings.HasPrefix(s, "TRYAGAIN ") {
		return true
	}
	if strings.HasPrefix(s, "MASTERDOWN ") {
		return true
	}
	if strings.HasPrefix(s, "NOREPLICAS ") {
		return true
	}

	// Other server errors are not retried. This includes the logical
	// -SEARCH_TIMEOUT (search-on-timeout fail): retrying would just repeat the
	// same expensive query.
	return false
}

func isRedisError(err error) bool {
	// Check if error implements the Error interface (works with wrapped errors)
	var redisErr Error
	if errors.As(err, &redisErr) {
		return true
	}
	// Also check for proto.RedisError specifically
	var protoRedisErr proto.RedisError
	return errors.As(err, &protoRedisErr)
}

func isBadConn(err error, allowTimeout bool, addr string) bool {
	if err == nil {
		return false
	}

	// Check for context errors (works with wrapped errors)
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	// Check for pool timeout errors (works with wrapped errors)
	if errors.Is(err, pool.ErrConnUnusableTimeout) {
		return true
	}

	if isRedisError(err) {
		switch {
		case isReadOnlyError(err):
			// Close connections in read only state in case domain addr is used
			// and domain resolves to a different Redis Server. See #790.
			return true
		case isMovedSameConnAddr(err, addr):
			// Close connections when we are asked to move to the same addr
			// of the connection. Force a DNS resolution when all connections
			// of the pool are recycled
			return true
		default:
			return false
		}
	}

	if allowTimeout {
		// Check for network timeout errors (works with wrapped errors)
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return false
		}
	}

	return true
}

func isMovedError(err error) (moved bool, ask bool, addr string) {
	// Check for typed MovedError
	if movedErr, ok := proto.IsMovedError(err); ok {
		addr = movedErr.Addr()
		addr = internal.GetAddr(addr)
		return true, false, addr
	}

	// Check for typed AskError
	if askErr, ok := proto.IsAskError(err); ok {
		addr = askErr.Addr()
		addr = internal.GetAddr(addr)
		return false, true, addr
	}

	// Fallback to string checking for backward compatibility
	s := err.Error()
	if strings.HasPrefix(s, "MOVED ") {
		// Parse: MOVED 3999 127.0.0.1:6381
		parts := strings.Split(s, " ")
		if len(parts) == 3 {
			addr = internal.GetAddr(parts[2])
			return true, false, addr
		}
	}
	if strings.HasPrefix(s, "ASK ") {
		// Parse: ASK 3999 127.0.0.1:6381
		parts := strings.Split(s, " ")
		if len(parts) == 3 {
			addr = internal.GetAddr(parts[2])
			return false, true, addr
		}
	}

	return false, false, ""
}

func isLoadingError(err error) bool {
	return proto.IsLoadingError(err)
}

func isReadOnlyError(err error) bool {
	return proto.IsReadOnlyError(err)
}

func isMovedSameConnAddr(err error, addr string) bool {
	if movedErr, ok := proto.IsMovedError(err); ok {
		return strings.HasSuffix(movedErr.Addr(), addr)
	}
	return false
}

//------------------------------------------------------------------------------

// Typed error checking functions for public use.
// These functions work correctly even when errors are wrapped in hooks.

// IsLoadingError checks if an error is a Redis LOADING error, even if wrapped.
// LOADING errors occur when Redis is loading the dataset in memory.
func IsLoadingError(err error) bool {
	return proto.IsLoadingError(err)
}

// IsReadOnlyError checks if an error is a Redis READONLY error, even if wrapped.
// READONLY errors occur when trying to write to a read-only replica.
func IsReadOnlyError(err error) bool {
	return proto.IsReadOnlyError(err)
}

// IsClusterDownError checks if an error is a Redis CLUSTERDOWN error, even if wrapped.
// CLUSTERDOWN errors occur when the cluster is down.
func IsClusterDownError(err error) bool {
	return proto.IsClusterDownError(err)
}

// IsTryAgainError checks if an error is a Redis TRYAGAIN error, even if wrapped.
// TRYAGAIN errors occur when a command cannot be processed and should be retried.
func IsTryAgainError(err error) bool {
	return proto.IsTryAgainError(err)
}

// IsMasterDownError checks if an error is a Redis MASTERDOWN error, even if wrapped.
// MASTERDOWN errors occur when the master is down.
func IsMasterDownError(err error) bool {
	return proto.IsMasterDownError(err)
}

// IsMaxClientsError checks if an error is a Redis max clients error, even if wrapped.
// This error occurs when the maximum number of clients has been reached.
func IsMaxClientsError(err error) bool {
	return proto.IsMaxClientsError(err)
}

// IsMovedError checks if an error is a Redis MOVED error, even if wrapped.
// MOVED errors occur in cluster mode when a key has been moved to a different node.
// Returns the address of the node where the key has been moved and a boolean indicating if it's a MOVED error.
func IsMovedError(err error) (addr string, ok bool) {
	if movedErr, isMovedErr := proto.IsMovedError(err); isMovedErr {
		return movedErr.Addr(), true
	}
	return "", false
}

// IsAskError checks if an error is a Redis ASK error, even if wrapped.
// ASK errors occur in cluster mode when a key is being migrated and the client should ask another node.
// Returns the address of the node to ask and a boolean indicating if it's an ASK error.
func IsAskError(err error) (addr string, ok bool) {
	if askErr, isAskErr := proto.IsAskError(err); isAskErr {
		return askErr.Addr(), true
	}
	return "", false
}

// IsAuthError checks if an error is a Redis authentication error, even if wrapped.
// Authentication errors occur when:
// - NOAUTH: Redis requires authentication but none was provided
// - WRONGPASS: Redis authentication failed due to incorrect password
// - unauthenticated: Error returned when password changed
func IsAuthError(err error) bool {
	return proto.IsAuthError(err)
}

// IsPermissionError checks if an error is a Redis permission error, even if wrapped.
// Permission errors (NOPERM) occur when a user does not have permission to execute a command.
func IsPermissionError(err error) bool {
	return proto.IsPermissionError(err)
}

// IsExecAbortError checks if an error is a Redis EXECABORT error, even if wrapped.
// EXECABORT errors occur when a transaction is aborted.
func IsExecAbortError(err error) bool {
	return proto.IsExecAbortError(err)
}

// IsOOMError checks if an error is a Redis OOM (Out Of Memory) error, even if wrapped.
// OOM errors occur when Redis is out of memory.
func IsOOMError(err error) bool {
	return proto.IsOOMError(err)
}

// IsNoReplicasError checks if an error is a Redis NOREPLICAS error, even if wrapped.
// NOREPLICAS errors occur when not enough replicas acknowledge a write operation.
// This typically happens with WAIT/WAITAOF commands or CLUSTER SETSLOT with synchronous
// replication when the required number of replicas cannot confirm the write within the timeout.
func IsNoReplicasError(err error) bool {
	return proto.IsNoReplicasError(err)
}

//------------------------------------------------------------------------------

type timeoutError interface {
	Timeout() bool
}
