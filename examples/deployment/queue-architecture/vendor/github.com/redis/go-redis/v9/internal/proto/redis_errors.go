package proto

import (
	"errors"
	"strings"
)

// Typed Redis errors for better error handling with wrapping support.
// These errors maintain backward compatibility by keeping the same error messages.

// LoadingError is returned when Redis is loading the dataset in memory.
type LoadingError struct {
	msg string
}

func (e *LoadingError) Error() string {
	return e.msg
}

func (e *LoadingError) RedisError() {}

// NewLoadingError creates a new LoadingError with the given message.
func NewLoadingError(msg string) *LoadingError {
	return &LoadingError{msg: msg}
}

// ReadOnlyError is returned when trying to write to a read-only replica.
type ReadOnlyError struct {
	msg string
}

func (e *ReadOnlyError) Error() string {
	return e.msg
}

func (e *ReadOnlyError) RedisError() {}

// NewReadOnlyError creates a new ReadOnlyError with the given message.
func NewReadOnlyError(msg string) *ReadOnlyError {
	return &ReadOnlyError{msg: msg}
}

// MovedError is returned when a key has been moved to a different node in a cluster.
type MovedError struct {
	msg  string
	addr string
}

func (e *MovedError) Error() string {
	return e.msg
}

func (e *MovedError) RedisError() {}

// Addr returns the address of the node where the key has been moved.
func (e *MovedError) Addr() string {
	return e.addr
}

// NewMovedError creates a new MovedError with the given message and address.
func NewMovedError(msg string, addr string) *MovedError {
	return &MovedError{msg: msg, addr: addr}
}

// AskError is returned when a key is being migrated and the client should ask another node.
type AskError struct {
	msg  string
	addr string
}

func (e *AskError) Error() string {
	return e.msg
}

func (e *AskError) RedisError() {}

// Addr returns the address of the node to ask.
func (e *AskError) Addr() string {
	return e.addr
}

// NewAskError creates a new AskError with the given message and address.
func NewAskError(msg string, addr string) *AskError {
	return &AskError{msg: msg, addr: addr}
}

// ClusterDownError is returned when the cluster is down.
type ClusterDownError struct {
	msg string
}

func (e *ClusterDownError) Error() string {
	return e.msg
}

func (e *ClusterDownError) RedisError() {}

// NewClusterDownError creates a new ClusterDownError with the given message.
func NewClusterDownError(msg string) *ClusterDownError {
	return &ClusterDownError{msg: msg}
}

// TryAgainError is returned when a command cannot be processed and should be retried.
type TryAgainError struct {
	msg string
}

func (e *TryAgainError) Error() string {
	return e.msg
}

func (e *TryAgainError) RedisError() {}

// NewTryAgainError creates a new TryAgainError with the given message.
func NewTryAgainError(msg string) *TryAgainError {
	return &TryAgainError{msg: msg}
}

// MasterDownError is returned when the master is down.
type MasterDownError struct {
	msg string
}

func (e *MasterDownError) Error() string {
	return e.msg
}

func (e *MasterDownError) RedisError() {}

// NewMasterDownError creates a new MasterDownError with the given message.
func NewMasterDownError(msg string) *MasterDownError {
	return &MasterDownError{msg: msg}
}

// MaxClientsError is returned when the maximum number of clients has been reached.
type MaxClientsError struct {
	msg string
}

func (e *MaxClientsError) Error() string {
	return e.msg
}

func (e *MaxClientsError) RedisError() {}

// NewMaxClientsError creates a new MaxClientsError with the given message.
func NewMaxClientsError(msg string) *MaxClientsError {
	return &MaxClientsError{msg: msg}
}

// AuthError is returned when authentication fails.
type AuthError struct {
	msg string
}

func (e *AuthError) Error() string {
	return e.msg
}

func (e *AuthError) RedisError() {}

// NewAuthError creates a new AuthError with the given message.
func NewAuthError(msg string) *AuthError {
	return &AuthError{msg: msg}
}

// PermissionError is returned when a user lacks required permissions.
type PermissionError struct {
	msg string
}

func (e *PermissionError) Error() string {
	return e.msg
}

func (e *PermissionError) RedisError() {}

// NewPermissionError creates a new PermissionError with the given message.
func NewPermissionError(msg string) *PermissionError {
	return &PermissionError{msg: msg}
}

// ExecAbortError is returned when a transaction is aborted.
type ExecAbortError struct {
	msg string
}

func (e *ExecAbortError) Error() string {
	return e.msg
}

func (e *ExecAbortError) RedisError() {}

// NewExecAbortError creates a new ExecAbortError with the given message.
func NewExecAbortError(msg string) *ExecAbortError {
	return &ExecAbortError{msg: msg}
}

// OOMError is returned when Redis is out of memory.
type OOMError struct {
	msg string
}

func (e *OOMError) Error() string {
	return e.msg
}

func (e *OOMError) RedisError() {}

// NewOOMError creates a new OOMError with the given message.
func NewOOMError(msg string) *OOMError {
	return &OOMError{msg: msg}
}

// NoReplicasError is returned when not enough replicas acknowledge a write.
// This error occurs when using WAIT/WAITAOF commands or CLUSTER SETSLOT with
// synchronous replication, and the required number of replicas cannot confirm
// the write within the timeout period.
type NoReplicasError struct {
	msg string
}

func (e *NoReplicasError) Error() string {
	return e.msg
}

func (e *NoReplicasError) RedisError() {}

// NewNoReplicasError creates a new NoReplicasError with the given message.
func NewNoReplicasError(msg string) *NoReplicasError {
	return &NoReplicasError{msg: msg}
}

// parseTypedRedisError parses a Redis error message and returns a typed error if applicable.
// This function maintains backward compatibility by keeping the same error messages.
func parseTypedRedisError(msg string) error {
	// Check for specific error patterns and return typed errors
	switch {
	case strings.HasPrefix(msg, "LOADING "):
		return NewLoadingError(msg)
	case strings.HasPrefix(msg, "READONLY "):
		return NewReadOnlyError(msg)
	case strings.HasPrefix(msg, "MOVED "):
		// Extract address from "MOVED <slot> <addr>"
		addr := extractAddr(msg)
		return NewMovedError(msg, addr)
	case strings.HasPrefix(msg, "ASK "):
		// Extract address from "ASK <slot> <addr>"
		addr := extractAddr(msg)
		return NewAskError(msg, addr)
	case strings.HasPrefix(msg, "CLUSTERDOWN "):
		return NewClusterDownError(msg)
	case strings.HasPrefix(msg, "TRYAGAIN "):
		return NewTryAgainError(msg)
	case strings.HasPrefix(msg, "MASTERDOWN "):
		return NewMasterDownError(msg)
	case strings.HasPrefix(msg, "NOREPLICAS "):
		return NewNoReplicasError(msg)
	case msg == "ERR max number of clients reached":
		return NewMaxClientsError(msg)
	case strings.HasPrefix(msg, "NOAUTH "), strings.HasPrefix(msg, "WRONGPASS "), strings.Contains(msg, "unauthenticated"):
		return NewAuthError(msg)
	case strings.HasPrefix(msg, "NOPERM "):
		return NewPermissionError(msg)
	case strings.HasPrefix(msg, "EXECABORT "):
		return NewExecAbortError(msg)
	case strings.HasPrefix(msg, "OOM "):
		return NewOOMError(msg)
	default:
		// Return generic RedisError for unknown error types
		return RedisError(msg)
	}
}

// extractAddr extracts the address from MOVED/ASK error messages.
// Format: "MOVED <slot> <addr>" or "ASK <slot> <addr>"
func extractAddr(msg string) string {
	ind := strings.LastIndex(msg, " ")
	if ind == -1 {
		return ""
	}
	return msg[ind+1:]
}

// IsLoadingError checks if an error is a LoadingError, even if wrapped.
func IsLoadingError(err error) bool {
	if err == nil {
		return false
	}
	var loadingErr *LoadingError
	if errors.As(err, &loadingErr) {
		return true
	}
	// Check if wrapped error is a RedisError with LOADING prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "LOADING ") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "LOADING ")
}

// IsReadOnlyError checks if an error is a ReadOnlyError, even if wrapped.
func IsReadOnlyError(err error) bool {
	if err == nil {
		return false
	}
	var readOnlyErr *ReadOnlyError
	if errors.As(err, &readOnlyErr) {
		return true
	}
	// Check if wrapped error is a RedisError with READONLY prefix or Lua script READONLY
	var redisErr RedisError
	if errors.As(err, &redisErr) {
		s := redisErr.Error()
		if strings.HasPrefix(s, "READONLY ") {
			return true
		}
		// Lua script wrapped READONLY errors:
		//   "ERR Error running script (call to f_<sha>): @user_script:N: -READONLY You can't write against a read only replica."
		if strings.Contains(s, "-READONLY You can't write against a read only replica") {
			return true
		}
	}
	// Fallback to string checking for backward compatibility
	s := err.Error()
	if strings.HasPrefix(s, "READONLY ") {
		return true
	}
	return strings.Contains(s, "-READONLY You can't write against a read only replica")
}

// IsMovedError checks if an error is a MovedError, even if wrapped.
// Returns the error and a boolean indicating if it's a MovedError.
func IsMovedError(err error) (*MovedError, bool) {
	if err == nil {
		return nil, false
	}
	var movedErr *MovedError
	if errors.As(err, &movedErr) {
		return movedErr, true
	}
	// Fallback to string checking for backward compatibility
	s := err.Error()
	if strings.HasPrefix(s, "MOVED ") {
		// Parse: MOVED 3999 127.0.0.1:6381
		parts := strings.Split(s, " ")
		if len(parts) == 3 {
			return &MovedError{msg: s, addr: parts[2]}, true
		}
	}
	return nil, false
}

// IsAskError checks if an error is an AskError, even if wrapped.
// Returns the error and a boolean indicating if it's an AskError.
func IsAskError(err error) (*AskError, bool) {
	if err == nil {
		return nil, false
	}
	var askErr *AskError
	if errors.As(err, &askErr) {
		return askErr, true
	}
	// Fallback to string checking for backward compatibility
	s := err.Error()
	if strings.HasPrefix(s, "ASK ") {
		// Parse: ASK 3999 127.0.0.1:6381
		parts := strings.Split(s, " ")
		if len(parts) == 3 {
			return &AskError{msg: s, addr: parts[2]}, true
		}
	}
	return nil, false
}

// IsClusterDownError checks if an error is a ClusterDownError, even if wrapped.
func IsClusterDownError(err error) bool {
	if err == nil {
		return false
	}
	var clusterDownErr *ClusterDownError
	if errors.As(err, &clusterDownErr) {
		return true
	}
	// Check if wrapped error is a RedisError with CLUSTERDOWN prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "CLUSTERDOWN ") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "CLUSTERDOWN ")
}

// IsTryAgainError checks if an error is a TryAgainError, even if wrapped.
func IsTryAgainError(err error) bool {
	if err == nil {
		return false
	}
	var tryAgainErr *TryAgainError
	if errors.As(err, &tryAgainErr) {
		return true
	}
	// Check if wrapped error is a RedisError with TRYAGAIN prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "TRYAGAIN ") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "TRYAGAIN ")
}

// IsMasterDownError checks if an error is a MasterDownError, even if wrapped.
func IsMasterDownError(err error) bool {
	if err == nil {
		return false
	}
	var masterDownErr *MasterDownError
	if errors.As(err, &masterDownErr) {
		return true
	}
	// Check if wrapped error is a RedisError with MASTERDOWN prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "MASTERDOWN ") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "MASTERDOWN ")
}

// IsMaxClientsError checks if an error is a MaxClientsError, even if wrapped.
func IsMaxClientsError(err error) bool {
	if err == nil {
		return false
	}
	var maxClientsErr *MaxClientsError
	if errors.As(err, &maxClientsErr) {
		return true
	}
	// Check if wrapped error is a RedisError with max clients prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "ERR max number of clients reached") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "ERR max number of clients reached")
}

// IsAuthError checks if an error is an AuthError, even if wrapped.
func IsAuthError(err error) bool {
	if err == nil {
		return false
	}
	var authErr *AuthError
	if errors.As(err, &authErr) {
		return true
	}
	// Check if wrapped error is a RedisError with auth error prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) {
		s := redisErr.Error()
		return strings.HasPrefix(s, "NOAUTH ") || strings.HasPrefix(s, "WRONGPASS ") || strings.Contains(s, "unauthenticated")
	}
	// Fallback to string checking for backward compatibility
	s := err.Error()
	return strings.HasPrefix(s, "NOAUTH ") || strings.HasPrefix(s, "WRONGPASS ") || strings.Contains(s, "unauthenticated")
}

// IsPermissionError checks if an error is a PermissionError, even if wrapped.
func IsPermissionError(err error) bool {
	if err == nil {
		return false
	}
	var permErr *PermissionError
	if errors.As(err, &permErr) {
		return true
	}
	// Check if wrapped error is a RedisError with NOPERM prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "NOPERM ") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "NOPERM ")
}

// IsExecAbortError checks if an error is an ExecAbortError, even if wrapped.
func IsExecAbortError(err error) bool {
	if err == nil {
		return false
	}
	var execAbortErr *ExecAbortError
	if errors.As(err, &execAbortErr) {
		return true
	}
	// Check if wrapped error is a RedisError with EXECABORT prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "EXECABORT ") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "EXECABORT ")
}

// IsOOMError checks if an error is an OOMError, even if wrapped.
func IsOOMError(err error) bool {
	if err == nil {
		return false
	}
	var oomErr *OOMError
	if errors.As(err, &oomErr) {
		return true
	}
	// Check if wrapped error is a RedisError with OOM prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "OOM ") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "OOM ")
}

// IsNoReplicasError checks if an error is a NoReplicasError, even if wrapped.
func IsNoReplicasError(err error) bool {
	if err == nil {
		return false
	}
	var noReplicasErr *NoReplicasError
	if errors.As(err, &noReplicasErr) {
		return true
	}
	// Check if wrapped error is a RedisError with NOREPLICAS prefix
	var redisErr RedisError
	if errors.As(err, &redisErr) && strings.HasPrefix(redisErr.Error(), "NOREPLICAS ") {
		return true
	}
	// Fallback to string checking for backward compatibility
	return strings.HasPrefix(err.Error(), "NOREPLICAS ")
}
