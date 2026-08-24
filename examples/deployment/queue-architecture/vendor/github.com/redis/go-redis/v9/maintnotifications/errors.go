package maintnotifications

import (
	"errors"

	"github.com/redis/go-redis/v9/internal/maintnotifications/logs"
)

// Configuration errors
var (
	ErrInvalidRelaxedTimeout             = errors.New(logs.InvalidRelaxedTimeoutError())
	ErrInvalidHandoffTimeout             = errors.New(logs.InvalidHandoffTimeoutError())
	ErrInvalidHandoffWorkers             = errors.New(logs.InvalidHandoffWorkersError())
	ErrInvalidHandoffQueueSize           = errors.New(logs.InvalidHandoffQueueSizeError())
	ErrInvalidPostHandoffRelaxedDuration = errors.New(logs.InvalidPostHandoffRelaxedDurationError())
	ErrInvalidEndpointType               = errors.New(logs.InvalidEndpointTypeError())
	ErrInvalidMaintNotifications         = errors.New(logs.InvalidMaintNotificationsError())
	ErrMaxHandoffRetriesReached          = errors.New(logs.MaxHandoffRetriesReachedError())

	// Configuration validation errors

	// ErrInvalidHandoffRetries is returned when the number of handoff retries is invalid
	ErrInvalidHandoffRetries = errors.New(logs.InvalidHandoffRetriesError())
)

// Integration errors
var (
	// ErrInvalidClient is returned when the client does not support push notifications
	ErrInvalidClient = errors.New(logs.InvalidClientError())
)

// Handoff errors
var (
	// ErrHandoffQueueFull is returned when the handoff queue is full
	ErrHandoffQueueFull = errors.New(logs.HandoffQueueFullError())
)

// Notification errors
var (
	// ErrInvalidNotification is returned when a notification is in an invalid format
	ErrInvalidNotification = errors.New(logs.InvalidNotificationError())
)

// connection handoff errors
var (
	// ErrConnectionMarkedForHandoff is returned when a connection is marked for handoff
	// and should not be used until the handoff is complete
	ErrConnectionMarkedForHandoff = errors.New(logs.ConnectionMarkedForHandoffErrorMessage)
	// ErrConnectionMarkedForHandoffWithState is returned when a connection is marked for handoff
	// and should not be used until the handoff is complete
	ErrConnectionMarkedForHandoffWithState = errors.New(logs.ConnectionMarkedForHandoffErrorMessage + " with state")
	// ErrConnectionInvalidHandoffState is returned when a connection is in an invalid state for handoff
	ErrConnectionInvalidHandoffState = errors.New(logs.ConnectionInvalidHandoffStateErrorMessage)
)

// shutdown errors
var (
	// ErrShutdown is returned when the maintnotifications manager is shutdown
	ErrShutdown = errors.New(logs.ShutdownError())
)

// circuit breaker errors
var (
	// ErrCircuitBreakerOpen is returned when the circuit breaker is open
	ErrCircuitBreakerOpen = errors.New(logs.CircuitBreakerOpenErrorMessage)
)

// circuit breaker configuration errors
var (
	// ErrInvalidCircuitBreakerFailureThreshold is returned when the circuit breaker failure threshold is invalid
	ErrInvalidCircuitBreakerFailureThreshold = errors.New(logs.InvalidCircuitBreakerFailureThresholdError())
	// ErrInvalidCircuitBreakerResetTimeout is returned when the circuit breaker reset timeout is invalid
	ErrInvalidCircuitBreakerResetTimeout = errors.New(logs.InvalidCircuitBreakerResetTimeoutError())
	// ErrInvalidCircuitBreakerMaxRequests is returned when the circuit breaker max requests is invalid
	ErrInvalidCircuitBreakerMaxRequests = errors.New(logs.InvalidCircuitBreakerMaxRequestsError())
)
