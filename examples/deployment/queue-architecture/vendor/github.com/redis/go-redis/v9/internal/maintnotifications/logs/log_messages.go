package logs

import (
	"encoding/json"
	"fmt"
	"regexp"

	"github.com/redis/go-redis/v9/internal"
)

// appendJSONIfDebug appends JSON data to a message only if the global log level is Debug
func appendJSONIfDebug(message string, data map[string]interface{}) string {
	if internal.LogLevel.DebugOrAbove() {
		jsonData, _ := json.Marshal(data)
		return fmt.Sprintf("%s %s", message, string(jsonData))
	}
	return message
}

const (
	// ========================================
	// CIRCUIT_BREAKER.GO - Circuit breaker management
	// ========================================
	CircuitBreakerTransitioningToHalfOpenMessage = "circuit breaker transitioning to half-open"
	CircuitBreakerOpenedMessage                  = "circuit breaker opened"
	CircuitBreakerReopenedMessage                = "circuit breaker reopened"
	CircuitBreakerClosedMessage                  = "circuit breaker closed"
	CircuitBreakerCleanupMessage                 = "circuit breaker cleanup"
	CircuitBreakerOpenMessage                    = "circuit breaker is open, failing fast"

	// ========================================
	// CONFIG.GO - Configuration and debug
	// ========================================
	DebugLoggingEnabledMessage = "debug logging enabled"
	ConfigDebugMessage         = "config debug"

	// ========================================
	// ERRORS.GO - Error message constants
	// ========================================
	InvalidRelaxedTimeoutErrorMessage                 = "relaxed timeout must be greater than 0"
	InvalidHandoffTimeoutErrorMessage                 = "handoff timeout must be greater than 0"
	InvalidHandoffWorkersErrorMessage                 = "MaxWorkers must be greater than or equal to 0"
	InvalidHandoffQueueSizeErrorMessage               = "handoff queue size must be greater than 0"
	InvalidPostHandoffRelaxedDurationErrorMessage     = "post-handoff relaxed duration must be greater than or equal to 0"
	InvalidEndpointTypeErrorMessage                   = "invalid endpoint type"
	InvalidMaintNotificationsErrorMessage             = "invalid maintenance notifications setting (must be 'disabled', 'enabled', or 'auto')"
	InvalidHandoffRetriesErrorMessage                 = "MaxHandoffRetries must be between 1 and 10"
	InvalidClientErrorMessage                         = "invalid client type"
	InvalidNotificationErrorMessage                   = "invalid notification format"
	MaxHandoffRetriesReachedErrorMessage              = "max handoff retries reached"
	HandoffQueueFullErrorMessage                      = "handoff queue is full, cannot queue new handoff requests - consider increasing HandoffQueueSize or MaxWorkers in configuration"
	InvalidCircuitBreakerFailureThresholdErrorMessage = "circuit breaker failure threshold must be >= 1"
	InvalidCircuitBreakerResetTimeoutErrorMessage     = "circuit breaker reset timeout must be >= 0"
	InvalidCircuitBreakerMaxRequestsErrorMessage      = "circuit breaker max requests must be >= 1"
	ConnectionMarkedForHandoffErrorMessage            = "connection marked for handoff"
	ConnectionInvalidHandoffStateErrorMessage         = "connection is in invalid state for handoff"
	ShutdownErrorMessage                              = "shutdown"
	CircuitBreakerOpenErrorMessage                    = "circuit breaker is open, failing fast"

	// ========================================
	// EXAMPLE_HOOKS.GO - Example metrics hooks
	// ========================================
	MetricsHookProcessingNotificationMessage = "metrics hook processing"
	MetricsHookRecordedErrorMessage          = "metrics hook recorded error"

	// ========================================
	// HANDOFF_WORKER.GO - Connection handoff processing
	// ========================================
	HandoffStartedMessage                            = "handoff started"
	HandoffFailedMessage                             = "handoff failed"
	ConnectionNotMarkedForHandoffMessage             = "is not marked for handoff and has no retries"
	ConnectionNotMarkedForHandoffErrorMessage        = "is not marked for handoff"
	HandoffRetryAttemptMessage                       = "Performing handoff"
	CannotQueueHandoffForRetryMessage                = "can't queue handoff for retry"
	HandoffQueueFullMessage                          = "handoff queue is full"
	FailedToDialNewEndpointMessage                   = "failed to dial new endpoint"
	ApplyingRelaxedTimeoutDueToPostHandoffMessage    = "applying relaxed timeout due to post-handoff"
	HandoffSuccessMessage                            = "handoff succeeded"
	RemovingConnectionFromPoolMessage                = "removing connection from pool"
	NoPoolProvidedMessageCannotRemoveMessage         = "no pool provided, cannot remove connection, closing it"
	WorkerExitingDueToShutdownMessage                = "worker exiting due to shutdown"
	WorkerExitingDueToShutdownWhileProcessingMessage = "worker exiting due to shutdown while processing request"
	WorkerPanicRecoveredMessage                      = "worker panic recovered"
	WorkerExitingDueToInactivityTimeoutMessage       = "worker exiting due to inactivity timeout"
	ReachedMaxHandoffRetriesMessage                  = "reached max handoff retries"

	// ========================================
	// MANAGER.GO - Moving operation tracking and handler registration
	// ========================================
	DuplicateMovingOperationMessage  = "duplicate MOVING operation ignored"
	TrackingMovingOperationMessage   = "tracking MOVING operation"
	UntrackingMovingOperationMessage = "untracking MOVING operation"
	OperationNotTrackedMessage       = "operation not tracked"
	FailedToRegisterHandlerMessage   = "failed to register handler"

	// ========================================
	// HOOKS.GO - Notification processing hooks
	// ========================================
	ProcessingNotificationMessage          = "processing notification started"
	ProcessingNotificationFailedMessage    = "proccessing notification failed"
	ProcessingNotificationSucceededMessage = "processing notification succeeded"

	// ========================================
	// POOL_HOOK.GO - Pool connection management
	// ========================================
	FailedToQueueHandoffMessage = "failed to queue handoff"
	MarkedForHandoffMessage     = "connection marked for handoff"

	// ========================================
	// PUSH_NOTIFICATION_HANDLER.GO - Push notification validation and processing
	// ========================================
	InvalidNotificationFormatMessage              = "invalid notification format"
	InvalidNotificationTypeFormatMessage          = "invalid notification type format"
	InvalidSeqIDInMovingNotificationMessage       = "invalid seqID in MOVING notification"
	InvalidTimeSInMovingNotificationMessage       = "invalid timeS in MOVING notification"
	InvalidNewEndpointInMovingNotificationMessage = "invalid newEndpoint in MOVING notification"
	NoConnectionInHandlerContextMessage           = "no connection in handler context"
	InvalidConnectionTypeInHandlerContextMessage  = "invalid connection type in handler context"
	SchedulingHandoffToCurrentEndpointMessage     = "scheduling handoff to current endpoint"
	RelaxedTimeoutDueToNotificationMessage        = "applying relaxed timeout due to notification"
	UnrelaxedTimeoutMessage                       = "clearing relaxed timeout"
	ManagerNotInitializedMessage                  = "manager not initialized"
	FailedToMarkForHandoffMessage                 = "failed to mark connection for handoff"
	InvalidSeqIDInSMigratingNotificationMessage   = "invalid SeqID in SMIGRATING notification"
	InvalidSeqIDInSMigratedNotificationMessage    = "invalid SeqID in SMIGRATED notification"
	TriggeringClusterStateReloadMessage           = "triggering cluster state reload"

	// ========================================
	// used in pool/conn
	// ========================================
	UnrelaxedTimeoutAfterDeadlineMessage = "clearing relaxed timeout after deadline"
)

func HandoffStarted(connID uint64, newEndpoint string) string {
	message := fmt.Sprintf("conn[%d] %s to %s", connID, HandoffStartedMessage, newEndpoint)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":   connID,
		"endpoint": newEndpoint,
	})
}

func HandoffFailed(connID uint64, newEndpoint string, attempt int, maxAttempts int, err error) string {
	message := fmt.Sprintf("conn[%d] %s to %s (attempt %d/%d): %v", connID, HandoffFailedMessage, newEndpoint, attempt, maxAttempts, err)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":      connID,
		"endpoint":    newEndpoint,
		"attempt":     attempt,
		"maxAttempts": maxAttempts,
		"error":       err.Error(),
	})
}

func HandoffSucceeded(connID uint64, newEndpoint string) string {
	message := fmt.Sprintf("conn[%d] %s to %s", connID, HandoffSuccessMessage, newEndpoint)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":   connID,
		"endpoint": newEndpoint,
	})
}

// Timeout-related log functions
func RelaxedTimeoutDueToNotification(connID uint64, notificationType string, timeout interface{}) string {
	message := fmt.Sprintf("conn[%d] %s %s (%v)", connID, RelaxedTimeoutDueToNotificationMessage, notificationType, timeout)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":           connID,
		"notificationType": notificationType,
		"timeout":          fmt.Sprintf("%v", timeout),
	})
}

func UnrelaxedTimeout(connID uint64) string {
	message := fmt.Sprintf("conn[%d] %s", connID, UnrelaxedTimeoutMessage)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID": connID,
	})
}

func UnrelaxedTimeoutAfterDeadline(connID uint64) string {
	message := fmt.Sprintf("conn[%d] %s", connID, UnrelaxedTimeoutAfterDeadlineMessage)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID": connID,
	})
}

// Handoff queue and marking functions
func HandoffQueueFull(queueLen, queueCap int) string {
	message := fmt.Sprintf("%s (%d/%d), cannot queue new handoff requests - consider increasing HandoffQueueSize or MaxWorkers in configuration", HandoffQueueFullMessage, queueLen, queueCap)
	return appendJSONIfDebug(message, map[string]interface{}{
		"queueLen": queueLen,
		"queueCap": queueCap,
	})
}

func FailedToQueueHandoff(connID uint64, err error) string {
	message := fmt.Sprintf("conn[%d] %s: %v", connID, FailedToQueueHandoffMessage, err)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID": connID,
		"error":  err.Error(),
	})
}

func FailedToMarkForHandoff(connID uint64, err error) string {
	message := fmt.Sprintf("conn[%d] %s: %v", connID, FailedToMarkForHandoffMessage, err)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID": connID,
		"error":  err.Error(),
	})
}

func FailedToDialNewEndpoint(connID uint64, endpoint string, err error) string {
	message := fmt.Sprintf("conn[%d] %s %s: %v", connID, FailedToDialNewEndpointMessage, endpoint, err)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":   connID,
		"endpoint": endpoint,
		"error":    err.Error(),
	})
}

func ReachedMaxHandoffRetries(connID uint64, endpoint string, maxRetries int) string {
	message := fmt.Sprintf("conn[%d] %s to %s (max retries: %d)", connID, ReachedMaxHandoffRetriesMessage, endpoint, maxRetries)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":     connID,
		"endpoint":   endpoint,
		"maxRetries": maxRetries,
	})
}

// Notification processing functions
func ProcessingNotification(connID uint64, seqID int64, notificationType string, notification interface{}) string {
	message := fmt.Sprintf("conn[%d] seqID[%d] %s %s: %v", connID, seqID, ProcessingNotificationMessage, notificationType, notification)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":           connID,
		"seqID":            seqID,
		"notificationType": notificationType,
		"notification":     fmt.Sprintf("%v", notification),
	})
}

func ProcessingNotificationFailed(connID uint64, notificationType string, err error, notification interface{}) string {
	message := fmt.Sprintf("conn[%d] %s %s: %v - %v", connID, ProcessingNotificationFailedMessage, notificationType, err, notification)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":           connID,
		"notificationType": notificationType,
		"error":            err.Error(),
		"notification":     fmt.Sprintf("%v", notification),
	})
}

func ProcessingNotificationSucceeded(connID uint64, notificationType string) string {
	message := fmt.Sprintf("conn[%d] %s %s", connID, ProcessingNotificationSucceededMessage, notificationType)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":           connID,
		"notificationType": notificationType,
	})
}

// Moving operation tracking functions
func DuplicateMovingOperation(connID uint64, endpoint string, seqID int64) string {
	message := fmt.Sprintf("conn[%d] %s for %s seqID[%d]", connID, DuplicateMovingOperationMessage, endpoint, seqID)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":   connID,
		"endpoint": endpoint,
		"seqID":    seqID,
	})
}

func TrackingMovingOperation(connID uint64, endpoint string, seqID int64) string {
	message := fmt.Sprintf("conn[%d] %s for %s seqID[%d]", connID, TrackingMovingOperationMessage, endpoint, seqID)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":   connID,
		"endpoint": endpoint,
		"seqID":    seqID,
	})
}

func UntrackingMovingOperation(connID uint64, seqID int64) string {
	message := fmt.Sprintf("conn[%d] %s seqID[%d]", connID, UntrackingMovingOperationMessage, seqID)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID": connID,
		"seqID":  seqID,
	})
}

func OperationNotTracked(connID uint64, seqID int64) string {
	message := fmt.Sprintf("conn[%d] %s seqID[%d]", connID, OperationNotTrackedMessage, seqID)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID": connID,
		"seqID":  seqID,
	})
}

// Connection pool functions
func RemovingConnectionFromPool(connID uint64, reason error) string {
	metadata := map[string]interface{}{
		"connID": connID,
		"reason": "unknown", // this will be overwritten if reason is not nil
	}
	if reason != nil {
		metadata["reason"] = reason.Error()
	}

	message := fmt.Sprintf("conn[%d] %s due to: %v", connID, RemovingConnectionFromPoolMessage, reason)
	return appendJSONIfDebug(message, metadata)
}

func NoPoolProvidedCannotRemove(connID uint64, reason error) string {
	metadata := map[string]interface{}{
		"connID": connID,
		"reason": "unknown", // this will be overwritten if reason is not nil
	}
	if reason != nil {
		metadata["reason"] = reason.Error()
	}

	message := fmt.Sprintf("conn[%d] %s due to: %v", connID, NoPoolProvidedMessageCannotRemoveMessage, reason)
	return appendJSONIfDebug(message, metadata)
}

// Circuit breaker functions
func CircuitBreakerOpen(connID uint64, endpoint string) string {
	message := fmt.Sprintf("conn[%d] %s for %s", connID, CircuitBreakerOpenMessage, endpoint)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":   connID,
		"endpoint": endpoint,
	})
}

// Additional handoff functions for specific cases
func ConnectionNotMarkedForHandoff(connID uint64) string {
	message := fmt.Sprintf("conn[%d] %s", connID, ConnectionNotMarkedForHandoffMessage)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID": connID,
	})
}

func ConnectionNotMarkedForHandoffError(connID uint64) string {
	return fmt.Sprintf("conn[%d] %s", connID, ConnectionNotMarkedForHandoffErrorMessage)
}

func HandoffRetryAttempt(connID uint64, retries int, newEndpoint string, oldEndpoint string) string {
	message := fmt.Sprintf("conn[%d] Retry %d: %s to %s(was %s)", connID, retries, HandoffRetryAttemptMessage, newEndpoint, oldEndpoint)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":      connID,
		"retries":     retries,
		"newEndpoint": newEndpoint,
		"oldEndpoint": oldEndpoint,
	})
}

func CannotQueueHandoffForRetry(err error) string {
	message := fmt.Sprintf("%s: %v", CannotQueueHandoffForRetryMessage, err)
	return appendJSONIfDebug(message, map[string]interface{}{
		"error": err.Error(),
	})
}

// Validation and error functions
func InvalidNotificationFormat(notification interface{}) string {
	message := fmt.Sprintf("%s: %v", InvalidNotificationFormatMessage, notification)
	return appendJSONIfDebug(message, map[string]interface{}{
		"notification": fmt.Sprintf("%v", notification),
	})
}

func InvalidNotificationTypeFormat(notificationType interface{}) string {
	message := fmt.Sprintf("%s: %v", InvalidNotificationTypeFormatMessage, notificationType)
	return appendJSONIfDebug(message, map[string]interface{}{
		"notificationType": fmt.Sprintf("%v", notificationType),
	})
}

// InvalidNotification creates a log message for invalid notifications of any type
func InvalidNotification(notificationType string, notification interface{}) string {
	message := fmt.Sprintf("invalid %s notification: %v", notificationType, notification)
	return appendJSONIfDebug(message, map[string]interface{}{
		"notificationType": notificationType,
		"notification":     fmt.Sprintf("%v", notification),
	})
}

func InvalidSeqIDInMovingNotification(seqID interface{}) string {
	message := fmt.Sprintf("%s: %v", InvalidSeqIDInMovingNotificationMessage, seqID)
	return appendJSONIfDebug(message, map[string]interface{}{
		"seqID": fmt.Sprintf("%v", seqID),
	})
}

func InvalidTimeSInMovingNotification(timeS interface{}) string {
	message := fmt.Sprintf("%s: %v", InvalidTimeSInMovingNotificationMessage, timeS)
	return appendJSONIfDebug(message, map[string]interface{}{
		"timeS": fmt.Sprintf("%v", timeS),
	})
}

func InvalidNewEndpointInMovingNotification(newEndpoint interface{}) string {
	message := fmt.Sprintf("%s: %v", InvalidNewEndpointInMovingNotificationMessage, newEndpoint)
	return appendJSONIfDebug(message, map[string]interface{}{
		"newEndpoint": fmt.Sprintf("%v", newEndpoint),
	})
}

func NoConnectionInHandlerContext(notificationType string) string {
	message := fmt.Sprintf("%s for %s notification", NoConnectionInHandlerContextMessage, notificationType)
	return appendJSONIfDebug(message, map[string]interface{}{
		"notificationType": notificationType,
	})
}

func InvalidConnectionTypeInHandlerContext(notificationType string, conn interface{}, handlerCtx interface{}) string {
	message := fmt.Sprintf("%s for %s notification - %T %#v", InvalidConnectionTypeInHandlerContextMessage, notificationType, conn, handlerCtx)
	return appendJSONIfDebug(message, map[string]interface{}{
		"notificationType": notificationType,
		"connType":         fmt.Sprintf("%T", conn),
	})
}

func SchedulingHandoffToCurrentEndpoint(connID uint64, seconds float64) string {
	message := fmt.Sprintf("conn[%d] %s in %v seconds", connID, SchedulingHandoffToCurrentEndpointMessage, seconds)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":  connID,
		"seconds": seconds,
	})
}

func ManagerNotInitialized() string {
	return appendJSONIfDebug(ManagerNotInitializedMessage, map[string]interface{}{})
}

func FailedToRegisterHandler(notificationType string, err error) string {
	message := fmt.Sprintf("%s for %s: %v", FailedToRegisterHandlerMessage, notificationType, err)
	return appendJSONIfDebug(message, map[string]interface{}{
		"notificationType": notificationType,
		"error":            err.Error(),
	})
}

func ShutdownError() string {
	return appendJSONIfDebug(ShutdownErrorMessage, map[string]interface{}{})
}

// Configuration validation error functions
func InvalidRelaxedTimeoutError() string {
	return appendJSONIfDebug(InvalidRelaxedTimeoutErrorMessage, map[string]interface{}{})
}

func InvalidHandoffTimeoutError() string {
	return appendJSONIfDebug(InvalidHandoffTimeoutErrorMessage, map[string]interface{}{})
}

func InvalidHandoffWorkersError() string {
	return appendJSONIfDebug(InvalidHandoffWorkersErrorMessage, map[string]interface{}{})
}

func InvalidHandoffQueueSizeError() string {
	return appendJSONIfDebug(InvalidHandoffQueueSizeErrorMessage, map[string]interface{}{})
}

func InvalidPostHandoffRelaxedDurationError() string {
	return appendJSONIfDebug(InvalidPostHandoffRelaxedDurationErrorMessage, map[string]interface{}{})
}

func InvalidEndpointTypeError() string {
	return appendJSONIfDebug(InvalidEndpointTypeErrorMessage, map[string]interface{}{})
}

func InvalidMaintNotificationsError() string {
	return appendJSONIfDebug(InvalidMaintNotificationsErrorMessage, map[string]interface{}{})
}

func InvalidHandoffRetriesError() string {
	return appendJSONIfDebug(InvalidHandoffRetriesErrorMessage, map[string]interface{}{})
}

func InvalidClientError() string {
	return appendJSONIfDebug(InvalidClientErrorMessage, map[string]interface{}{})
}

func InvalidNotificationError() string {
	return appendJSONIfDebug(InvalidNotificationErrorMessage, map[string]interface{}{})
}

func MaxHandoffRetriesReachedError() string {
	return appendJSONIfDebug(MaxHandoffRetriesReachedErrorMessage, map[string]interface{}{})
}

func HandoffQueueFullError() string {
	return appendJSONIfDebug(HandoffQueueFullErrorMessage, map[string]interface{}{})
}

func InvalidCircuitBreakerFailureThresholdError() string {
	return appendJSONIfDebug(InvalidCircuitBreakerFailureThresholdErrorMessage, map[string]interface{}{})
}

func InvalidCircuitBreakerResetTimeoutError() string {
	return appendJSONIfDebug(InvalidCircuitBreakerResetTimeoutErrorMessage, map[string]interface{}{})
}

func InvalidCircuitBreakerMaxRequestsError() string {
	return appendJSONIfDebug(InvalidCircuitBreakerMaxRequestsErrorMessage, map[string]interface{}{})
}

// Configuration and debug functions
func DebugLoggingEnabled() string {
	return appendJSONIfDebug(DebugLoggingEnabledMessage, map[string]interface{}{})
}

func ConfigDebug(config interface{}) string {
	message := fmt.Sprintf("%s: %+v", ConfigDebugMessage, config)
	return appendJSONIfDebug(message, map[string]interface{}{
		"config": fmt.Sprintf("%+v", config),
	})
}

// Handoff worker functions
func WorkerExitingDueToShutdown() string {
	return appendJSONIfDebug(WorkerExitingDueToShutdownMessage, map[string]interface{}{})
}

func WorkerExitingDueToShutdownWhileProcessing() string {
	return appendJSONIfDebug(WorkerExitingDueToShutdownWhileProcessingMessage, map[string]interface{}{})
}

func WorkerPanicRecovered(panicValue interface{}) string {
	message := fmt.Sprintf("%s: %v", WorkerPanicRecoveredMessage, panicValue)
	return appendJSONIfDebug(message, map[string]interface{}{
		"panic": fmt.Sprintf("%v", panicValue),
	})
}

func WorkerExitingDueToInactivityTimeout(timeout interface{}) string {
	message := fmt.Sprintf("%s (%v)", WorkerExitingDueToInactivityTimeoutMessage, timeout)
	return appendJSONIfDebug(message, map[string]interface{}{
		"timeout": fmt.Sprintf("%v", timeout),
	})
}

func ApplyingRelaxedTimeoutDueToPostHandoff(connID uint64, timeout interface{}, until string) string {
	message := fmt.Sprintf("conn[%d] %s (%v) until %s", connID, ApplyingRelaxedTimeoutDueToPostHandoffMessage, timeout, until)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID":  connID,
		"timeout": fmt.Sprintf("%v", timeout),
		"until":   until,
	})
}

// Example hooks functions
func MetricsHookProcessingNotification(notificationType string, connID uint64) string {
	message := fmt.Sprintf("%s %s notification on conn[%d]", MetricsHookProcessingNotificationMessage, notificationType, connID)
	return appendJSONIfDebug(message, map[string]interface{}{
		"notificationType": notificationType,
		"connID":           connID,
	})
}

func MetricsHookRecordedError(notificationType string, connID uint64, err error) string {
	message := fmt.Sprintf("%s for %s notification on conn[%d]: %v", MetricsHookRecordedErrorMessage, notificationType, connID, err)
	return appendJSONIfDebug(message, map[string]interface{}{
		"notificationType": notificationType,
		"connID":           connID,
		"error":            err.Error(),
	})
}

// Pool hook functions
func MarkedForHandoff(connID uint64) string {
	message := fmt.Sprintf("conn[%d] %s", connID, MarkedForHandoffMessage)
	return appendJSONIfDebug(message, map[string]interface{}{
		"connID": connID,
	})
}

// Circuit breaker additional functions
func CircuitBreakerTransitioningToHalfOpen(endpoint string) string {
	message := fmt.Sprintf("%s for %s", CircuitBreakerTransitioningToHalfOpenMessage, endpoint)
	return appendJSONIfDebug(message, map[string]interface{}{
		"endpoint": endpoint,
	})
}

func CircuitBreakerOpened(endpoint string, failures int64) string {
	message := fmt.Sprintf("%s for endpoint %s after %d failures", CircuitBreakerOpenedMessage, endpoint, failures)
	return appendJSONIfDebug(message, map[string]interface{}{
		"endpoint": endpoint,
		"failures": failures,
	})
}

func CircuitBreakerReopened(endpoint string) string {
	message := fmt.Sprintf("%s for endpoint %s due to failure in half-open state", CircuitBreakerReopenedMessage, endpoint)
	return appendJSONIfDebug(message, map[string]interface{}{
		"endpoint": endpoint,
	})
}

func CircuitBreakerClosed(endpoint string, successes int64) string {
	message := fmt.Sprintf("%s for endpoint %s after %d successful requests", CircuitBreakerClosedMessage, endpoint, successes)
	return appendJSONIfDebug(message, map[string]interface{}{
		"endpoint":  endpoint,
		"successes": successes,
	})
}

func CircuitBreakerCleanup(removed int, total int) string {
	message := fmt.Sprintf("%s removed %d/%d entries", CircuitBreakerCleanupMessage, removed, total)
	return appendJSONIfDebug(message, map[string]interface{}{
		"removed": removed,
		"total":   total,
	})
}

// ExtractDataFromLogMessage extracts structured data from maintnotifications log messages
// Returns a map containing the parsed key-value pairs from the structured data section
// Example: "conn[123] handoff started to localhost:6379 {"connID":123,"endpoint":"localhost:6379"}"
// Returns: map[string]interface{}{"connID": 123, "endpoint": "localhost:6379"}
func ExtractDataFromLogMessage(logMessage string) map[string]interface{} {
	result := make(map[string]interface{})

	// Find the JSON data section at the end of the message
	re := regexp.MustCompile(`(\{.*\})$`)
	matches := re.FindStringSubmatch(logMessage)
	if len(matches) < 2 {
		return result
	}

	jsonStr := matches[1]
	if jsonStr == "" {
		return result
	}

	// Parse the JSON directly
	var jsonResult map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &jsonResult); err == nil {
		return jsonResult
	}

	// If JSON parsing fails, return empty map
	return result
}

// Cluster notification functions
func InvalidSeqIDInSMigratingNotification(seqID interface{}) string {
	message := fmt.Sprintf("%s: %v", InvalidSeqIDInSMigratingNotificationMessage, seqID)
	return appendJSONIfDebug(message, map[string]interface{}{
		"seqID": fmt.Sprintf("%v", seqID),
	})
}

func InvalidSeqIDInSMigratedNotification(seqID interface{}) string {
	message := fmt.Sprintf("%s: %v", InvalidSeqIDInSMigratedNotificationMessage, seqID)
	return appendJSONIfDebug(message, map[string]interface{}{
		"seqID": fmt.Sprintf("%v", seqID),
	})
}

// TriggeringClusterStateReload logs when cluster state reload is triggered (deduplicated, once per seqID)
func TriggeringClusterStateReload(seqID int64, hostPort string, slotRanges []string) string {
	message := fmt.Sprintf("%s seqID=%d host:port=%s slots=%v", TriggeringClusterStateReloadMessage, seqID, hostPort, slotRanges)
	return appendJSONIfDebug(message, map[string]interface{}{
		"seqID":      seqID,
		"hostPort":   hostPort,
		"slotRanges": slotRanges,
	})
}
