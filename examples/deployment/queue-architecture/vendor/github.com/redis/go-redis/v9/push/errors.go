package push

import (
	"errors"
	"fmt"
)

// Push notification error definitions
// This file contains all error types and messages used by the push notification system

// Error reason constants
const (
	// HandlerReasons
	ReasonHandlerNil       = "handler cannot be nil"
	ReasonHandlerExists    = "cannot overwrite existing handler"
	ReasonHandlerProtected = "handler is protected"

	// ProcessorReasons
	ReasonPushNotificationsDisabled = "push notifications are disabled"
)

// ProcessorType represents the type of processor involved in the error
// defined as a custom type for better readability and easier maintenance
type ProcessorType string

const (
	// ProcessorTypes
	ProcessorTypeProcessor     = ProcessorType("processor")
	ProcessorTypeVoidProcessor = ProcessorType("void_processor")
	ProcessorTypeCustom        = ProcessorType("custom")
)

// ProcessorOperation represents the operation being performed by the processor
// defined as a custom type for better readability and easier maintenance
type ProcessorOperation string

const (
	// ProcessorOperations
	ProcessorOperationProcess    = ProcessorOperation("process")
	ProcessorOperationRegister   = ProcessorOperation("register")
	ProcessorOperationUnregister = ProcessorOperation("unregister")
	ProcessorOperationUnknown    = ProcessorOperation("unknown")
)

// Common error variables for reuse
var (
	// ErrHandlerNil is returned when attempting to register a nil handler
	ErrHandlerNil = errors.New(ReasonHandlerNil)
)

// Registry errors

// ErrHandlerExists creates an error for when attempting to overwrite an existing handler
func ErrHandlerExists(pushNotificationName string) error {
	return NewHandlerError(ProcessorOperationRegister, pushNotificationName, ReasonHandlerExists, nil)
}

// ErrProtectedHandler creates an error for when attempting to unregister a protected handler
func ErrProtectedHandler(pushNotificationName string) error {
	return NewHandlerError(ProcessorOperationUnregister, pushNotificationName, ReasonHandlerProtected, nil)
}

// VoidProcessor errors

// ErrVoidProcessorRegister creates an error for when attempting to register a handler on void processor
func ErrVoidProcessorRegister(pushNotificationName string) error {
	return NewProcessorError(ProcessorTypeVoidProcessor, ProcessorOperationRegister, pushNotificationName, ReasonPushNotificationsDisabled, nil)
}

// ErrVoidProcessorUnregister creates an error for when attempting to unregister a handler on void processor
func ErrVoidProcessorUnregister(pushNotificationName string) error {
	return NewProcessorError(ProcessorTypeVoidProcessor, ProcessorOperationUnregister, pushNotificationName, ReasonPushNotificationsDisabled, nil)
}

// Error type definitions for advanced error handling

// HandlerError represents errors related to handler operations
type HandlerError struct {
	Operation            ProcessorOperation
	PushNotificationName string
	Reason               string
	Err                  error
}

func (e *HandlerError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("handler %s failed for '%s': %s (%v)", e.Operation, e.PushNotificationName, e.Reason, e.Err)
	}
	return fmt.Sprintf("handler %s failed for '%s': %s", e.Operation, e.PushNotificationName, e.Reason)
}

func (e *HandlerError) Unwrap() error {
	return e.Err
}

// NewHandlerError creates a new HandlerError
func NewHandlerError(operation ProcessorOperation, pushNotificationName, reason string, err error) *HandlerError {
	return &HandlerError{
		Operation:            operation,
		PushNotificationName: pushNotificationName,
		Reason:               reason,
		Err:                  err,
	}
}

// ProcessorError represents errors related to processor operations
type ProcessorError struct {
	ProcessorType        ProcessorType      // "processor", "void_processor"
	Operation            ProcessorOperation // "process", "register", "unregister"
	PushNotificationName string             // Name of the push notification involved
	Reason               string
	Err                  error
}

func (e *ProcessorError) Error() string {
	notifInfo := ""
	if e.PushNotificationName != "" {
		notifInfo = fmt.Sprintf(" for '%s'", e.PushNotificationName)
	}
	if e.Err != nil {
		return fmt.Sprintf("%s %s failed%s: %s (%v)", e.ProcessorType, e.Operation, notifInfo, e.Reason, e.Err)
	}
	return fmt.Sprintf("%s %s failed%s: %s", e.ProcessorType, e.Operation, notifInfo, e.Reason)
}

func (e *ProcessorError) Unwrap() error {
	return e.Err
}

// NewProcessorError creates a new ProcessorError
func NewProcessorError(processorType ProcessorType, operation ProcessorOperation, pushNotificationName, reason string, err error) *ProcessorError {
	return &ProcessorError{
		ProcessorType:        processorType,
		Operation:            operation,
		PushNotificationName: pushNotificationName,
		Reason:               reason,
		Err:                  err,
	}
}

// Helper functions for common error scenarios

// IsHandlerNilError checks if an error is due to a nil handler
func IsHandlerNilError(err error) bool {
	return errors.Is(err, ErrHandlerNil)
}

// IsHandlerExistsError checks if an error is due to attempting to overwrite an existing handler.
// This function works correctly even when the error is wrapped.
func IsHandlerExistsError(err error) bool {
	var handlerErr *HandlerError
	if errors.As(err, &handlerErr) {
		return handlerErr.Operation == ProcessorOperationRegister && handlerErr.Reason == ReasonHandlerExists
	}
	return false
}

// IsProtectedHandlerError checks if an error is due to attempting to unregister a protected handler.
// This function works correctly even when the error is wrapped.
func IsProtectedHandlerError(err error) bool {
	var handlerErr *HandlerError
	if errors.As(err, &handlerErr) {
		return handlerErr.Operation == ProcessorOperationUnregister && handlerErr.Reason == ReasonHandlerProtected
	}
	return false
}

// IsVoidProcessorError checks if an error is due to void processor operations.
// This function works correctly even when the error is wrapped.
func IsVoidProcessorError(err error) bool {
	var procErr *ProcessorError
	if errors.As(err, &procErr) {
		return procErr.ProcessorType == ProcessorTypeVoidProcessor && procErr.Reason == ReasonPushNotificationsDisabled
	}
	return false
}
