package push

import (
	"sync"
)

// Registry manages push notification handlers
type Registry struct {
	mu        sync.RWMutex
	handlers  map[string]NotificationHandler
	protected map[string]bool
}

// NewRegistry creates a new push notification registry
func NewRegistry() *Registry {
	return &Registry{
		handlers:  make(map[string]NotificationHandler),
		protected: make(map[string]bool),
	}
}

// RegisterHandler registers a handler for a specific push notification name
func (r *Registry) RegisterHandler(pushNotificationName string, handler NotificationHandler, protected bool) error {
	if handler == nil {
		return ErrHandlerNil
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if handler already exists
	if _, exists := r.protected[pushNotificationName]; exists {
		return ErrHandlerExists(pushNotificationName)
	}

	r.handlers[pushNotificationName] = handler
	r.protected[pushNotificationName] = protected
	return nil
}

// GetHandler returns the handler for a specific push notification name
func (r *Registry) GetHandler(pushNotificationName string) NotificationHandler {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.handlers[pushNotificationName]
}

// UnregisterHandler removes a handler for a specific push notification name
func (r *Registry) UnregisterHandler(pushNotificationName string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if handler is protected
	if protected, exists := r.protected[pushNotificationName]; exists && protected {
		return ErrProtectedHandler(pushNotificationName)
	}

	delete(r.handlers, pushNotificationName)
	delete(r.protected, pushNotificationName)
	return nil
}
