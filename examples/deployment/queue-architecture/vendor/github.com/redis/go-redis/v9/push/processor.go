package push

import (
	"context"
	"errors"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/proto"
)

// NotificationProcessor defines the interface for push notification processors.
type NotificationProcessor interface {
	// GetHandler returns the handler for a specific push notification name.
	GetHandler(pushNotificationName string) NotificationHandler
	// ProcessPendingNotifications checks for and processes any pending push notifications.
	// To be used when it is known that there are notifications on the socket.
	// It will try to read from the socket and if it is empty - it may block.
	ProcessPendingNotifications(ctx context.Context, handlerCtx NotificationHandlerContext, rd *proto.Reader) error
	// RegisterHandler registers a handler for a specific push notification name.
	RegisterHandler(pushNotificationName string, handler NotificationHandler, protected bool) error
	// UnregisterHandler removes a handler for a specific push notification name.
	UnregisterHandler(pushNotificationName string) error
}

// Processor handles push notifications with a registry of handlers
type Processor struct {
	registry *Registry
}

type timeoutError interface {
	Timeout() bool
}

func isTimeoutError(err error) bool {
	var timeoutErr timeoutError
	return errors.As(err, &timeoutErr) && timeoutErr.Timeout()
}

// NewProcessor creates a new push notification processor
func NewProcessor() *Processor {
	return &Processor{
		registry: NewRegistry(),
	}
}

// GetHandler returns the handler for a specific push notification name
func (p *Processor) GetHandler(pushNotificationName string) NotificationHandler {
	return p.registry.GetHandler(pushNotificationName)
}

// RegisterHandler registers a handler for a specific push notification name
func (p *Processor) RegisterHandler(pushNotificationName string, handler NotificationHandler, protected bool) error {
	return p.registry.RegisterHandler(pushNotificationName, handler, protected)
}

// UnregisterHandler removes a handler for a specific push notification name
func (p *Processor) UnregisterHandler(pushNotificationName string) error {
	return p.registry.UnregisterHandler(pushNotificationName)
}

// ProcessPendingNotifications checks for and processes any pending push notifications
// This method should be called by the client in WithReader before reading the reply
// It will try to read from the socket and if it is empty - it may block.
func (p *Processor) ProcessPendingNotifications(ctx context.Context, handlerCtx NotificationHandlerContext, rd *proto.Reader) error {
	return p.processPendingNotifications(ctx, handlerCtx, rd, false)
}

// ProcessPendingNotificationsBuffered processes one pending push notification
// and then continues only through frames already buffered by that read. It is
// used by callers that have already established socket readiness and must not
// wait for another frame after draining the current batch.
func (p *Processor) ProcessPendingNotificationsBuffered(
	ctx context.Context, handlerCtx NotificationHandlerContext, rd *proto.Reader,
) error {
	return p.processPendingNotifications(ctx, handlerCtx, rd, true)
}

func (p *Processor) processPendingNotifications(
	ctx context.Context,
	handlerCtx NotificationHandlerContext,
	rd *proto.Reader,
	bufferedContinuation bool,
) error {
	if rd == nil {
		return nil
	}

	processed := false
	for !bufferedContinuation || !processed || rd.Buffered() > 0 {
		// Check if there's data available to read
		var replyType byte
		if bufferedContinuation {
			for {
				b, err := rd.Peek(1)
				if err != nil {
					if isTimeoutError(err) {
						return nil
					}
					return err
				}
				replyType = b[0]
				if replyType != proto.RespAttr {
					break
				}
				// Unlike Peek, DiscardNext consumes bytes. Any error here is
				// fatal because the reader may be left mid-frame.
				if err := rd.DiscardNext(); err != nil {
					return err
				}
			}
		} else {
			var err error
			replyType, err = rd.PeekReplyType()
			if err != nil {
				// No more data available or error reading.
				break
			}
		}

		// Only process push notifications (arrays starting with >)
		if replyType != proto.RespPush {
			break
		}

		// see if we should skip this notification
		notificationName, err := rd.PeekPushNotificationName()
		if err != nil {
			// Name too long to peek: consume & dispatch below rather than leave the
			// frame at the buffer head (which would stall and desync the next reply).
			if !errors.Is(err, proto.ErrPushNotificationNameTooLong) {
				if bufferedContinuation {
					if isTimeoutError(err) {
						return nil
					}
					return err
				}
				break
			}
		} else if willHandleNotificationInClient(notificationName) {
			break
		}

		// Surface a ReadReply error (unlike the boundary peek errors above,
		// which consumed nothing): it happens mid-frame after bytes are
		// consumed, so the conn is desynced and the CSC drainer must remove it.
		// Normal reply-read callers log-and-ignore this and let their own read fail.
		reply, err := rd.ReadReply()
		if err != nil {
			internal.Logger.Printf(ctx, "push: error reading push notification: %v", err)
			return err
		}
		processed = true

		// Convert to slice of interfaces
		notification, ok := reply.([]interface{})
		if !ok {
			break
		}

		// Handle the notification directly
		if len(notification) > 0 {
			// Extract the notification type (first element)
			if notificationType, ok := notification[0].(string); ok {
				// Get the handler for this notification type
				if handler := p.registry.GetHandler(notificationType); handler != nil {
					// Handle the notification
					err := handler.HandlePushNotification(ctx, handlerCtx, notification)
					if err != nil {
						internal.Logger.Printf(ctx, "push: error handling push notification: %v", err)
					}
				}
			}
		}
	}

	return nil
}

// VoidProcessor discards all push notifications without processing them
type VoidProcessor struct{}

// NewVoidProcessor creates a new void push notification processor
func NewVoidProcessor() *VoidProcessor {
	return &VoidProcessor{}
}

// GetHandler returns nil for void processor since it doesn't maintain handlers
func (v *VoidProcessor) GetHandler(_ string) NotificationHandler {
	return nil
}

// RegisterHandler returns an error for void processor since it doesn't maintain handlers
func (v *VoidProcessor) RegisterHandler(pushNotificationName string, _ NotificationHandler, _ bool) error {
	return ErrVoidProcessorRegister(pushNotificationName)
}

// UnregisterHandler returns an error for void processor since it doesn't maintain handlers
func (v *VoidProcessor) UnregisterHandler(pushNotificationName string) error {
	return ErrVoidProcessorUnregister(pushNotificationName)
}

// ProcessPendingNotifications for VoidProcessor does nothing since push notifications
// are only available in RESP3 and this processor is used for RESP2 connections.
// This avoids unnecessary buffer scanning overhead.
// It does however read and discard all push notifications from the buffer to avoid
// them being interpreted as a reply.
// This method should be called by the client in WithReader before reading the reply
// to be sure there are no buffered push notifications.
// It will try to read from the socket and if it is empty - it may block.
func (v *VoidProcessor) ProcessPendingNotifications(_ context.Context, handlerCtx NotificationHandlerContext, rd *proto.Reader) error {
	// read and discard all push notifications
	if rd == nil {
		return nil
	}

	for {
		// Check if there's data available to read
		replyType, err := rd.PeekReplyType()
		if err != nil {
			// No more data available or error reading
			// if timeout, it will be handled by the caller
			break
		}

		// Only process push notifications (arrays starting with >)
		if replyType != proto.RespPush {
			break
		}
		// see if we should skip this notification
		notificationName, err := rd.PeekPushNotificationName()
		if err != nil {
			// Name too long to peek: still consume the frame below so it isn't
			// misread as a reply.
			if !errors.Is(err, proto.ErrPushNotificationNameTooLong) {
				break
			}
		} else if willHandleNotificationInClient(notificationName) {
			break
		}

		// Read the push notification
		_, err = rd.ReadReply()
		if err != nil {
			internal.Logger.Printf(context.Background(), "push: error reading push notification: %v", err)
			return nil
		}
	}
	return nil
}

// willHandleNotificationInClient checks if a notification type should be ignored by the push notification
// processor and handled by other specialized systems instead (pub/sub, streams, keyspace, etc.).
func willHandleNotificationInClient(notificationType string) bool {
	switch notificationType {
	// Pub/Sub notifications - handled by pub/sub system
	case "message", // Regular pub/sub message
		"pmessage",     // Pattern pub/sub message
		"subscribe",    // Subscription confirmation
		"unsubscribe",  // Unsubscription confirmation
		"psubscribe",   // Pattern subscription confirmation
		"punsubscribe", // Pattern unsubscription confirmation
		"smessage",     // Sharded pub/sub message (Redis 7.0+)
		"ssubscribe",   // Sharded subscription confirmation
		"sunsubscribe": // Sharded unsubscription confirmation
		return true
	default:
		return false
	}
}
