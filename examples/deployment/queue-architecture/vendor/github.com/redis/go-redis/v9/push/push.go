// Package push provides push notifications for Redis.
// This is an EXPERIMENTAL API for handling push notifications from Redis.
// It is not yet stable and may change in the future.
// Although this is in a public package, in its current form public use is not advised.
// Pending push notifications should be processed before executing any readReply from the connection
// as per RESP3 specification push notifications can be sent at any time.
package push
