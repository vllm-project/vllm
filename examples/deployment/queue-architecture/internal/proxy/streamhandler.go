package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/queue"
)

// HandleStreaming returns an HTTP handler that processes streaming requests.
// It builds a Job with Stream=true, enqueues it, subscribes to the job's Pub/Sub channel,
// and relays each chunk to the client as SSE until receiving the done sentinel.
func HandleStreaming(rdb *redis.Client, streamName string, timeout time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		// Generate a unique job ID
		jobID := fmt.Sprintf("job-%d", time.Now().UnixNano())

		// Read request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to read body: %v", err), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		// Build the Job with Stream=true
		job := queue.Job{
			JobID:   jobID,
			Method:  r.Method,
			Path:    r.RequestURI,
			Headers: headerMapFromRequest(r),
			Body:    body,
			Stream:  true,
		}

		// Subscribe to the job's Pub/Sub channel BEFORE enqueueing
		channel := fmt.Sprintf("stream:%s", jobID)
		subscription := rdb.Subscribe(ctx, channel)
		defer subscription.Close()

		// Receive confirmation that the subscription is established
		// (go-redis sends SUBSCRIBE asynchronously; Receive blocks until confirmed)
		_, err = subscription.Receive(ctx)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to subscribe: %v", err), http.StatusInternalServerError)
			return
		}

		// Set up SSE headers
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Enqueue the job AFTER subscription is confirmed
		_, err = queue.Enqueue(ctx, rdb, streamName, job)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to enqueue job: %v", err), http.StatusInternalServerError)
			return
		}

		// Create a timeout context for the subscription
		timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		// Read from the subscription channel and relay to client
		for {
			select {
			case <-timeoutCtx.Done():
				// Timeout occurred
				fmt.Fprintf(w, "data: {\"error\": \"timeout\"}\n\n")
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
				return
			case msg := <-subscription.Channel():
				if msg == nil {
					// Channel closed
					return
				}

				// Parse the message payload
				var payload map[string]interface{}
				if err := json.Unmarshal([]byte(msg.Payload), &payload); err != nil {
					fmt.Fprintf(w, "data: {\"error\": \"invalid payload\"}\n\n")
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
					continue
				}

				// Check for done sentinel
				if done, ok := payload["__done"].(bool); ok && done {
					// Send final message and close
					fmt.Fprintf(w, "data: {\"__done\": true}\n\n")
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
					return
				}

				// Write chunk as SSE
				data, err := json.Marshal(payload)
				if err != nil {
					fmt.Fprintf(w, "data: {\"error\": \"marshal failed\"}\n\n")
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
					continue
				}

				fmt.Fprintf(w, "data: %s\n\n", string(data))
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
		}
	}
}

// headerMapFromRequest extracts headers from the HTTP request.
func headerMapFromRequest(r *http.Request) map[string]string {
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 {
			headers[key] = values[0]
		}
	}
	return headers
}
