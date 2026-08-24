package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"
)

// NewServer creates and returns an HTTP server that routes all requests
// through a catch-all handler. The handler peeks at the request body to check
// for "stream":true and dispatches to either HandleStreaming or HandleNonStreaming.
func NewServer(rdb *redis.Client, streamName string, maxBodyBytes int64, requestTimeout time.Duration, streamTimeout time.Duration) *http.Server {
	mux := http.NewServeMux()

	// Catch-all handler for all paths
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Wrap the body with MaxBytesReader to enforce size limit
		r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)

		// Read the request body
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			// Check if the error is due to body size exceeding the limit
			if err.Error() == "http: request body too large" {
				http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
				return
			}
			http.Error(w, "failed to read body", http.StatusBadRequest)
			return
		}

		// Restore the body for the downstream handler
		r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

		// Peek at the body to check for "stream":true
		isStreaming := peekStreamFlag(bodyBytes)

		// Dispatch to the appropriate handler
		var handler http.HandlerFunc
		if isStreaming {
			handler = HandleStreaming(rdb, streamName, streamTimeout)
		} else {
			handler = HandleNonStreaming(rdb, streamName, requestTimeout)
		}

		handler(w, r)
	})

	return &http.Server{
		Handler: mux,
	}
}

// peekStreamFlag performs a cheap JSON peek to check if "stream":true is present.
// It returns true if the stream flag is found and set to true, false otherwise.
func peekStreamFlag(bodyBytes []byte) bool {
	if len(bodyBytes) == 0 {
		return false
	}

	// Try to unmarshal into a map to check for the stream field
	var data map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		// If it's not valid JSON, assume non-streaming
		return false
	}

	// Check if "stream" key exists and is true
	if stream, ok := data["stream"]; ok {
		if streamBool, ok := stream.(bool); ok {
			return streamBool
		}
	}

	return false
}
