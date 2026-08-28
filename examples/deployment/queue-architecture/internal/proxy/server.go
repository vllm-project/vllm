package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/model"
)

type Producer interface {
	Enqueue(ctx context.Context, job model.Job) error
	SubscribeSync() (string, *nats.Subscription, error)
}

// NewServer creates and returns an HTTP server that routes all requests
// through a catch-all handler. The handler peeks at the request body to check
// for "stream":true and dispatches to either HandleStreaming or HandleNonStreaming.
func NewServer(
	prod Producer,
	maxBodyBytes int64,
	requestTimeout time.Duration,
	streamTimeout time.Duration,
) *http.Server {
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			if err.Error() == "http: request body too large" {
				http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
				return
			}
			http.Error(w, "failed to read body", http.StatusBadRequest)
			return
		}

		r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

		isStreaming := peekStreamFlag(bodyBytes)

		var handler http.HandlerFunc
		if isStreaming {
			handler = HandleStreaming(prod, streamTimeout)
		} else {
			handler = HandleNonStreaming(prod, requestTimeout)
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

	var data map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		return false
	}

	if stream, ok := data["stream"]; ok {
		if streamBool, ok := stream.(bool); ok {
			return streamBool
		}
	}

	return false
}
