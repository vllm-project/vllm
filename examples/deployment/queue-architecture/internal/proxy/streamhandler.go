package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/oklog/ulid/v2"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/model"
)

// HandleStreaming returns an HTTP handler that processes streaming requests.
// It builds a Job with Stream=true, subscribes on a NATS inbox before enqueue,
// and relays each inbox token to the client as SSE until receiving __done.
func HandleStreaming(prod Producer, timeout time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to read body: %v", err), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		job := model.Job{
			JobID:   ulid.Make().String(),
			Method:  r.Method,
			Path:    r.RequestURI,
			Headers: headerMapFromRequest(r),
			Body:    body,
			Stream:  true,
		}

		inbox, sub, err := prod.SubscribeSync()
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to subscribe inbox: %v", err), http.StatusInternalServerError)
			return
		}
		defer sub.Unsubscribe()

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		job.ReplyTo = inbox

		if err := prod.Enqueue(ctx, job); err != nil {
			status, msg := enqueueHTTPStatus(err)
			http.Error(w, msg, status)
			return
		}

		timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		for {
			msg, err := sub.NextMsgWithContext(timeoutCtx)
			if err != nil {
				if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, nats.ErrTimeout) {
					fmt.Fprintf(w, "data: {\"error\": \"timeout\"}\n\n")
					flushSSE(w)
					return
				}
				fmt.Fprintf(w, "data: {\"error\": \"subscription error\"}\n\n")
				flushSSE(w)
				return
			}

			var payload map[string]interface{}
			if err := json.Unmarshal(msg.Data, &payload); err != nil {
				fmt.Fprintf(w, "data: {\"error\": \"invalid payload\"}\n\n")
				flushSSE(w)
				continue
			}

			if done, ok := payload["__done"].(bool); ok && done {
				fmt.Fprintf(w, "data: {\"__done\": true}\n\n")
				flushSSE(w)
				return
			}

			data, err := json.Marshal(payload)
			if err != nil {
				fmt.Fprintf(w, "data: {\"error\": \"marshal failed\"}\n\n")
				flushSSE(w)
				continue
			}

			fmt.Fprintf(w, "data: %s\n\n", string(data))
			flushSSE(w)
		}
	}
}

func headerMapFromRequest(r *http.Request) map[string]string {
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 {
			headers[key] = values[0]
		}
	}
	return headers
}

func flushSSE(w http.ResponseWriter) {
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}
