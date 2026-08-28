package sidecar

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/queue"
	"golang.org/x/sync/errgroup"
)

const sidecarFetchBatchSize = 1

const inProgressInterval = 15 * time.Second // AckWait / 2

type ConsumerClient interface {
	Fetch(ctx context.Context, batchSize int) ([]queue.Message, error)
	Mail(recipient string, data []byte) error
}

type Consumer struct {
	client ConsumerClient
}

func NewConsumer(client ConsumerClient) *Consumer {
	return &Consumer{
		client: client,
	}
}

// processJob forwards a pulled job to vLLM, publishes the reply on the proxy
// inbox, and ACKs the JetStream work message. Reply publish failures are logged
// but never block ACK (orphan replies are acceptable when the proxy is gone).
func (c *Consumer) processJob(ctx context.Context, fetched queue.Message, target string) {
	job := fetched.Job
	replyTo := fetched.ReplyTo()

	if job.Stream {
		err := withInProgressErr(ctx, fetched, func(ctx context.Context) error {
			return ForwardStreaming(ctx, c.client, replyTo, job, target)
		})
		if err != nil {
			slog.ErrorContext(ctx, "forward streaming job",
				"job_id", job.JobID,
				"err", err,
			)
			c.publishStreamError(replyTo, 502, err.Error())
		}
	} else {
		status, headers, body, err := withInProgressForward(ctx, fetched, func(ctx context.Context) (int, map[string]string, []byte, error) {
			return ForwardNonStreaming(ctx, job, target)
		})
		if err != nil {
			slog.ErrorContext(ctx, "forward non-streaming job",
				"job_id", job.JobID,
				"err", err,
			)
			c.publishNonStreamReply(replyTo, 502, map[string]string{}, []byte(err.Error()))
		} else {
			c.publishNonStreamReply(replyTo, status, headers, body)
		}
	}

	if err := fetched.Ack(ctx); err != nil {
		slog.ErrorContext(ctx, "ack job",
			"job_id", job.JobID,
			"err", err,
		)
	}
}

func withInProgressErr(ctx context.Context, msg queue.Message, fn func(context.Context) error) error {
	stop := startInProgress(ctx, msg)
	defer stop()
	return fn(ctx)
}

func withInProgressForward(
	ctx context.Context,
	msg queue.Message,
	fn func(context.Context) (int, map[string]string, []byte, error),
) (int, map[string]string, []byte, error) {
	stop := startInProgress(ctx, msg)
	defer stop()
	return fn(ctx)
}

func startInProgress(ctx context.Context, msg queue.Message) func() {
	if msg.Msg == nil {
		return func() {}
	}

	ipCtx, cancel := context.WithCancel(ctx)
	go func() {
		tick := time.NewTicker(inProgressInterval)
		defer tick.Stop()
		for {
			select {
			case <-ipCtx.Done():
				return
			case <-tick.C:
				if err := msg.InProgress(ipCtx); err != nil && ipCtx.Err() == nil {
					slog.WarnContext(ipCtx, "in progress", "err", err)
				}
			}
		}
	}()
	return cancel
}

func (c *Consumer) publishNonStreamReply(replyTo string, status int, headers map[string]string, body []byte) {
	if headers == nil {
		headers = map[string]string{}
	}
	result := map[string]any{
		"status":  status,
		"headers": headers,
		"body":    string(body),
	}
	resultJSON, err := json.Marshal(result)
	if err != nil {
		slog.Error("marshal non-stream reply", "err", err)
		return
	}

	err = c.client.Mail(replyTo, resultJSON)
	if err != nil {
		slog.Warn("failed to mail result",
			"reply_to", replyTo,
			"err", err,
		)
	}
}

func (c *Consumer) publishStreamError(replyTo string, status int, body string) {
	errorMessage := map[string]any{
		"error":  true,
		"status": status,
		"body":   body,
	}
	errorJSON, err := json.Marshal(errorMessage)
	if err != nil {
		slog.Error("marshal stream error reply", "err", err)
		return
	}
	err = c.client.Mail(replyTo, errorJSON)
	if err != nil {
		slog.Warn("failed to mail error",
			"reply_to", replyTo,
			"err", err,
		)
	}
	doneJSON, err := json.Marshal(map[string]bool{"__done": true})
	if err != nil {
		slog.Error("marshal stream done reply", "err", err)
		return
	}
	err = c.client.Mail(replyTo, doneJSON)
	if err != nil {
		slog.Warn("failed to mail done message",
			"reply_to", replyTo,
			"err", err,
		)
	}
}

// ConsumerLoop runs up to maxConcurrent independent worker goroutines, each
// looping: gate on vLLM's real-time load, pull one job from JetStream,
// process it, repeat.
//
// GRACEFUL SHUTDOWN / SCALE-DOWN: this takes two separate contexts, not one.
//   - shutdownCtx is cancelled on SIGTERM (pod being terminated -- scale-down,
//     node drain, etc). It gates every FETCH decision: checked explicitly
//     before each pull attempt, and threaded into waitForCapacity/queue.Fetch
//     so a worker blocked waiting for capacity or for a new message unblocks
//     and stops immediately when shutdown begins.
//   - workCtx is NOT cancelled by shutdown. Once a message has actually been
//     fetched, processJob runs on workCtx, so an in-flight job (already
//     consuming GPU time) always runs to completion -- forwarded, reply
//     published, acked -- regardless of a shutdown signal arriving mid-flight.
func (c *Consumer) ConsumerLoop(
	shutdownCtx, workCtx context.Context,
	target string,
	httpClient *http.Client,
	maxConcurrent int,
	capacityPollInterval time.Duration,
) error {
	workers := maxConcurrent
	if workers <= 0 {
		workers = 1
	}

	g, claimCtx := errgroup.WithContext(shutdownCtx)
	for i := 0; i < workers; i++ {
		workerID := i
		g.Go(func() error {
			return c.runWorker(claimCtx, workCtx, workerID, target, httpClient, maxConcurrent, capacityPollInterval)
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}
	return shutdownCtx.Err()
}

func (c *Consumer) runWorker(
	claimCtx, workCtx context.Context,
	workerID int,
	target string,
	httpClient *http.Client,
	maxConcurrent int,
	capacityPollInterval time.Duration,
) error {
	for {
		select {
		case <-claimCtx.Done():
			return nil
		default:
		}

		if err := waitForCapacity(claimCtx, httpClient, target, maxConcurrent, capacityPollInterval); err != nil {
			if claimCtx.Err() != nil {
				return nil
			}
			return err
		}

		fetched, err := c.client.Fetch(claimCtx, sidecarFetchBatchSize)
		if err != nil {
			if claimCtx.Err() != nil {
				return nil
			}
			return fmt.Errorf("fetch job (worker %d): %w", workerID, err)
		}
		if len(fetched) == 0 {
			continue
		}

		c.processJob(workCtx, fetched[0], target)
	}
}
