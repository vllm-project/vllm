package queue

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

func Read(ctx context.Context, rdb *redis.Client, streamName, groupName, consumerName string) (Job, string, error) {
	res, err := rdb.XReadGroup(ctx, &redis.XReadGroupArgs{
		Group:    groupName,
		Consumer: consumerName,
		Streams:  []string{streamName, ">"},
		Count:    1,
		Block:    0,
	}).Result()
	if err != nil {
		return Job{}, "", fmt.Errorf("xreadgroup: %w", err)
	}
	if len(res) == 0 || len(res[0].Messages) == 0 {
		return Job{}, "", fmt.Errorf("no messages returned")
	}
	msg := res[0].Messages[0]
	raw, ok := msg.Values["job"].(string)
	if !ok {
		return Job{}, "", fmt.Errorf("message %s missing string field 'job'", msg.ID)
	}
	var job Job
	if err := json.Unmarshal([]byte(raw), &job); err != nil {
		return Job{}, "", fmt.Errorf("unmarshal job %s: %w", msg.ID, err)
	}
	return job, msg.ID, nil
}

func Ack(ctx context.Context, rdb *redis.Client, streamName, groupName, entryID string) error {
	if err := rdb.XAck(ctx, streamName, groupName, entryID).Err(); err != nil {
		return fmt.Errorf("xack stream %s group %s entry %s: %w", streamName, groupName, entryID, err)
	}
	return nil
}

// ClaimedJob represents a job that was claimed from the pending entries list,
// including both the job data and its stream entry ID.
type ClaimedJob struct {
	Job     Job
	EntryID string
}

// Claim reclaims idle jobs from the pending entries list using XAutoClaim.
// It returns both the jobs and their entry IDs so they can be processed and acked.
func Claim(ctx context.Context, rdb *redis.Client, streamName, groupName, consumerName string, idleThreshold time.Duration) ([]ClaimedJob, error) {
	msgs, _, err := rdb.XAutoClaim(ctx, &redis.XAutoClaimArgs{
		Stream:   streamName,
		Group:    groupName,
		Consumer: consumerName,
		MinIdle:  idleThreshold,
		Start:    "0",
	}).Result()
	if err != nil {
		return nil, fmt.Errorf("xautoclaim: %w", err)
	}
	claimedJobs := make([]ClaimedJob, 0, len(msgs))
	var errs []error
	for _, msg := range msgs {
		raw, ok := msg.Values["job"].(string)
		if !ok {
			errs = append(errs, fmt.Errorf("claimed job %s: type assertion failed (job field not a string)", msg.ID))
			continue
		}
		var job Job
		if err := json.Unmarshal([]byte(raw), &job); err != nil {
			errs = append(errs, fmt.Errorf("claimed job %s: unmarshal failed: %w", msg.ID, err))
			continue
		}
		claimedJobs = append(claimedJobs, ClaimedJob{
			Job:     job,
			EntryID: msg.ID,
		})
	}
	if len(errs) > 0 {
		return claimedJobs, errors.Join(errs...)
	}
	return claimedJobs, nil
}

// EnsureConsumerGroup idempotently creates the Redis stream and consumer group.
// It uses XGroupCreateMkStream to create the stream if it doesn't exist and
// creates the consumer group starting from the end of the stream ("$").
// The "BUSYGROUP Consumer Group name already exists" error is treated as success
// so this function is safe to call every time the sidecar starts, including restarts.
func EnsureConsumerGroup(ctx context.Context, rdb *redis.Client, streamName, groupName string) error {
	err := rdb.XGroupCreateMkStream(ctx, streamName, groupName, "$").Err()
	if err != nil {
		// Treat BUSYGROUP error as success (group already exists)
		if strings.Contains(err.Error(), "BUSYGROUP") {
			return nil
		}
		return fmt.Errorf("xgroupcreate stream %s group %s: %w", streamName, groupName, err)
	}
	return nil
}
