package queue

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/redis/go-redis/v9"
)

func Enqueue(ctx context.Context, rdb *redis.Client, streamName string, job Job) (string, error) {
	jobJSON, err := json.Marshal(job)
	if err != nil {
		return "", fmt.Errorf("marshal job: %w", err)
	}
	entryID, err := rdb.XAdd(ctx, &redis.XAddArgs{
		Stream: streamName,
		ID:     "*",
		Values: map[string]interface{}{"job": string(jobJSON)},
	}).Result()
	if err != nil {
		return "", fmt.Errorf("xadd to stream %s: %w", streamName, err)
	}
	return entryID, nil
}
