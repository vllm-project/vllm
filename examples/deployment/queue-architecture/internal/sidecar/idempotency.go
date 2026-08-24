package sidecar

import (
	"context"
	"fmt"

	"github.com/redis/go-redis/v9"
)

// ResultExists checks whether a job's result has already been produced in Redis.
// It returns true if the result key exists, false if not, and an error if the check fails.
func ResultExists(ctx context.Context, rdb *redis.Client, jobID string) (bool, error) {
	count, err := rdb.Exists(ctx, "result:"+jobID).Result()
	if err != nil {
		return false, fmt.Errorf("exists check for result:%s: %w", jobID, err)
	}
	return count > 0, nil
}
