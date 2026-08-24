package redis

import (
	"time"

	"github.com/redis/go-redis/v9/internal"
)

// DialRetryBackoffConstant returns a dial retry backoff function that always returns d.
// attempt is 0-based: attempt=0 is the delay after the 1st failed dial.
func DialRetryBackoffConstant(d time.Duration) func(attempt int) time.Duration {
	if d < 0 {
		d = 0
	}
	return func(int) time.Duration { return d }
}

// DialRetryBackoffExponential returns a dial retry backoff function that uses exponential
// backoff with jitter and a cap, using internal.RetryBackoff.
//
// attempt is 0-based: attempt=0 is the delay after the 1st failed dial.
func DialRetryBackoffExponential(minBackoff, maxBackoff time.Duration) func(attempt int) time.Duration {
	if minBackoff < 0 {
		minBackoff = 0
	}
	if maxBackoff < 0 {
		maxBackoff = 0
	}
	if minBackoff > maxBackoff {
		minBackoff = maxBackoff
	}
	return func(attempt int) time.Duration {
		// internal.RetryBackoff expects retry >= 0.
		if attempt < 0 {
			attempt = 0
		}
		return internal.RetryBackoff(attempt, minBackoff, maxBackoff)
	}
}
