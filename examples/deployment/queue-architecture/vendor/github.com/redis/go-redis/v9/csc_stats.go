package redis

// CSCStats reports cumulative client-side cache activity and current
// residency.
//
// Experimental: this API may change in a minor release.
type CSCStats struct {
	Hits             uint64
	Misses           uint64
	Entries          int
	MemoryUsageBytes int64
}

// cacheStatsReporter is an optional interface a Cache implementation may
// satisfy to expose statistics. The built-in LocalCache does; user
// implementations are not required to.
type cacheStatsReporter interface {
	Stats() CSCStats
}

// CSCStats returns statistics for this client's client-side cache, read from
// the shared cache when its implementation exposes them (the built-in
// LocalCache does).
//
// It returns a zero value when CSC is not configured or stats are unavailable.
//
// Experimental: this API may change in a minor release.
func (c *Client) CSCStats() CSCStats {
	if c == nil || c.baseClient.csc == nil {
		return CSCStats{}
	}
	if r, ok := c.baseClient.csc.(cacheStatsReporter); ok {
		return r.Stats()
	}
	return CSCStats{}
}
