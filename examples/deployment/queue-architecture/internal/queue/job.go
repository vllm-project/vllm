package queue

// Job represents a queued HTTP request to be processed by the sidecar.
type Job struct {
	JobID   string            `json:"job_id"`
	Method  string            `json:"method"`
	Path    string            `json:"path"`
	Headers map[string]string `json:"headers"`
	Body    []byte            `json:"body"`
	Stream  bool              `json:"stream"`
}
