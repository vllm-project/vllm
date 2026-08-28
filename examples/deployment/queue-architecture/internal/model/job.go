package model

// Job is a queued HTTP request processed by a sidecar.
type Job struct {
	JobID   string            `json:"job_id"`
	Method  string            `json:"method"`
	Path    string            `json:"path"`
	Headers map[string]string `json:"headers"`
	Body    []byte            `json:"body"`
	Stream  bool              `json:"stream"`
	ReplyTo string            `json:"reply_to"`
}
