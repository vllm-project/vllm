/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

// httpJSONClient is the shared client used by HTTP helpers. We do NOT
// set Client.Timeout because that cap would override the caller's
// context deadline — and callers like the vLLM-integration spec need
// minutes-long deadlines for cold-cache completions. Every helper here
// uses NewRequestWithContext, so cancellation is governed entirely by
// the caller's ctx.
//
// DisableKeepAlives is on purpose: every test request goes through a
// `kubectl port-forward` proxy that quietly drops idle TCP connections
// after a few seconds. With keep-alive on, the second HTTP call in a
// spec ends up trying to reuse a dead connection and gets EOF before
// the request even reaches the upstream. Re-establishing the TCP
// connection per request costs a few ms in tests; the alternative is
// flaky failures that look like upstream crashes when they aren't.
var httpJSONClient = &http.Client{
	Transport: &http.Transport{DisableKeepAlives: true},
}

// HTTPGetJSON issues a GET against url and decodes the response body
// into out. Returns an error if the request fails, the status is not
// 2xx, or the body is not valid JSON. The caller owns out's lifetime.
func HTTPGetJSON(ctx context.Context, url string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("build GET %s: %w", url, err)
	}
	resp, err := httpJSONClient.Do(req)
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read GET %s body: %w", url, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("GET %s: status %d body=%s", url, resp.StatusCode, string(body))
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("decode JSON from GET %s: %w\nbody=%s", url, err, string(body))
	}
	return nil
}

// HTTPGetText issues a GET against url and returns the raw response
// body as a string. Used for endpoints that don't speak JSON
// (Prometheus text exposition at /metrics).
func HTTPGetText(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("build GET %s: %w", url, err)
	}
	resp, err := httpJSONClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("GET %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read GET %s body: %w", url, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("GET %s: status %d body=%s", url, resp.StatusCode, string(body))
	}
	return string(body), nil
}

// HTTPPostJSON issues a POST with a JSON-encoded body and decodes the
// 2xx response body into out (when non-nil). Returns the parsed JSON
// status code so callers that care about distinguishing 200 vs 201 can
// check it themselves.
func HTTPPostJSON(ctx context.Context, url string, body, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("encode POST %s body: %w", url, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return fmt.Errorf("build POST %s: %w", url, err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpJSONClient.Do(req)
	if err != nil {
		return fmt.Errorf("POST %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read POST %s body: %w", url, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("POST %s: status %d body=%s", url, resp.StatusCode, string(respBody))
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("decode JSON from POST %s: %w\nbody=%s", url, err, string(respBody))
	}
	return nil
}

// WaitHTTP200 polls url with GET until it returns 2xx or the timeout
// elapses. Tighter than HTTPGetJSON because it tolerates transient
// connection-refused / status-503 noise during pod startup: the LMCache
// HTTP frontend is reachable on the readiness-probed port (TCP-readiness
// just confirms the ZMQ socket is bound), and the FastAPI app may not
// have finished initializing yet. The poll absorbs that race so the
// caller's assertion runs against a fully-ready server.
func WaitHTTP200(ctx context.Context, url string, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return false, err
		}
		resp, err := httpJSONClient.Do(req)
		if err != nil {
			return false, nil
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		return resp.StatusCode >= 200 && resp.StatusCode < 300, nil
	})
}
