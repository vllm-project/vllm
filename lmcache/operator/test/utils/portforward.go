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
	"fmt"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// PortForwardSpec identifies what kubectl port-forward should target.
// Target follows the kubectl convention (e.g. "svc/my-cache",
// "pod/my-cache-abc", "deployment/my-cache").
type PortForwardSpec struct {
	Namespace string
	Target    string
}

// PortForward starts a `kubectl port-forward` subprocess and waits until
// the first local port begins accepting TCP connections. Each port arg
// follows kubectl syntax: "LOCAL:REMOTE" or just "PORT" (where the local
// and remote ports are equal). A LOCAL of "0" is replaced with a
// kernel-picked free port chosen before kubectl starts, so concurrent
// specs in the same run never collide on a fixed local port.
//
// Returns:
//   - closer: must be called to terminate the subprocess and free ports.
//     Safe to call multiple times.
//   - localBase: "http://127.0.0.1:<localport>" using the first port mapping.
//
// Namespace is passed via the spec struct rather than encoded into
// target because kubectl requires namespace as a separate -n flag
// and silently ignores prefixes embedded in the target string.
func PortForward(spec PortForwardSpec, ports ...string) (func(), string, error) {
	if len(ports) == 0 {
		return nil, "", fmt.Errorf("PortForward: at least one port mapping is required")
	}

	// Substitute "0:REMOTE" and "0" with a concrete kernel-picked port
	// in every mapping. We resolve the port BEFORE invoking kubectl
	// because kubectl's "LOCAL=0 => pick one" mode writes the chosen
	// port to stdout asynchronously, which is racy to scrape — and
	// because the first mapping's local port is what waitForLocalPort
	// + the returned localBase URL refer to. The race between Close+
	// kubectl-bind is acceptable: we don't run concurrent forwards in
	// the same spec, and other processes binding ephemerals during that
	// microsecond is highly unlikely.
	resolved := make([]string, len(ports))
	for i, p := range ports {
		r, err := resolvePortMapping(p)
		if err != nil {
			return nil, "", err
		}
		resolved[i] = r
	}
	localPort, err := localPortFromMapping(resolved[0])
	if err != nil {
		return nil, "", err
	}

	args := []string{"port-forward"}
	if spec.Namespace != "" {
		args = append(args, "-n", spec.Namespace)
	}
	args = append(args, spec.Target)
	args = append(args, resolved...)

	cmd := exec.Command("kubectl", args...)
	if err := cmd.Start(); err != nil {
		return nil, "", fmt.Errorf("start kubectl port-forward: %w", err)
	}

	closer := func() {
		// Killing the process is sufficient — kubectl port-forward
		// closes the listener on SIGKILL, freeing the local port.
		// We drop the Wait error because once we kill, the typical
		// exit status is "signal: killed" which is expected.
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	}

	if err := waitForLocalPort(localPort, 30*time.Second); err != nil {
		closer()
		return nil, "", fmt.Errorf("port-forward to %s/%s did not become ready: %w",
			spec.Namespace, spec.Target, err)
	}
	return closer, fmt.Sprintf("http://127.0.0.1:%d", localPort), nil
}

// resolvePortMapping replaces a "0:REMOTE" mapping (or bare "0") with
// "<picked>:REMOTE" using a kernel-allocated ephemeral port. Mappings
// that already specify a concrete LOCAL pass through unchanged.
func resolvePortMapping(mapping string) (string, error) {
	parts := strings.SplitN(mapping, ":", 2)
	local := parts[0]
	if local != "0" {
		return mapping, nil
	}
	picked, err := pickEphemeralPort()
	if err != nil {
		return "", fmt.Errorf("pick ephemeral local port: %w", err)
	}
	if len(parts) == 1 {
		// "0" alone is ambiguous — there's no remote to forward to.
		return "", fmt.Errorf("invalid port mapping %q: LOCAL=0 requires an explicit :REMOTE", mapping)
	}
	return fmt.Sprintf("%d:%s", picked, parts[1]), nil
}

// pickEphemeralPort asks the kernel for a free TCP port on 127.0.0.1
// and immediately releases it. There is a small race window between
// release and kubectl re-binding; tests don't run forwards concurrently
// inside a single spec, and parallel host workloads rarely steal the
// exact port in that microsecond.
func pickEphemeralPort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		return 0, err
	}
	return port, nil
}

// localPortFromMapping extracts the LOCAL port from a kubectl mapping
// of the form "LOCAL:REMOTE" or just "PORT" (where the local and remote
// ports are equal). Expects LOCAL to already be a concrete integer —
// the "0" case is handled by resolvePortMapping upstream.
func localPortFromMapping(mapping string) (int, error) {
	parts := strings.SplitN(mapping, ":", 2)
	p, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, fmt.Errorf("invalid port mapping %q: %w", mapping, err)
	}
	return p, nil
}

// waitForLocalPort polls 127.0.0.1:port until a TCP connection succeeds
// or timeout elapses. kubectl port-forward briefly accepts connections
// and immediately closes them once before the upstream is wired, so we
// also require the connection to stay open long enough to write to.
func waitForLocalPort(port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("local port %d not reachable after %s", port, timeout)
}
