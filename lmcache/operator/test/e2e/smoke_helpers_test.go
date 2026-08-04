//go:build e2e
// +build e2e

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

package e2e

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2" //nolint:revive,staticcheck
	. "github.com/onsi/gomega"    //nolint:revive,staticcheck

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// nsCounter is a process-wide monotonic counter appended to every
// generated namespace name. CurrentSpecReport().StartTime alone is not
// enough — specs that call createTestNamespace twice in a single
// BeforeEach (e.g. the cross-namespace auth spec, which needs both an
// engine namespace and a source-Secret namespace) would otherwise compute the same name
// twice and collide on AlreadyExists.
var nsCounter atomic.Int64

// createTestNamespace creates a fresh namespace whose name embeds the
// running spec's text to make failing-spec dumps easier to attribute.
// The deferred AfterEach is wired up by the caller via DeferCleanup so
// each spec gets per-test isolation without relying on package state.
//
// The namespace is pre-labeled with the `privileged` PodSecurity profile
// because the LMCache DaemonSet's pod template sets hostIPC=true (and
// privileged=true only when spec.privileged is enabled), and hostIPC alone
// is rejected by the restricted/baseline profiles. On clusters that enforce
// PodSecurity admission (OCP in particular), a `restricted` namespace would
// reject the DaemonSet at creation time, which the smokes need to succeed
// even though they never wait for pods to schedule. The label is a no-op on
// clusters that don't enforce PodSecurity.
func createTestNamespace(ctx context.Context) string {
	GinkgoHelper()
	name := uniqueNamespaceName()
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"pod-security.kubernetes.io/enforce": "privileged",
				"pod-security.kubernetes.io/audit":   "privileged",
				"pod-security.kubernetes.io/warn":    "privileged",
			},
		},
	}
	Expect(k8sClient.Create(ctx, ns)).To(Succeed(), "Failed to create test namespace %q", name)
	DeferCleanup(func(ctx SpecContext) {
		deleteNamespace(ctx, name)
	})
	return name
}

// uniqueNamespaceName produces a DNS-1123-safe namespace name that's
// unique within the test process. The time component disambiguates
// across runs / Kind-cluster restarts; the counter disambiguates
// multiple calls inside a single spec.
func uniqueNamespaceName() string {
	n := nsCounter.Add(1)
	hash := fmt.Sprintf("%x", time.Now().UnixNano())
	if len(hash) > 8 {
		hash = hash[len(hash)-8:]
	}
	return fmt.Sprintf("lmc-smoke-%s-%d", hash, n)
}

// deleteNamespace tears down a test namespace. We tolerate NotFound so
// repeated cleanup calls don't fail; the namespace's contents are GC'd
// asynchronously by Kubernetes.
func deleteNamespace(ctx context.Context, name string) {
	GinkgoHelper()
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if err := k8sClient.Delete(ctx, ns); err != nil && !apierrors.IsNotFound(err) {
		_, _ = fmt.Fprintf(GinkgoWriter, "warning: delete namespace %q failed: %v\n", name, err)
	}
}

// dumpControllerLogs writes the last <tail> lines of every
// controller-manager pod's log to GinkgoWriter. Used by AfterEach
// failure handlers to make CI dumps useful without paging the entire
// reconcile history.
func dumpControllerLogs(tail int) {
	GinkgoHelper()
	podName, err := controllerPodNameOnce()
	if err != nil {
		_, _ = fmt.Fprintf(GinkgoWriter, "warning: could not locate controller pod: %v\n", err)
		return
	}
	cmd := exec.Command("kubectl", "logs", podName,
		"-n", "lmcache-operator-system",
		fmt.Sprintf("--tail=%d", tail),
	)
	out, _ := cmd.CombinedOutput()
	_, _ = fmt.Fprintf(GinkgoWriter, "controller-manager logs (tail=%d):\n%s\n", tail, string(out))
}

// dumpNamespace writes events, pods, and the LMCacheEngine CR yaml from
// the given namespace to GinkgoWriter. Designed to be cheap to call
// from AfterEach without flooding output on success.
func dumpNamespace(ns string) {
	GinkgoHelper()
	for _, args := range [][]string{
		{"get", "events", "-n", ns, "--sort-by=.lastTimestamp"},
		{"get", "lmcacheengines", "-n", ns, "-o", "yaml"},
		{"get", "cacheblendengines", "-n", ns, "-o", "yaml"},
		{"get", "pods", "-n", ns, "-o", "wide"},
		// imagePullSecrets is not rendered by `describe pod`; print it
		// explicitly so a wedged private-image pull can be triaged as
		// "secret not attached" vs "bad credentials".
		{"get", "pods", "-n", ns, "-o",
			"jsonpath={range .items[*]}{.metadata.name}{\" imagePullSecrets=\"}{.spec.imagePullSecrets}{\"\\n\"}{end}"},
		{"describe", "pods", "-n", ns},
	} {
		out, _ := exec.Command("kubectl", args...).CombinedOutput()
		_, _ = fmt.Fprintf(GinkgoWriter, "$ kubectl %v\n%s\n", args, string(out))
	}

	// Per-pod logs (last 200 lines). The vLLM and LMCache pods produce
	// most of the diagnostic signal on failure; without their logs,
	// `describe pod` only tells us readiness flapped, not why. The
	// `--previous` fetches the prior-instantiation logs when the
	// container has crashed and restarted — that's where mid-inference
	// CUDA / Python tracebacks live.
	podNames := exec.Command("kubectl", "get", "pods", "-n", ns,
		"-o", "jsonpath={range .items[*]}{.metadata.name} {end}")
	out, err := podNames.Output()
	if err != nil {
		return
	}
	for _, podName := range strings.Fields(string(out)) {
		for _, args := range [][]string{
			{"logs", "-n", ns, podName, "--tail=200", "--all-containers"},
			{"logs", "-n", ns, podName, "--tail=200", "--all-containers", "--previous"},
		} {
			cmd := exec.Command("kubectl", args...)
			cmd.Stderr = nil // mute "previous terminated container not found" noise
			out, _ := cmd.Output()
			if len(out) == 0 {
				continue
			}
			_, _ = fmt.Fprintf(GinkgoWriter, "$ kubectl %v\n%s\n", args, string(out))
		}
	}
}

// controllerPodNameOnce returns the first running controller-manager
// pod name. We resolve fresh each call rather than caching because the
// pod can be evicted/restarted between specs.
func controllerPodNameOnce() (string, error) {
	cmd := exec.Command("kubectl", "get", "pods",
		"-n", "lmcache-operator-system",
		"-l", "control-plane=controller-manager",
		"-o", "jsonpath={.items[0].metadata.name}",
	)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	if len(out) == 0 {
		return "", fmt.Errorf("no controller-manager pod found")
	}
	return string(out), nil
}

// recordOnFailure runs the standard dump bundle iff the current spec
// failed. Use as the body of an AfterEach so passing specs stay quiet.
func recordOnFailure(ns string) {
	if !CurrentSpecReport().Failed() {
		return
	}
	dumpControllerLogs(200)
	dumpNamespace(ns)
}

// engineKey is shorthand for the typed namespaced name pair used throughout
// the smoke specs.
func engineKey(ns, name string) client.ObjectKey {
	return client.ObjectKey{Namespace: ns, Name: name}
}

// containerArgs returns the args of the first (and only) container in
// the DaemonSet pod template — the smoke contract requires exactly one.
func containerArgs(ds *appsv1.DaemonSet) []string {
	if len(ds.Spec.Template.Spec.Containers) == 0 {
		return nil
	}
	return ds.Spec.Template.Spec.Containers[0].Args
}

// argValue returns the value following the first occurrence of flag in
// args, or "" if flag is not present or has no value. Args are flat
// ["--flag", "val", ...] pairs emitted by BuildContainerArgs.
func argValue(args []string, flag string) string {
	for i := range args {
		if args[i] == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

// argValueLast returns the value following the LAST occurrence of flag.
// argparse-style flag handling lets a later --flag override an earlier
// one, so the operator's contract for spec.extraArgs is "appended last,
// can override any auto-generated flag." This helper mirrors that
// runtime semantics from the operator's perspective: which value would
// the server actually use?
func argValueLast(args []string, flag string) string {
	last := ""
	for i := range args {
		if args[i] == flag && i+1 < len(args) {
			last = args[i+1]
		}
	}
	return last
}

// serviceMonitorCRDInstalled reports whether the Prometheus Operator's
// ServiceMonitor CRD is registered on the target cluster. Specs that
// only make sense in that environment (the ServiceMonitor spec) call this in
// BeforeEach and Skip() when it returns false, so the same suite can
// run unchanged on bare Kind clusters and on clusters with the
// kube-prometheus-stack pre-installed. Errors other than NoMatch
// fail loudly so a transient API hiccup is never silently masked.
func serviceMonitorCRDInstalled() bool {
	GinkgoHelper()
	_, err := k8sClient.RESTMapper().RESTMapping(schema.GroupKind{
		Group: "monitoring.coreos.com",
		Kind:  "ServiceMonitor",
	}, "v1")
	if err == nil {
		return true
	}
	if meta.IsNoMatchError(err) {
		return false
	}
	Fail(fmt.Sprintf("unexpected error querying ServiceMonitor RESTMapping: %v", err))
	return false // unreachable
}

// firstPodForDeployment returns the first non-terminating pod selected by the
// Deployment's pod selector, or nil if none exist yet. Returns the typed Pod
// so callers can read both annotations (injection stamp) and container args.
func firstPodForDeployment(ctx context.Context, ns string, dep *appsv1.Deployment) *corev1.Pod {
	pods := &corev1.PodList{}
	if err := k8sClient.List(ctx, pods,
		client.InNamespace(ns),
		client.MatchingLabels(dep.Spec.Selector.MatchLabels),
	); err != nil {
		return nil
	}
	for i := range pods.Items {
		if pods.Items[i].DeletionTimestamp == nil {
			return &pods.Items[i]
		}
	}
	return nil
}

// podInitContainer returns the named init container of the pod, or nil.
func podInitContainer(pod *corev1.Pod, name string) *corev1.Container {
	for i := range pod.Spec.InitContainers {
		if pod.Spec.InitContainers[i].Name == name {
			return &pod.Spec.InitContainers[i]
		}
	}
	return nil
}

// podContainer returns the named (non-init) container of the pod, or nil.
func podContainer(pod *corev1.Pod, name string) *corev1.Container {
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == name {
			return &pod.Spec.Containers[i]
		}
	}
	return nil
}

// containerEnvValue returns the value of env var name on the container, or "".
func containerEnvValue(c *corev1.Container, name string) string {
	for _, e := range c.Env {
		if e.Name == name {
			return e.Value
		}
	}
	return ""
}

// containerVolumeMount returns the named volume mount on the container, or nil.
func containerVolumeMount(c *corev1.Container, name string) *corev1.VolumeMount {
	for i := range c.VolumeMounts {
		if c.VolumeMounts[i].Name == name {
			return &c.VolumeMounts[i]
		}
	}
	return nil
}

// podHasEmptyDirVolume reports whether the pod carries an emptyDir volume of the
// given name.
func podHasEmptyDirVolume(pod *corev1.Pod, name string) bool {
	for i := range pod.Spec.Volumes {
		if pod.Spec.Volumes[i].Name == name {
			return pod.Spec.Volumes[i].EmptyDir != nil
		}
	}
	return false
}

// vllmContainerArgs returns the args of the vLLM container in the pod (the one
// named "vllm", falling back to the first container). The mutating webhooks
// append vLLM flags to this container's args.
func vllmContainerArgs(pod *corev1.Pod) []string {
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == "vllm" {
			return pod.Spec.Containers[i].Args
		}
	}
	if len(pod.Spec.Containers) > 0 {
		return pod.Spec.Containers[0].Args
	}
	return nil
}
