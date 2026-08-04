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
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// WaitDaemonSetPodReady waits until at least one pod selected by the
// DaemonSet's pod selector reports a Ready=True condition. Returns the
// name of the first Ready pod so the caller can target it with
// port-forward (kubectl port-forward needs a stable pod name; selecting
// the DaemonSet directly is flaky when multiple pods exist on a
// multi-node cluster). The DaemonSet's TCP readiness probe binds the
// LMCache server, so once the pod is Ready the HTTP frontend is also
// (very nearly) up — the caller should still retry HTTP calls briefly.
func WaitDaemonSetPodReady(
	ctx context.Context,
	c client.Client,
	key types.NamespacedName,
	timeout time.Duration,
) (string, error) {
	var podName string
	err := wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		ds := &appsv1.DaemonSet{}
		if err := c.Get(ctx, key, ds); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		pods := &corev1.PodList{}
		if err := c.List(ctx, pods,
			client.InNamespace(key.Namespace),
			client.MatchingLabels(ds.Spec.Selector.MatchLabels),
		); err != nil {
			return false, err
		}
		for i := range pods.Items {
			p := &pods.Items[i]
			if p.DeletionTimestamp != nil {
				continue
			}
			if p.Status.Phase != corev1.PodRunning {
				continue
			}
			for _, cond := range p.Status.Conditions {
				if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
					podName = p.Name
					return true, nil
				}
			}
		}
		return false, nil
	})
	if err != nil {
		return "", fmt.Errorf("wait DaemonSet %s pod Ready: %w", key, err)
	}
	return podName, nil
}

// WaitDeploymentAvailable polls a Deployment until its Available
// condition is True. Used to gate test logic on vLLM (or any other
// auxiliary workload) being fully scheduled and accepting traffic.
// We intentionally key off the Available condition rather than
// ReadyReplicas because Available encodes the rolling-update completion
// semantics — ReadyReplicas can flap during a fresh rollout even on a
// clean cluster.
func WaitDeploymentAvailable(
	ctx context.Context,
	c client.Client,
	key types.NamespacedName,
	timeout time.Duration,
) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		dep := &appsv1.Deployment{}
		if err := c.Get(ctx, key, dep); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		for _, cond := range dep.Status.Conditions {
			if cond.Type == appsv1.DeploymentAvailable && cond.Status == corev1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	})
}

// WaitDeploymentAvailableOrImagePullError is WaitDeploymentAvailable with a
// fast-fail short-circuit: it returns a descriptive error the moment any pod
// selected by the Deployment has a container or init container wedged in
// ImagePullBackOff. A wrong image/tag, a missing pull Secret, or a credential
// without pull access never resolves on its own, so blocking for the full
// timeout only delays a guaranteed failure (and buries the registry's own
// error message, which ImagePullBackOff's waiting state carries verbatim).
// Returns nil once Available, the pull error when a pull is wedged, or the
// poll timeout error otherwise.
func WaitDeploymentAvailableOrImagePullError(
	ctx context.Context,
	c client.Client,
	key types.NamespacedName,
	timeout time.Duration,
) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		dep := &appsv1.Deployment{}
		if err := c.Get(ctx, key, dep); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		for _, cond := range dep.Status.Conditions {
			if cond.Type == appsv1.DeploymentAvailable && cond.Status == corev1.ConditionTrue {
				return true, nil
			}
		}
		if msg := imagePullBackOff(ctx, c, key.Namespace, dep.Spec.Selector.MatchLabels); msg != "" {
			return false, fmt.Errorf(
				"deployment %s will never become Available — image pull wedged: %s", key, msg)
		}
		return false, nil
	})
}

// imagePullBackOff scans pods matching sel and returns a one-line description
// (pod, container, image, kubelet message) of the first container or init
// container in ImagePullBackOff, or "" if none. The kubelet message is the
// registry's own error (e.g. "pull access denied" / "manifest unknown"), which
// is what a caller needs to tell a credential problem from a wrong tag.
func imagePullBackOff(ctx context.Context, c client.Client, ns string, sel map[string]string) string {
	pods := &corev1.PodList{}
	if err := c.List(ctx, pods, client.InNamespace(ns), client.MatchingLabels(sel)); err != nil {
		return ""
	}
	for i := range pods.Items {
		p := &pods.Items[i]
		if p.DeletionTimestamp != nil {
			continue
		}
		statuses := append(append([]corev1.ContainerStatus{},
			p.Status.InitContainerStatuses...), p.Status.ContainerStatuses...)
		for _, cs := range statuses {
			if w := cs.State.Waiting; w != nil && w.Reason == "ImagePullBackOff" {
				return fmt.Sprintf("pod %s container %q image %q: %s",
					p.Name, cs.Name, cs.Image, w.Message)
			}
		}
	}
	return ""
}
