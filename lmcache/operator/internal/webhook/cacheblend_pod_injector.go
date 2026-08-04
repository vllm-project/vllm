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

package webhook

import (
	"context"
	"net/http"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// CacheBlend annotation keys the webhook reads and stamps (design §8).
const (
	// AnnotationEngine binds a pod to a CacheBlendEngine. Its presence is the
	// opt-in signal; its value is the engine name, optionally namespace-qualified
	// as "<namespace>/<name>" to bind a shared engine in ANOTHER namespace. A
	// bare name resolves in the pod's own namespace.
	AnnotationEngine = "lmcache.ai/cacheblend-engine"

	// AnnotationContainer optionally names the target vLLM container; empty or
	// absent selects the first container.
	AnnotationContainer = "lmcache.ai/cacheblend-container"

	// AnnotationImagePullSecrets optionally overrides the engine's
	// injection.imagePullSecrets with a comma-separated list of Secret names
	// appended to the pod's spec.imagePullSecrets for the private payload image.
	AnnotationImagePullSecrets = "lmcache.ai/cacheblend-image-pull-secrets"

	// AnnotationInjected is the idempotency guard stamped after a successful
	// injection; a re-admitted pod carrying it is allowed unchanged.
	AnnotationInjected = "lmcache.ai/cacheblend-injected"

	// AnnotationSkipReason records why injection was skipped (fail-open).
	AnnotationSkipReason = "lmcache.ai/cacheblend-skip-reason"
)

// CacheBlendLabelInject is the opt-in label the MutatingWebhookConfiguration's
// objectSelector matches (mutating_webhook_selectors_patch.yaml). It gates which
// pods reach the webhook; the handler itself gates on AnnotationEngine.
const CacheBlendLabelInject = "lmcache.ai/cacheblend-inject"

// SkipReasonPayloadImageUnset is stamped when the engine's
// injection.payloadImage resolves to an empty reference (no repository). The
// webhook skips rather than inject an init container with an empty image, which
// the API server would reject. CRD validation normally prevents this. This skip
// reason is CacheBlend-only (the shared reasons live in pod_inject_common.go).
const SkipReasonPayloadImageUnset = "payload-image-unset"

// cacheBlendKeys is the CacheBlend injector's annotation key set, consumed by
// the shared gate / skip / stamp helpers.
var cacheBlendKeys = injectionKeys{
	engine:     AnnotationEngine,
	container:  AnnotationContainer,
	injected:   AnnotationInjected,
	skipReason: AnnotationSkipReason,
}

// +kubebuilder:webhook:path=/mutate--v1-pod,mutating=true,failurePolicy=ignore,sideEffects=None,groups="",resources=pods,verbs=create,versions=v1,name=mcacheblendpod.lmcache.ai,admissionReviewVersions=v1,reinvocationPolicy=Never

// CacheBlendPodInjector is the mutating admission handler that injects the
// lmcache-cacheblend vLLM plugin into opted-in pods (design §7). It is gated by
// the CacheBlendEngine CR: it mutates a pod only when the pod's
// lmcache.ai/cacheblend-engine annotation names an engine whose connection
// ConfigMap exists. It fails open (failurePolicy: Ignore) and is idempotent.
type CacheBlendPodInjector struct {
	// Client reads the named CacheBlendEngine and its connection ConfigMap. It
	// uses the shared manager ServiceAccount, whose RBAC already grants
	// cacheblendengines get and configmaps get (design §7 RBAC note).
	Client client.Client

	// Decoder decodes the admission request's raw pod object.
	Decoder admission.Decoder
}

// Handle implements admission.Handler. It applies mutations M0–M7 to an opted-in
// pod whose named CacheBlendEngine connection ConfigMap exists, then returns a
// JSON patch. It short-circuits to an unchanged Allowed response for non-opted-in
// or already-injected pods, and stamps a skip-reason annotation (still Allowed,
// fail-open) when it declines to mutate (engine missing, command override,
// payload image unset, target container missing, or user-supplied
// --kv-transfer-config).
func (p *CacheBlendPodInjector) Handle(ctx context.Context, req admission.Request) admission.Response {
	log := ctrl.LoggerFrom(ctx)

	pod, engineRef, podNamespace, resp, handled := cacheBlendKeys.gate(p.Decoder, req)
	if handled {
		return resp
	}

	// Resolve an optional "<namespace>/<name>" engine reference. A bare name
	// resolves in the pod's namespace (the same-namespace default); a namespaced
	// ref targets a shared engine in another namespace.
	engineNamespace, engineName := parseEngineRef(engineRef, podNamespace)

	// (3a) Resolve the engine CR (in the engine's namespace) for its injection
	// defaults.
	engine := &lmcachev1alpha1.CacheBlendEngine{}
	if err := p.Client.Get(ctx, types.NamespacedName{Name: engineName, Namespace: engineNamespace}, engine); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Skipped CacheBlend injection: engine not found",
				"engine", engineName, "namespace", engineNamespace)
			return cacheBlendKeys.skip(req, pod, SkipReasonEngineNotFound)
		}
		return admission.Errored(http.StatusInternalServerError, err)
	}
	engine.SetDefaults()

	// (3b/6/4) Read the connection ConfigMap (in the engine's namespace), resolve
	// the target container, and apply the command-override gate (shared with the
	// LMCache injector).
	kvTransferConfigJSON, containerIdx, resp, ok := prepareInjection(
		ctx, p.Client, req, pod, cacheBlendKeys, engineName, engineNamespace,
		engine.Spec.Injection.TargetContainer)
	if !ok {
		return resp
	}
	target := &pod.Spec.Containers[containerIdx]

	// (5) user --kv-transfer-config gate: skip that flag (do not clobber the
	// user's structured JSON) but still apply the rest of the mutation.
	userHasKVTransferConfig := argsHasFlag(target.Args, cbFlagKVTransferConfig)

	// --- Apply mutations M0–M7 ---

	// M0: pod hostIPC for CUDA IPC with the node-local engine.
	pod.Spec.HostIPC = true

	// M1: shared emptyDir volume.
	pod.Spec.Volumes = appendVolumeIfAbsent(pod.Spec.Volumes, BuildCBPluginVolume())

	// M2: payload init container (payloadImage is an ImageSpec: repo/tag/policy).
	// Fail open if the image resolves to empty (no repository) rather than inject
	// an init container with an empty image, which the API server would reject.
	payloadRef, payloadPullPolicy := resolvePayloadImage(engine.Spec.Injection.PayloadImage)
	if payloadRef == "" {
		log.Info("Skipped CacheBlend injection: payload image repository is unset",
			"engine", engineName)
		return cacheBlendKeys.skip(req, pod, SkipReasonPayloadImageUnset)
	}
	pod.Spec.InitContainers = appendInitContainerIfAbsent(pod.Spec.InitContainers,
		BuildCBInitContainer(payloadRef, payloadPullPolicy))

	// M3: read-only mount on the target container.
	target.VolumeMounts = appendVolumeMountIfAbsent(target.VolumeMounts, BuildCBVolumeMount())

	// M4: PYTHONPATH on the target container.
	target.Env = BuildCBPodEnv(target.Env)

	// M5: required vLLM args. Pass "" for the kv-transfer-config JSON when the
	// user already supplies one so BuildCBArgs leaves their value untouched.
	kvForArgs := kvTransferConfigJSON
	if userHasKVTransferConfig {
		kvForArgs = ""
	}
	cudagraph := deref(engine.Spec.Injection.Cudagraph)
	target.Args = BuildCBArgs(target.Args, kvForArgs, cudagraph)

	// M7: append injection pull secrets (annotation override wins) to the pod's
	// imagePullSecrets, deduped (private payload image).
	injectedSecrets := resolveInjectedPullSecrets(engine.Spec.Injection.ImagePullSecrets,
		pod.Annotations[AnnotationImagePullSecrets])
	pod.Spec.ImagePullSecrets = MergeImagePullSecrets(pod.Spec.ImagePullSecrets, injectedSecrets)

	// M6: stamp the idempotency guard (+ skip reason if --kv-transfer-config was
	// user-supplied) and return the patch.
	log.Info("Injected CacheBlend plugin", "engine", engineName, "container", target.Name)
	return cacheBlendKeys.stampInjected(req, pod, userHasKVTransferConfig)
}

// resolveInjectedPullSecrets returns the pull-secret references to inject: the
// per-pod annotation override (a comma-separated list of Secret names) when
// present, otherwise the engine's injection.imagePullSecrets.
//
// Parameters:
//   - specSecrets: the engine's injection.imagePullSecrets.
//   - annotationCSV: the cacheblend-image-pull-secrets annotation value.
func resolveInjectedPullSecrets(
	specSecrets []corev1.LocalObjectReference,
	annotationCSV string,
) []corev1.LocalObjectReference {
	csv := strings.TrimSpace(annotationCSV)
	if csv == "" {
		return specSecrets
	}
	out := make([]corev1.LocalObjectReference, 0)
	for part := range strings.SplitSeq(csv, ",") {
		name := strings.TrimSpace(part)
		if name == "" {
			continue
		}
		out = append(out, corev1.LocalObjectReference{Name: name})
	}
	return out
}

// parseEngineRef splits an engine reference of the form "<namespace>/<name>"
// into its parts. A bare "<name>" (no slash) resolves in defaultNS — the pod's
// namespace — preserving the same-namespace default. A blank namespace (e.g.
// "/name") also falls back to defaultNS. Returns (namespace, name).
func parseEngineRef(ref, defaultNS string) (namespace, name string) {
	namespace, name = defaultNS, strings.TrimSpace(ref)
	if ns, n, found := strings.Cut(name, "/"); found {
		namespace, name = strings.TrimSpace(ns), strings.TrimSpace(n)
	}
	if namespace == "" {
		namespace = defaultNS
	}
	return namespace, name
}

// appendVolumeIfAbsent appends v to volumes unless a volume of the same name is
// already present (idempotency within a single Handle call). Returns the slice.
func appendVolumeIfAbsent(volumes []corev1.Volume, v corev1.Volume) []corev1.Volume {
	for i := range volumes {
		if volumes[i].Name == v.Name {
			return volumes
		}
	}
	return append(volumes, v)
}

// appendInitContainerIfAbsent appends c to initContainers unless one of the same
// name is already present. Returns the slice.
func appendInitContainerIfAbsent(
	initContainers []corev1.Container,
	c corev1.Container,
) []corev1.Container {
	for i := range initContainers {
		if initContainers[i].Name == c.Name {
			return initContainers
		}
	}
	return append(initContainers, c)
}

// appendVolumeMountIfAbsent appends m to mounts unless one of the same name is
// already present. Returns the slice.
func appendVolumeMountIfAbsent(
	mounts []corev1.VolumeMount,
	m corev1.VolumeMount,
) []corev1.VolumeMount {
	for i := range mounts {
		if mounts[i].Name == m.Name {
			return mounts
		}
	}
	return append(mounts, m)
}

// deref returns the value pointed to by s, or "" if s is nil.
func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// resolvePayloadImage builds the "<repository>:<tag>" reference and pull policy
// for the payload init container from the engine's injection.payloadImage. Tag
// and pull policy fall back to "latest" / IfNotPresent when unset; repository is
// taken as-is (it has no sensible cluster-wide default — see InjectionSpec docs).
func resolvePayloadImage(img *lmcachev1alpha1.ImageSpec) (string, corev1.PullPolicy) {
	if img == nil || deref(img.Repository) == "" {
		return "", corev1.PullIfNotPresent
	}
	tag := deref(img.Tag)
	if tag == "" {
		tag = "latest"
	}
	policy := corev1.PullPolicy(deref(img.PullPolicy))
	if policy == "" {
		policy = corev1.PullIfNotPresent
	}
	return deref(img.Repository) + ":" + tag, policy
}
