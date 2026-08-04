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
	"crypto/tls"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// These package-level handles back the envtest-based integration spec
// (pod_injector_envtest_test.go). The fake-client unit tests in
// pod_injector_test.go do not use them. envtest is started once for the suite;
// if the binaries are missing the suite fails fast (run `make setup-envtest`).
var (
	envtestCtx    context.Context
	envtestCancel context.CancelFunc
	mgrCancel     context.CancelFunc
	testEnv       *envtest.Environment
	cfg           *rest.Config
	k8sClient     client.Client
	envtestScheme *runtime.Scheme
)

// TestWebhook runs the Ginkgo suite for the CacheBlend mutating webhook package.
func TestWebhook(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "CacheBlend Webhook Suite")
}

var _ = BeforeSuite(func() {
	logf.SetLogger(zap.New(zap.WriteTo(GinkgoWriter), zap.UseDevMode(true)))

	envtestCtx, envtestCancel = context.WithCancel(context.TODO())

	envtestScheme = runtime.NewScheme()
	Expect(clientgoscheme.AddToScheme(envtestScheme)).To(Succeed())
	Expect(lmcachev1alpha1.AddToScheme(envtestScheme)).To(Succeed())

	By("bootstrapping the test environment with the mutating webhook installed")
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
		// Point at the generated manifest FILE, not the config/webhook dir: the
		// dir also holds the kustomize selectors patch (a partial
		// MutatingWebhookConfiguration with the same name), which envtest's
		// path loader would try to install and reject as invalid. The base
		// manifest has no objectSelector, so the webhook matches every pod —
		// fine here, since the specs control which pods exist.
		WebhookInstallOptions: envtest.WebhookInstallOptions{
			Paths: []string{filepath.Join("..", "..", "config", "webhook", "manifests.yaml")},
		},
	}
	if dir := firstFoundEnvtestBinaryDir(); dir != "" {
		testEnv.BinaryAssetsDirectory = dir
	}

	var err error
	cfg, err = testEnv.Start()
	Expect(err).NotTo(HaveOccurred())
	Expect(cfg).NotTo(BeNil())

	k8sClient, err = client.New(cfg, client.Options{Scheme: envtestScheme})
	Expect(err).NotTo(HaveOccurred())

	By("starting a manager that serves the CacheBlendPodInjector webhook")
	wio := &testEnv.WebhookInstallOptions
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:  envtestScheme,
		Metrics: metricsserver.Options{BindAddress: "0"},
		WebhookServer: webhook.NewServer(webhook.Options{
			Host:    wio.LocalServingHost,
			Port:    wio.LocalServingPort,
			CertDir: wio.LocalServingCertDir,
		}),
		LeaderElection: false,
	})
	Expect(err).NotTo(HaveOccurred())

	// The handler uses a direct (uncached) client so reads of the
	// CacheBlendEngine and its connection ConfigMap succeed without waiting on
	// informer cache sync — the production wiring uses mgr.GetClient().
	directClient, err := client.New(cfg, client.Options{Scheme: envtestScheme})
	Expect(err).NotTo(HaveOccurred())
	mgr.GetWebhookServer().Register("/mutate--v1-pod", &webhook.Admission{Handler: &CacheBlendPodInjector{
		Client:  directClient,
		Decoder: admission.NewDecoder(mgr.GetScheme()),
	}})

	var mgrCtx context.Context
	mgrCtx, mgrCancel = context.WithCancel(envtestCtx)
	go func() {
		defer GinkgoRecover()
		Expect(mgr.Start(mgrCtx)).To(Succeed())
	}()

	By("waiting for the webhook server's TLS port to accept connections")
	addr := fmt.Sprintf("%s:%d", wio.LocalServingHost, wio.LocalServingPort)
	Eventually(func() error {
		conn, derr := tls.DialWithDialer(&net.Dialer{Timeout: time.Second}, "tcp", addr,
			&tls.Config{InsecureSkipVerify: true}) //nolint:gosec // test-only readiness probe
		if derr != nil {
			return derr
		}
		return conn.Close()
	}, 30*time.Second, 200*time.Millisecond).Should(Succeed())
})

var _ = AfterSuite(func() {
	if mgrCancel != nil {
		mgrCancel()
	}
	if envtestCancel != nil {
		envtestCancel()
	}
	By("tearing down the test environment")
	if testEnv != nil {
		Eventually(func() error { return testEnv.Stop() }, time.Minute, time.Second).Should(Succeed())
	}
})

// firstFoundEnvtestBinaryDir locates the envtest binary directory under bin/k8s
// so the suite runs from an IDE without KUBEBUILDER_ASSETS set, mirroring the
// controller suite's helper.
func firstFoundEnvtestBinaryDir() string {
	basePath := filepath.Join("..", "..", "bin", "k8s")
	entries, err := os.ReadDir(basePath)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if entry.IsDir() {
			return filepath.Join(basePath, entry.Name())
		}
	}
	return ""
}
