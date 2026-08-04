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
	"os"
	"os/exec"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/test/utils"
)

var (
	// managerImage is the manager image used by build / load / deploy.
	// Honors the IMG env var so test-e2e-cluster can point at an image
	// already pushed to a registry the target cluster pulls from.
	managerImage = envDefault("IMG", "example.com/operator:v0.0.1")
	// skipImageLoad disables the Kind-only docker-build + kind-load
	// steps. Set SMOKE_SKIP_IMAGE_LOAD=true when running against an
	// existing cluster (OpenShift, EKS, k3s, etc.) where the image has
	// already been pushed to a reachable registry.
	skipImageLoad = os.Getenv("SMOKE_SKIP_IMAGE_LOAD") == "true"
	// k8sClient is the typed controller-runtime client used by smoke specs.
	// Initialised once in BeforeSuite after CRDs are installed; spec files
	// read it directly without rebuilding their own client.
	k8sClient client.Client
	// certManagerInstalledBySuite records whether BeforeSuite installed
	// cert-manager (vs. finding it pre-installed). AfterSuite only uninstalls
	// what the suite installed, so a shared cluster's cert-manager is preserved.
	certManagerInstalledBySuite bool
)

// envDefault returns os.Getenv(key) or def if the env var is unset/empty.
// Pulled out as a helper so package-level var initialisers stay readable.
func envDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// TestE2E runs the e2e test suite to validate the solution in an isolated environment.
// Requires Kind (or an existing cluster reachable via the current kubeconfig).
func TestE2E(t *testing.T) {
	RegisterFailHandler(Fail)
	_, _ = fmt.Fprintf(GinkgoWriter, "Starting operator e2e test suite\n")
	RunSpecs(t, "e2e suite")
}

var _ = BeforeSuite(func() {
	_, _ = fmt.Fprintf(GinkgoWriter, "manager image: %s (skipImageLoad=%v)\n",
		managerImage, skipImageLoad)

	// cert-manager must exist before `make deploy` applies the operator's
	// Issuer/Certificate and the CA-injected mutating webhook. Install it up
	// front (before the slow image build/load) so its webhook endpoints are
	// warm by the time we deploy. Skip when the cluster already ships it.
	By("ensuring cert-manager is installed (required by the operator's webhook serving cert)")
	if utils.IsCertManagerCRDsInstalled() {
		By("cert-manager CRDs already present; skipping install")
	} else {
		Expect(utils.InstallCertManager()).To(Succeed(), "Failed to install cert-manager")
		certManagerInstalledBySuite = true
	}

	if skipImageLoad {
		// Existing-cluster path: the user pushed the image to a
		// registry before invoking the suite. We can't sideload, and
		// we don't rebuild because rebuilds wouldn't propagate to the
		// pushed copy anyway. The deploy step below uses managerImage
		// as the registry URL.
		By("skipping docker-build + kind-load (SMOKE_SKIP_IMAGE_LOAD=true)")
	} else {
		By("building the manager image")
		_, err := utils.RunMake("docker-build", fmt.Sprintf("IMG=%s", managerImage))
		Expect(err).NotTo(HaveOccurred(), "Failed to build the manager image")

		By("loading the manager image on Kind")
		err = utils.LoadImageToKindClusterWithName(managerImage)
		Expect(err).NotTo(HaveOccurred(), "Failed to load the manager image into Kind")
	}

	By("installing CRDs")
	_, err := utils.RunMake("install")
	Expect(err).NotTo(HaveOccurred(), "Failed to install CRDs")

	By("deploying the controller-manager")
	_, err = utils.RunMake("deploy", fmt.Sprintf("IMG=%s", managerImage))
	Expect(err).NotTo(HaveOccurred(), "Failed to deploy the controller-manager")

	By("labeling the operator namespace with the restricted Pod Security profile")
	labelCmd := exec.Command("kubectl", "label", "--overwrite", "ns",
		"lmcache-operator-system",
		"pod-security.kubernetes.io/enforce=restricted",
	)
	_, err = labelCmd.CombinedOutput()
	Expect(err).NotTo(HaveOccurred(), "Failed to label operator namespace")

	By("registering custom types in the scheme")
	Expect(lmcachev1alpha1.AddToScheme(scheme.Scheme)).To(Succeed())
	Expect(monitoringv1.AddToScheme(scheme.Scheme)).To(Succeed())

	By("building the typed Kubernetes client")
	cfg, err := ctrl.GetConfig()
	Expect(err).NotTo(HaveOccurred(), "Failed to load kubeconfig")
	k8sClient, err = client.New(cfg, client.Options{Scheme: scheme.Scheme})
	Expect(err).NotTo(HaveOccurred(), "Failed to construct typed client")

	By("waiting for the controller-manager Deployment to become Available")
	Expect(waitDeploymentAvailable(context.Background(),
		"lmcache-operator-system",
		"lmcache-operator-controller-manager",
		3*time.Minute,
	)).To(Succeed(), "Controller-manager Deployment did not become Available")
})

var _ = AfterSuite(func() {
	By("undeploying the controller-manager")
	if _, err := utils.RunMake("undeploy", "ignore-not-found=true"); err != nil {
		_, _ = fmt.Fprintf(GinkgoWriter, "warning: undeploy failed: %v\n", err)
	}

	By("uninstalling CRDs")
	if _, err := utils.RunMake("uninstall", "ignore-not-found=true"); err != nil {
		_, _ = fmt.Fprintf(GinkgoWriter, "warning: uninstall failed: %v\n", err)
	}

	if certManagerInstalledBySuite {
		By("uninstalling cert-manager")
		utils.UninstallCertManager()
	}
})

// waitDeploymentAvailable polls a Deployment's status until the
// Available condition is True, or until ctx is cancelled / timeout
// elapses. Uses kubectl rather than the typed client because the typed
// client only knows the schema we registered, not Deployments.
func waitDeploymentAvailable(ctx context.Context, namespace, name string, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		// kubectl exits 0 with empty output if the condition isn't met
		// yet, and 0 with "True" when it is — so we run it and inspect
		// stdout rather than relying on exit codes.
		cmd := exec.CommandContext(ctx, "kubectl", "get", "deployment", name,
			"-n", namespace,
			"-o", "jsonpath={.status.conditions[?(@.type=='Available')].status}")
		out, err := cmd.Output()
		if err != nil {
			return false, nil
		}
		return string(out) == "True", nil
	})
}
