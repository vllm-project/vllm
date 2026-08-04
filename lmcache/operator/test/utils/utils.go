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
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"

	. "github.com/onsi/ginkgo/v2" // nolint:revive,staticcheck
)

const (
	defaultKindBinary  = "kind"
	defaultKindCluster = "kind"

	// certManagerDefaultVersion is the cert-manager release installed on the
	// target cluster when none is already present. cert-manager is a hard
	// prerequisite for the operator's mutating admission webhook: it mints the
	// serving certificate (config/certmanager) and injects the CA bundle into
	// the MutatingWebhookConfiguration. Override with CERT_MANAGER_VERSION.
	certManagerDefaultVersion = "v1.16.3"
	// certManagerManifestURLTmpl is the upstream static install manifest,
	// parameterised by release tag.
	certManagerManifestURLTmpl = "https://github.com/cert-manager/cert-manager/releases/download/%s/cert-manager.yaml"
)

// certManagerVersion returns the cert-manager release to install, honoring the
// CERT_MANAGER_VERSION env var and falling back to certManagerDefaultVersion.
func certManagerVersion() string {
	if v := os.Getenv("CERT_MANAGER_VERSION"); v != "" {
		return v
	}
	return certManagerDefaultVersion
}

// IsCertManagerCRDsInstalled reports whether cert-manager's CRDs are already
// registered in the target cluster. Used to skip installation (and teardown)
// when running against a cluster that ships cert-manager (e.g. a shared
// OpenShift/EKS cluster), so the suite never uninstalls something it did not
// install.
func IsCertManagerCRDsInstalled() bool {
	cmd := exec.Command("kubectl", "get", "crd", "certificates.cert-manager.io", "--ignore-not-found=true")
	out, err := Run(cmd)
	if err != nil {
		return false
	}
	return strings.Contains(out, "certificates.cert-manager.io")
}

// InstallCertManager applies the upstream cert-manager static manifest and
// blocks until its controller, webhook, and CA-injector Deployments report
// Available. cert-manager must be ready before `make deploy` applies the
// operator's Issuer/Certificate and the CA-injected webhook, otherwise the
// apply is rejected (unknown cert-manager.io kinds) and the serving-cert
// secret the controller-manager mounts never materialises.
func InstallCertManager() error {
	url := fmt.Sprintf(certManagerManifestURLTmpl, certManagerVersion())
	if _, err := Run(exec.Command("kubectl", "apply", "-f", url)); err != nil {
		return fmt.Errorf("failed to apply cert-manager manifest %q: %w", url, err)
	}
	for _, deploy := range []string{"cert-manager", "cert-manager-webhook", "cert-manager-cainjector"} {
		cmd := exec.Command("kubectl", "wait", "deployment/"+deploy,
			"--for=condition=Available",
			"--namespace=cert-manager",
			"--timeout=5m",
		)
		if _, err := Run(cmd); err != nil {
			return fmt.Errorf("cert-manager deployment %q did not become Available: %w", deploy, err)
		}
	}
	return nil
}

// UninstallCertManager removes the cert-manager release installed by
// InstallCertManager. Best-effort: failures are logged to GinkgoWriter rather
// than failing the suite, since teardown runs after the specs have reported.
func UninstallCertManager() {
	url := fmt.Sprintf(certManagerManifestURLTmpl, certManagerVersion())
	if _, err := Run(exec.Command("kubectl", "delete", "-f", url, "--ignore-not-found=true")); err != nil {
		_, _ = fmt.Fprintf(GinkgoWriter, "warning: cert-manager uninstall failed: %v\n", err)
	}
}

// Run executes the provided command within this context
func Run(cmd *exec.Cmd) (string, error) {
	dir, _ := GetProjectDir()
	cmd.Dir = dir

	if err := os.Chdir(cmd.Dir); err != nil {
		_, _ = fmt.Fprintf(GinkgoWriter, "chdir dir: %q\n", err)
	}

	cmd.Env = append(os.Environ(), "GO111MODULE=on")
	command := strings.Join(cmd.Args, " ")
	_, _ = fmt.Fprintf(GinkgoWriter, "running: %q\n", command)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return string(output), fmt.Errorf("%q failed with error %q: %w", command, string(output), err)
	}

	return string(output), nil
}

// LoadImageToKindClusterWithName loads a local docker image to the kind cluster
func LoadImageToKindClusterWithName(name string) error {
	cluster := defaultKindCluster
	if v, ok := os.LookupEnv("KIND_CLUSTER"); ok {
		cluster = v
	}
	kindOptions := []string{"load", "docker-image", name, "--name", cluster}
	kindBinary := defaultKindBinary
	if v, ok := os.LookupEnv("KIND"); ok {
		kindBinary = v
	}
	cmd := exec.Command(kindBinary, kindOptions...)
	_, err := Run(cmd)
	return err
}

// GetNonEmptyLines converts given command output string into individual objects
// according to line breakers, and ignores the empty elements in it.
func GetNonEmptyLines(output string) []string {
	var res []string
	elements := strings.SplitSeq(output, "\n")
	for element := range elements {
		if element != "" {
			res = append(res, element)
		}
	}

	return res
}

// GetProjectDir will return the directory where the project is
func GetProjectDir() (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return wd, fmt.Errorf("failed to get current working directory: %w", err)
	}
	wd = strings.ReplaceAll(wd, "/test/e2e", "")
	return wd, nil
}

// UncommentCode searches for target in the file and remove the comment prefix
// of the target content. The target content may span multiple lines.
func UncommentCode(filename, target, prefix string) error {
	// false positive
	// nolint:gosec
	content, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read file %q: %w", filename, err)
	}
	strContent := string(content)

	idx := strings.Index(strContent, target)
	if idx < 0 {
		return fmt.Errorf("unable to find the code %q to be uncommented", target)
	}

	out := new(bytes.Buffer)
	_, err = out.Write(content[:idx])
	if err != nil {
		return fmt.Errorf("failed to write to output: %w", err)
	}

	scanner := bufio.NewScanner(bytes.NewBufferString(target))
	if !scanner.Scan() {
		return nil
	}
	for {
		if _, err = out.WriteString(strings.TrimPrefix(scanner.Text(), prefix)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
		// Avoid writing a newline in case the previous line was the last in target.
		if !scanner.Scan() {
			break
		}
		if _, err = out.WriteString("\n"); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	}

	if _, err = out.Write(content[idx+len(target):]); err != nil {
		return fmt.Errorf("failed to write to output: %w", err)
	}

	// false positive
	// nolint:gosec
	if err = os.WriteFile(filename, out.Bytes(), 0644); err != nil {
		return fmt.Errorf("failed to write file %q: %w", filename, err)
	}

	return nil
}
