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
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"github.com/LMCache/LMCache/test/utils"
)

// Cross-namespace authSecretRef.
//
// The operator mirrors a Secret referenced by spec.l2Backend.resp.
// authSecretRef into the LMCacheEngine's namespace, then injects
// LMCACHE_RESP_USERNAME / LMCACHE_RESP_PASSWORD into the DaemonSet
// container as env vars (NOT as CLI args). This spec asserts:
//
//  1. A managed copy of the source Secret appears in the engine's
//     namespace at <engine-name>-resp-auth.
//  2. The pod template references the local copy via env-var
//     SecretKeyRef — never via container args.
//  3. The literal Secret values never leak into container args
//     (i.e. they wouldn't show up in `kubectl describe pod`).
var _ = Describe("LMCacheEngine cross-namespace auth smoke (no-GPU)", Ordered, func() {
	var (
		ctx          context.Context
		nsName       string
		sourceNSName string
		expectedUser = "tm-test-user"
		expectedPass = "tm-test-password" // pragma: allowlist secret
	)

	BeforeEach(func() {
		ctx = context.Background()
		nsName = createTestNamespace(ctx)
		// Source namespace lives separately so we test true cross-NS
		// mirroring rather than same-NS direct reference.
		sourceNSName = createTestNamespace(ctx)
	})

	AfterEach(func() {
		recordOnFailure(nsName)
	})

	It("mirrors a cross-namespace authSecretRef and uses env-var injection", func() {
		By("creating the source Secret in the upstream namespace")
		src := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "redis-auth",
				Namespace: sourceNSName,
			},
			Data: map[string][]byte{
				"username": []byte(expectedUser),
				"password": []byte(expectedPass),
			},
		}
		Expect(k8sClient.Create(ctx, src)).To(Succeed())

		By("loading the fixture and pointing it at the source namespace")
		lmc, err := utils.NewLMCFromFixture("lmc_with_redis_l2_authsecret.yaml", nsName, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(lmc.Spec.L2Backend).NotTo(BeNil())
		Expect(lmc.Spec.L2Backend.RESP).NotTo(BeNil())
		Expect(lmc.Spec.L2Backend.RESP.AuthSecretRef).NotTo(BeNil())
		lmc.Spec.L2Backend.RESP.AuthSecretRef.Namespace = sourceNSName
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		managedSecretName := lmc.Name + "-resp-auth"

		By("eventually a managed copy of the Secret appears in the engine namespace")
		Eventually(func(g Gomega) {
			got := &corev1.Secret{}
			g.Expect(k8sClient.Get(ctx,
				types.NamespacedName{Namespace: nsName, Name: managedSecretName},
				got,
			)).To(Succeed())
			g.Expect(got.Data).To(HaveKeyWithValue("password", []byte(expectedPass)))
			g.Expect(got.Data).To(HaveKeyWithValue("username", []byte(expectedUser)))
		}, 30*time.Second, time.Second).Should(Succeed())

		By("checking the DaemonSet wires creds through env-var SecretKeyRef")
		ds := &appsv1.DaemonSet{}
		Expect(k8sClient.Get(ctx, types.NamespacedName(key), ds)).To(Succeed())
		Expect(ds.Spec.Template.Spec.Containers).To(HaveLen(1))
		c := ds.Spec.Template.Spec.Containers[0]

		passwordEnv := findEnvVar(c.Env, "LMCACHE_RESP_PASSWORD")
		Expect(passwordEnv).NotTo(BeNil(), "expected LMCACHE_RESP_PASSWORD env var")
		Expect(passwordEnv.Value).To(BeEmpty(),
			"LMCACHE_RESP_PASSWORD must be sourced from a Secret, not a literal value")
		Expect(passwordEnv.ValueFrom).NotTo(BeNil())
		Expect(passwordEnv.ValueFrom.SecretKeyRef).NotTo(BeNil())
		Expect(passwordEnv.ValueFrom.SecretKeyRef.Name).To(Equal(managedSecretName),
			"password env var must reference the local managed Secret, not the source namespace")
		Expect(passwordEnv.ValueFrom.SecretKeyRef.Key).To(Equal("password"))

		usernameEnv := findEnvVar(c.Env, "LMCACHE_RESP_USERNAME")
		Expect(usernameEnv).NotTo(BeNil(), "expected LMCACHE_RESP_USERNAME env var")
		Expect(usernameEnv.ValueFrom).NotTo(BeNil())
		Expect(usernameEnv.ValueFrom.SecretKeyRef).NotTo(BeNil())
		Expect(usernameEnv.ValueFrom.SecretKeyRef.Name).To(Equal(managedSecretName))

		By("ensuring the literal Secret values do NOT appear in container args")
		// "kubectl describe pod" prints the args verbatim; if the
		// operator inlined credentials the user/pass would leak there.
		// Env-var injection (the contract) keeps them out of the args.
		joinedArgs := strings.Join(c.Args, " ")
		Expect(joinedArgs).NotTo(ContainSubstring(expectedUser))
		Expect(joinedArgs).NotTo(ContainSubstring(expectedPass))
	})
})

// findEnvVar returns a pointer to the named env var (so callers can
// inspect ValueFrom) or nil if absent. We return *corev1.EnvVar rather
// than the value type so a missing entry is unambiguously nil instead
// of the zero struct, which would be indistinguishable from an empty
// env var Value="".
func findEnvVar(envs []corev1.EnvVar, name string) *corev1.EnvVar {
	for i := range envs {
		if envs[i].Name == name {
			return &envs[i]
		}
	}
	return nil
}
