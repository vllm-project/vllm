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
	"encoding/base64"
	"encoding/json"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// CreateDockerConfigJSONSecret creates a kubernetes.io/dockerconfigjson pull
// Secret in the given namespace, equivalent to
//
//	kubectl create secret docker-registry <name> \
//	  --docker-server=<server> --docker-username=<username> --docker-password=<password>
//
// It is used by the GPU smoke tier to pull the PRIVATE cacheblend-plugin
// payload image: the credentials arrive as env vars (e.g. from the Buildkite
// pipeline secret store) and this Secret is referenced by the engine's
// injection.imagePullSecrets, which the mutating webhook appends to the vLLM
// pod that runs the payload init container.
//
// The Secret is recreated (delete-then-create) if one of the same name already
// exists so reruns against a reused namespace pick up rotated credentials. The
// password is never logged.
func CreateDockerConfigJSONSecret(
	ctx context.Context,
	c client.Client,
	namespace, name, server, username, password string,
) error {
	if server == "" || username == "" || password == "" {
		return fmt.Errorf(
			"docker-registry secret %s/%s: server, username and password are all required", namespace, name)
	}

	// .dockerconfigjson schema: {"auths":{"<server>":{username,password,auth}}}.
	// "auth" is base64(username:password) — Docker reads this field; username
	// and password are kept too for tooling that inspects the entry.
	auth := base64.StdEncoding.EncodeToString([]byte(username + ":" + password))
	dockerCfg := map[string]any{
		"auths": map[string]any{
			server: map[string]any{
				"username": username,
				"password": password,
				"auth":     auth,
			},
		},
	}
	raw, err := json.Marshal(dockerCfg)
	if err != nil {
		return fmt.Errorf("marshal dockerconfigjson for %s/%s: %w", namespace, name, err)
	}

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Type:       corev1.SecretTypeDockerConfigJson,
		Data:       map[string][]byte{corev1.DockerConfigJsonKey: raw},
	}

	err = c.Create(ctx, secret)
	if apierrors.IsAlreadyExists(err) {
		existing := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace}}
		if delErr := c.Delete(ctx, existing); delErr != nil && !apierrors.IsNotFound(delErr) {
			return fmt.Errorf("replace docker-registry secret %s/%s: delete: %w", namespace, name, delErr)
		}
		err = c.Create(ctx, secret)
	}
	if err != nil {
		return fmt.Errorf("create docker-registry secret %s/%s: %w", namespace, name, err)
	}
	return nil
}
