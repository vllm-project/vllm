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

package resources

import (
	"encoding/json"
	"fmt"
	"maps"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// LookupServiceName returns the name of the node-local lookup Service for discovery.
func LookupServiceName(engineName string) string {
	return engineName
}

// ConnectionConfigMapName returns the name of the <engine>-connection ConfigMap.
func ConnectionConfigMapName(engineName string) string {
	return fmt.Sprintf("%s-connection", engineName)
}

// BuildConnectionConfigMap creates the <name>-connection ConfigMap with kv-transfer-config JSON.
func BuildConnectionConfigMap(engine *lmcachev1alpha1.LMCacheEngine) *corev1.ConfigMap {
	port := derefInt32(getServerPort(&engine.Spec), 5555)

	return buildConnectionConfigMapCore(
		engine.Name,
		engine.Namespace,
		"LMCacheMPConnector",
		"lmcache.integration.vllm.lmcache_mp_connector",
		port,
		nil,
	)
}

// buildConnectionConfigMapCore is the shared core for the <engine>-connection
// ConfigMap that both engine controllers emit. It produces the kv-transfer-config
// JSON with the node-local Service host/port and lets the caller select the
// connector name, its module path, and any connector-specific extra config keys
// (e.g. CacheBlend's cb.check_layer / cb.recomp_ratio).
//
// Parameters:
//   - name, namespace: the owning engine's identity (drives the ConfigMap name,
//     labels, and the node-local Service DNS host).
//   - connectorName: the kv_connector value (e.g. "LMCacheMPConnector" or
//     "CBKVConnector").
//   - modulePath: the kv_connector_module_path value.
//   - port: the engine server port, emitted as lmcache.mp.port (string).
//   - extraConfig: additional kv_connector_extra_config keys merged on top of the
//     base lmcache.mp.host / lmcache.mp.port entries; nil for the default
//     connector.
func buildConnectionConfigMapCore(
	name, namespace, connectorName, modulePath string,
	port int32,
	extraConfig map[string]any,
) *corev1.ConfigMap {
	svcHost := fmt.Sprintf("%s.%s.svc.cluster.local", LookupServiceName(name), namespace)

	extra := map[string]any{
		"lmcache.mp.host": fmt.Sprintf("tcp://%s", svcHost),
		"lmcache.mp.port": fmt.Sprintf("%d", port),
	}
	maps.Copy(extra, extraConfig)

	config := map[string]any{
		"kv_connector":              connectorName,
		"kv_connector_module_path":  modulePath,
		"kv_role":                   "kv_both",
		"kv_connector_extra_config": extra,
	}

	configJSON, _ := json.MarshalIndent(config, "", "  ")

	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ConnectionConfigMapName(name),
			Namespace: namespace,
			Labels:    StandardLabels(name),
		},
		Data: map[string]string{
			"kv-transfer-config.json": string(configJSON),
		},
	}
}
