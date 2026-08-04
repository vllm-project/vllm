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
	"embed"
	"fmt"
	"path"

	"sigs.k8s.io/yaml"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// fixturesFS embeds the fixtures tree shipped with the smoke harness:
// CR YAMLs at fixtures/*.yaml and golden JSON snapshots at
// fixtures/golden/*.json. Embedding sidesteps the working-directory
// hack in utils.GetProjectDir: fixtures are compiled into the test
// binary and resolved by relative path.
//
//go:embed fixtures/*.yaml fixtures/golden/*.json
var fixturesFS embed.FS

// LoadFixture returns the raw bytes of a fixture file by name (no path).
func LoadFixture(name string) ([]byte, error) {
	data, err := fixturesFS.ReadFile(path.Join("fixtures", name))
	if err != nil {
		return nil, fmt.Errorf("load fixture %q: %w", name, err)
	}
	return data, nil
}

// LoadGolden returns the raw bytes of a golden snapshot by name (no path).
// Golden files live under fixtures/golden/ and represent expected JSON
// payloads emitted by the operator. Tests typically substitute namespace
// placeholders into the result before comparing.
func LoadGolden(name string) ([]byte, error) {
	data, err := fixturesFS.ReadFile(path.Join("fixtures", "golden", name))
	if err != nil {
		return nil, fmt.Errorf("load golden %q: %w", name, err)
	}
	return data, nil
}

// NewLMCFromFixture loads a fixture YAML, decodes it as an LMCacheEngine,
// and overrides metadata.name and metadata.namespace with the supplied
// values. Callers can mutate the returned object further before applying.
func NewLMCFromFixture(fixtureName, namespace, name string) (*lmcachev1alpha1.LMCacheEngine, error) {
	data, err := LoadFixture(fixtureName)
	if err != nil {
		return nil, err
	}
	lmc := &lmcachev1alpha1.LMCacheEngine{}
	if err := yaml.Unmarshal(data, lmc); err != nil {
		return nil, fmt.Errorf("decode fixture %q: %w", fixtureName, err)
	}
	if name != "" {
		lmc.Name = name
	}
	if namespace != "" {
		lmc.Namespace = namespace
	}
	// Strip any server-side metadata that may have leaked through.
	lmc.ResourceVersion = ""
	lmc.UID = ""
	lmc.Generation = 0
	return lmc, nil
}
