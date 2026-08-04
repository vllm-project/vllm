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
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// operatorRoot resolves to the absolute path of the operator/ directory,
// derived at compile time from this file's location. Unlike GetProjectDir,
// this does not depend on the current working directory and never calls
// os.Chdir, so tests can run from any working directory.
var operatorRoot = func() string {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		// Should never happen — runtime.Caller(0) only fails if the
		// stack has fewer frames than requested.
		panic("utils.operatorRoot: runtime.Caller failed")
	}
	// thisFile = <repo>/operator/test/utils/runner.go
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))
}()

// OperatorRoot returns the absolute path of the operator/ directory.
func OperatorRoot() string {
	return operatorRoot
}

// RunFromOperator runs an exec.Cmd with cmd.Dir set to the operator/ root.
// It does not modify the process working directory. Output is captured
// and returned as a single string; on non-zero exit, the error wraps
// the combined output.
func RunFromOperator(cmd *exec.Cmd) (string, error) {
	cmd.Dir = operatorRoot
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("%s failed: %w\n%s", strings.Join(cmd.Args, " "), err, string(out))
	}
	return string(out), nil
}

// RunMake runs `make <args...>` from the operator/ directory and returns
// the combined output. It is the no-Chdir replacement for the working-
// directory hack in utils.Run.
func RunMake(args ...string) (string, error) {
	return RunFromOperator(exec.Command("make", args...))
}
