// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

package main

/*
#include <stdlib.h>

typedef struct {
	char* data;
	int size;
} ByteArray;

typedef struct {
	char* error;
} ErrorResult;

typedef struct {
	char* manifest_json;
	char* error;
} ManifestResult;

typedef struct {
	char* data;
	int size;
	char* error;
} BlobResult;
*/
import "C"

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
)

// getKeychain returns an authn.Keychain that uses Docker config for authentication
// This supports "docker login" authenticated pulls
func getKeychain() authn.Keychain {
	// DefaultKeychain reads credentials from Docker config (~/.docker/config.json)
	// and supports docker credential helpers
	return authn.DefaultKeychain
}

//export PullManifest
func PullManifest(imageRef *C.char) C.ManifestResult {
	var result C.ManifestResult
	
	refStr := C.GoString(imageRef)
	
	// Parse the image reference
	ref, err := name.ParseReference(refStr)
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to parse reference: %v", err))
		return result
	}
	
	// Get the image descriptor with authentication
	desc, err := remote.Get(ref, remote.WithAuthFromKeychain(getKeychain()))
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to get image: %v", err))
		return result
	}
	
	// The manifest is already in desc.Manifest as bytes
	// Just return it as a string
	result.manifest_json = C.CString(string(desc.Manifest))
	result.error = nil
	return result
}

//export PullBlob
func PullBlob(imageRef *C.char, digest *C.char, outputPath *C.char) C.ErrorResult {
	var result C.ErrorResult
	
	refStr := C.GoString(imageRef)
	digestStr := C.GoString(digest)
	outputPathStr := C.GoString(outputPath)
	
	// Parse the image reference
	ref, err := name.ParseReference(refStr)
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to parse reference: %v", err))
		return result
	}
	
	// Parse the digest
	hash, err := name.NewDigest(fmt.Sprintf("%s@%s", ref.Context(), digestStr))
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to parse digest: %v", err))
		return result
	}
	
	// Get the layer with authentication
	layer, err := remote.Layer(hash, remote.WithAuthFromKeychain(getKeychain()))
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to get layer: %v", err))
		return result
	}
	
	// Open the layer for reading
	rc, err := layer.Compressed()
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to open layer: %v", err))
		return result
	}
	defer rc.Close()
	
	// Create output directory if it doesn't exist
	outputDir := filepath.Dir(outputPathStr)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		result.error = C.CString(fmt.Sprintf("failed to create output directory: %v", err))
		return result
	}
	
	// Create output file
	out, err := os.Create(outputPathStr)
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to create output file: %v", err))
		return result
	}
	defer out.Close()
	
	// Copy the blob to the output file
	_, err = io.Copy(out, rc)
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to write blob: %v", err))
		return result
	}
	
	result.error = nil
	return result
}

//export FreeString
func FreeString(str *C.char) {
	C.free(unsafe.Pointer(str))
}

//export TestAuthentication
func TestAuthentication(imageRef *C.char) C.ErrorResult {
	var result C.ErrorResult
	
	refStr := C.GoString(imageRef)
	
	// Parse the image reference
	ref, err := name.ParseReference(refStr)
	if err != nil {
		result.error = C.CString(fmt.Sprintf("failed to parse reference: %v", err))
		return result
	}
	
	// Try to get the image with authentication
	_, err = remote.Get(ref, remote.WithAuthFromKeychain(getKeychain()), remote.WithContext(context.Background()))
	if err != nil {
		result.error = C.CString(fmt.Sprintf("authentication test failed: %v", err))
		return result
	}
	
	result.error = nil
	return result
}

func main() {
	// This is required for CGO but won't be executed
}
