package redis

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"errors"
	"io"
	"sync"
)

type Scripter interface {
	Eval(ctx context.Context, script string, keys []string, args ...interface{}) *Cmd
	EvalSha(ctx context.Context, sha1 string, keys []string, args ...interface{}) *Cmd
	EvalRO(ctx context.Context, script string, keys []string, args ...interface{}) *Cmd
	EvalShaRO(ctx context.Context, sha1 string, keys []string, args ...interface{}) *Cmd
	ScriptExists(ctx context.Context, hashes ...string) *BoolSliceCmd
	ScriptLoad(ctx context.Context, script string) *StringCmd
}

var (
	_ Scripter = (*Client)(nil)
	_ Scripter = (*Ring)(nil)
	_ Scripter = (*ClusterClient)(nil)
)

type Script struct {
	src       string
	mu        sync.RWMutex
	hash      string
	serverSHA bool // if true: do not compute SHA-1 in Go; load digest from Redis (SCRIPT LOAD)
}

func NewScript(src string) *Script {
	h := sha1.New()
	_, _ = io.WriteString(h, src)

	return &Script{
		src:       src,
		hash:      hex.EncodeToString(h.Sum(nil)),
		serverSHA: false,
	}
}

// NewScriptServerSHA creates a Script that avoids computing SHA-1 in Go.
// The digest is obtained from Redis via SCRIPT LOAD (server-side hashing),
// then EVALSHA/EVALSHA_RO is used.
func NewScriptServerSHA(src string) *Script {
	return &Script{
		src:       src,
		serverSHA: true,
	}
}

func (s *Script) Hash() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.hash
}

func (s *Script) Load(ctx context.Context, c Scripter) *StringCmd {
	cmd := c.ScriptLoad(ctx, s.src)
	if err := cmd.Err(); err == nil {
		s.mu.Lock()
		s.hash = cmd.Val()
		s.mu.Unlock()
	}
	return cmd
}

func (s *Script) Exists(ctx context.Context, c Scripter) *BoolSliceCmd {
	s.mu.RLock()
	hash := s.hash
	serverSHA := s.serverSHA
	s.mu.RUnlock()
	if hash == "" && serverSHA {
		// For server-side scripts, obtain digest from Redis first.
		// If hash is empty, it means SCRIPT LOAD was not called yet, so we check existence of empty hash which will return false.
		// This avoids unnecessary SCRIPT LOAD just to check existence.
		if err := s.ensureHash(ctx, c); err != nil {
			return c.ScriptExists(ctx, "")
		}
		s.mu.RLock()
		hash = s.hash
		s.mu.RUnlock()
	}
	if hash == "" {
		return c.ScriptExists(ctx, "")
	}
	return c.ScriptExists(ctx, hash)
}

func (s *Script) Eval(ctx context.Context, c Scripter, keys []string, args ...interface{}) *Cmd {
	return c.Eval(ctx, s.src, keys, args...)
}

func (s *Script) EvalRO(ctx context.Context, c Scripter, keys []string, args ...interface{}) *Cmd {
	return c.EvalRO(ctx, s.src, keys, args...)
}

// ensureHash ensures that s.hash is populated by using SCRIPT LOAD.
// It never calls SHA-1 in Go; Redis computes and returns the digest.
func (s *Script) ensureHash(ctx context.Context, c Scripter) error {
	// Fast path: read lock, return if hash is already set.
	s.mu.RLock()
	if s.hash != "" {
		s.mu.RUnlock()
		return nil
	}
	s.mu.RUnlock()

	// Slow path: acquire write lock and load.
	s.mu.Lock()
	if s.hash != "" {
		s.mu.Unlock()
		return nil
	}
	cmd := c.ScriptLoad(ctx, s.src)
	if err := cmd.Err(); err != nil {
		s.mu.Unlock()
		return err
	}
	s.hash = cmd.Val()
	s.mu.Unlock()
	return nil
}

func (s *Script) EvalSha(ctx context.Context, c Scripter, keys []string, args ...interface{}) *Cmd {
	// Default behavior: use client-side SHA-1 computed in NewScript.
	if !s.serverSHA {
		s.mu.RLock()
		hash := s.hash
		s.mu.RUnlock()
		return c.EvalSha(ctx, hash, keys, args...)
	}

	// Server-side SHA via SCRIPT LOAD + EVALSHA.
	if err := s.ensureHash(ctx, c); err != nil {
		return s.Eval(ctx, c, keys, args...)
	}

	s.mu.RLock()
	hash := s.hash
	s.mu.RUnlock()

	r := c.EvalSha(ctx, hash, keys, args...)
	if HasErrorPrefix(r.Err(), "NOSCRIPT") {
		// Script cache was flushed; reload and retry once.
		if err := s.ensureHash(ctx, c); err != nil {
			return s.Eval(ctx, c, keys, args...)
		}
		s.mu.RLock()
		hash = s.hash
		s.mu.RUnlock()
		return c.EvalSha(ctx, hash, keys, args...)
	}

	return r
}

func (s *Script) EvalShaRO(ctx context.Context, c Scripter, keys []string, args ...interface{}) *Cmd {
	if !s.serverSHA {
		s.mu.RLock()
		hash := s.hash
		s.mu.RUnlock()
		return c.EvalShaRO(ctx, hash, keys, args...)
	}

	if err := s.ensureHash(ctx, c); err != nil {
		return s.EvalRO(ctx, c, keys, args...)
	}

	s.mu.RLock()
	hash := s.hash
	s.mu.RUnlock()

	r := c.EvalShaRO(ctx, hash, keys, args...)
	if HasErrorPrefix(r.Err(), "NOSCRIPT") {
		if err := s.ensureHash(ctx, c); err != nil {
			return s.EvalRO(ctx, c, keys, args...)
		}
		s.mu.RLock()
		hash = s.hash
		s.mu.RUnlock()
		return c.EvalShaRO(ctx, hash, keys, args...)
	}

	return r
}

// Run optimistically uses EVALSHA to run the script. If script does not exist
// it is retried using EVAL.
func (s *Script) Run(ctx context.Context, c Scripter, keys []string, args ...interface{}) *Cmd {
	r := s.EvalSha(ctx, c, keys, args...)
	if isNoScriptErr(r.Err()) {
		return s.Eval(ctx, c, keys, args...)
	}
	return r
}

// RunRO optimistically uses EVALSHA_RO to run the script. If script does not exist
// it is retried using EVAL_RO.
func (s *Script) RunRO(ctx context.Context, c Scripter, keys []string, args ...interface{}) *Cmd {
	r := s.EvalShaRO(ctx, c, keys, args...)
	if isNoScriptErr(r.Err()) {
		return s.EvalRO(ctx, c, keys, args...)
	}
	return r
}

// isNoScriptErr reports whether err means "this digest is not cached", whether
// it arrived already normalized to ErrNoScript or as the server's raw NOSCRIPT
// error. Both are accepted because the Eval wrappers only normalize when the
// result is readable without blocking — on the deferred autopipeline face the
// raw error reaches here untouched (see cmdable.eval).
func isNoScriptErr(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, ErrNoScript) || HasErrorPrefix(err, "NOSCRIPT")
}
