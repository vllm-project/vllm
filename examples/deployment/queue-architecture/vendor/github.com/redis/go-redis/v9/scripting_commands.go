package redis

import "context"

type ScriptingFunctionsCmdable interface {
	Eval(ctx context.Context, script string, keys []string, args ...interface{}) *Cmd
	EvalSha(ctx context.Context, sha1 string, keys []string, args ...interface{}) *Cmd
	EvalRO(ctx context.Context, script string, keys []string, args ...interface{}) *Cmd
	EvalShaRO(ctx context.Context, sha1 string, keys []string, args ...interface{}) *Cmd
	ScriptExists(ctx context.Context, hashes ...string) *BoolSliceCmd
	ScriptFlush(ctx context.Context) *StatusCmd
	ScriptKill(ctx context.Context) *StatusCmd
	ScriptLoad(ctx context.Context, script string) *StringCmd

	FunctionLoad(ctx context.Context, code string) *StringCmd
	FunctionLoadReplace(ctx context.Context, code string) *StringCmd
	FunctionDelete(ctx context.Context, libName string) *StringCmd
	FunctionFlush(ctx context.Context) *StringCmd
	FunctionKill(ctx context.Context) *StringCmd
	FunctionFlushAsync(ctx context.Context) *StringCmd
	FunctionList(ctx context.Context, q FunctionListQuery) *FunctionListCmd
	FunctionDump(ctx context.Context) *StringCmd
	FunctionRestore(ctx context.Context, libDump string) *StringCmd
	FunctionStats(ctx context.Context) *FunctionStatsCmd
	FCall(ctx context.Context, function string, keys []string, args ...interface{}) *Cmd
	FCallRo(ctx context.Context, function string, keys []string, args ...interface{}) *Cmd
	FCallRO(ctx context.Context, function string, keys []string, args ...interface{}) *Cmd
}

func (c cmdable) Eval(ctx context.Context, script string, keys []string, args ...interface{}) *Cmd {
	return c.eval(ctx, "eval", script, keys, args...)
}

func (c cmdable) EvalRO(ctx context.Context, script string, keys []string, args ...interface{}) *Cmd {
	return c.eval(ctx, "eval_ro", script, keys, args...)
}

func (c cmdable) EvalSha(ctx context.Context, sha1 string, keys []string, args ...interface{}) *Cmd {
	return c.eval(ctx, "evalsha", sha1, keys, args...)
}

func (c cmdable) EvalShaRO(ctx context.Context, sha1 string, keys []string, args ...interface{}) *Cmd {
	return c.eval(ctx, "evalsha_ro", sha1, keys, args...)
}

func (c cmdable) eval(ctx context.Context, name, payload string, keys []string, args ...interface{}) *Cmd {
	cmdArgs := make([]interface{}, 3+len(keys), 3+len(keys)+len(args))
	cmdArgs[0] = name
	cmdArgs[1] = payload
	cmdArgs[2] = len(keys)
	for i, key := range keys {
		cmdArgs[3+i] = key
	}
	cmdArgs = appendArgs(cmdArgs, args)
	cmd := NewCmd(ctx, cmdArgs...)

	// it is possible that only args exist without a key.
	// rdb.eval(ctx, eval, script, nil, arg1, arg2)
	if len(keys) > 0 {
		cmd.SetFirstKeyPos(3)
	}
	_ = c(ctx, cmd)
	// Normalize NOSCRIPT to ErrNoScript for Script.Run/RunRO's EVAL fallback,
	// but only when the result is already readable: on the deferred
	// autopipeline face the call above merely enqueues, and reading the outcome
	// here would await execution — making the whole Eval family synchronous on
	// a face whose contract is to return immediately (review finding by codex
	// on #3942). When the result is still pending the normalization is skipped;
	// Script.Run/RunRO also match the raw NOSCRIPT prefix, so the fallback
	// keeps working on that face.
	if cmd.resultReady() {
		if err := cmd.rawErr(); err != nil && HasErrorPrefix(err, "NOSCRIPT") {
			cmd.SetErr(ErrNoScript)
		}
	}
	return cmd
}

func (c cmdable) ScriptExists(ctx context.Context, hashes ...string) *BoolSliceCmd {
	args := make([]interface{}, 2+len(hashes))
	args[0] = "script"
	args[1] = "exists"
	for i, hash := range hashes {
		args[2+i] = hash
	}
	cmd := NewBoolSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ScriptFlush(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "script", "flush")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ScriptKill(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "script", "kill")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ScriptLoad(ctx context.Context, script string) *StringCmd {
	cmd := NewStringCmd(ctx, "script", "load", script)
	_ = c(ctx, cmd)
	return cmd
}

// ------------------------------------------------------------------------------

// FunctionListQuery is used with FunctionList to query for Redis libraries
//
//	  	LibraryNamePattern 	- Use an empty string to get all libraries.
//	  						- Use a glob-style pattern to match multiple libraries with a matching name
//	  						- Use a library's full name to match a single library
//		WithCode			- If true, it will return the code of the library
type FunctionListQuery struct {
	LibraryNamePattern string
	WithCode           bool
}

func (c cmdable) FunctionLoad(ctx context.Context, code string) *StringCmd {
	cmd := NewStringCmd(ctx, "function", "load", code)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionLoadReplace(ctx context.Context, code string) *StringCmd {
	cmd := NewStringCmd(ctx, "function", "load", "replace", code)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionDelete(ctx context.Context, libName string) *StringCmd {
	cmd := NewStringCmd(ctx, "function", "delete", libName)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionFlush(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "function", "flush")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionKill(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "function", "kill")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionFlushAsync(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "function", "flush", "async")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionList(ctx context.Context, q FunctionListQuery) *FunctionListCmd {
	args := make([]interface{}, 2, 5)
	args[0] = "function"
	args[1] = "list"
	if q.LibraryNamePattern != "" {
		args = append(args, "libraryname", q.LibraryNamePattern)
	}
	if q.WithCode {
		args = append(args, "withcode")
	}
	cmd := NewFunctionListCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionDump(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "function", "dump")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionRestore(ctx context.Context, libDump string) *StringCmd {
	cmd := NewStringCmd(ctx, "function", "restore", libDump)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FunctionStats(ctx context.Context) *FunctionStatsCmd {
	cmd := NewFunctionStatsCmd(ctx, "function", "stats")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FCall(ctx context.Context, function string, keys []string, args ...interface{}) *Cmd {
	cmdArgs := fcallArgs("fcall", function, keys, args...)
	cmd := NewCmd(ctx, cmdArgs...)
	if len(keys) > 0 {
		cmd.SetFirstKeyPos(3)
	}
	_ = c(ctx, cmd)
	return cmd
}

// FCallRo this function simply calls FCallRO,
// Deprecated: to maintain convention FCallRO.
func (c cmdable) FCallRo(ctx context.Context, function string, keys []string, args ...interface{}) *Cmd {
	return c.FCallRO(ctx, function, keys, args...)
}

func (c cmdable) FCallRO(ctx context.Context, function string, keys []string, args ...interface{}) *Cmd {
	cmdArgs := fcallArgs("fcall_ro", function, keys, args...)
	cmd := NewCmd(ctx, cmdArgs...)
	if len(keys) > 0 {
		cmd.SetFirstKeyPos(3)
	}
	_ = c(ctx, cmd)
	return cmd
}

func fcallArgs(command string, function string, keys []string, args ...interface{}) []interface{} {
	cmdArgs := make([]interface{}, 3+len(keys), 3+len(keys)+len(args))
	cmdArgs[0] = command
	cmdArgs[1] = function
	cmdArgs[2] = len(keys)
	for i, key := range keys {
		cmdArgs[3+i] = key
	}

	cmdArgs = append(cmdArgs, args...)
	return cmdArgs
}
