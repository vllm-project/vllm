package redis

import (
	"context"
	"errors"
)

type pipelineExecer func(context.Context, []Cmder) error

// Pipeliner is a mechanism to realise Redis Pipeline technique.
//
// Pipelining is a technique to extremely speed up processing by packing
// operations to batches, send them at once to Redis and read a replies in a
// single step.
// See https://redis.io/topics/pipelining
//
// Pay attention, that Pipeline is not a transaction, so you can get unexpected
// results in case of big pipelines and small read/write timeouts.
// Redis client has retransmission logic in case of timeouts, pipeline
// can be retransmitted and commands can be executed more then once.
// To avoid this: it is good idea to use reasonable bigger read/write timeouts
// depends of your batch size and/or use TxPipeline.
type Pipeliner interface {
	StatefulCmdable

	// Len obtains the number of commands in the pipeline that have not yet been executed.
	Len() int

	// Do is an API for executing any command.
	// If a certain Redis command is not yet supported, you can use Do to execute it.
	Do(ctx context.Context, args ...interface{}) *Cmd

	// Process queues the cmd for later execution.
	Process(ctx context.Context, cmd Cmder) error

	// BatchProcess adds multiple commands to be executed into the pipeline buffer.
	BatchProcess(ctx context.Context, cmd ...Cmder) error

	// Discard discards all commands in the pipeline buffer that have not yet been executed.
	Discard()

	// Exec sends all the commands buffered in the pipeline to the redis server.
	Exec(ctx context.Context) ([]Cmder, error)

	// Cmds returns the list of queued commands.
	Cmds() []Cmder
}

var _ Pipeliner = (*Pipeline)(nil)

// Pipeline implements pipelining as described in
// https://redis.io/docs/latest/develop/using-commands/pipelining.
// Please note: it is not safe for concurrent use by multiple goroutines.
type Pipeline struct {
	cmdable
	statefulCmdable

	exec pipelineExecer
	cmds []Cmder
}

func (c *Pipeline) init() {
	c.cmdable = c.Process
	c.statefulCmdable = c.Process
}

// Len returns the number of queued commands.
func (c *Pipeline) Len() int {
	return len(c.cmds)
}

// Do queues the custom command for later execution.
func (c *Pipeline) Do(ctx context.Context, args ...interface{}) *Cmd {
	cmd := NewCmd(ctx, args...)
	if len(args) == 0 {
		cmd.SetErr(errors.New("redis: please enter the command to be executed"))
		return cmd
	}
	_ = c.Process(ctx, cmd)
	return cmd
}

// Process queues the cmd for later execution.
func (c *Pipeline) Process(ctx context.Context, cmd Cmder) error {
	return c.BatchProcess(ctx, cmd)
}

// BatchProcess queues multiple cmds for later execution.
func (c *Pipeline) BatchProcess(ctx context.Context, cmd ...Cmder) error {
	c.cmds = append(c.cmds, cmd...)
	return nil
}

// Discard resets the pipeline and discards queued commands.
func (c *Pipeline) Discard() {
	c.cmds = c.cmds[:0]
}

// Exec executes all previously queued commands using one
// client-server roundtrip.
//
// Exec always returns list of commands and error of the first failed
// command if any.
func (c *Pipeline) Exec(ctx context.Context) ([]Cmder, error) {
	if len(c.cmds) == 0 {
		return nil, nil
	}

	cmds := c.cmds
	c.cmds = nil

	return cmds, c.exec(ctx, cmds)
}

func (c *Pipeline) Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	if err := fn(c); err != nil {
		return nil, err
	}
	return c.Exec(ctx)
}

func (c *Pipeline) Pipeline() Pipeliner {
	return c
}

func (c *Pipeline) TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.Pipelined(ctx, fn)
}

func (c *Pipeline) TxPipeline() Pipeliner {
	return c
}

func (c *Pipeline) Cmds() []Cmder {
	return c.cmds
}
