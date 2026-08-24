package redis

import "context"

type ACLCmdable interface {
	ACLDryRun(ctx context.Context, username string, command ...interface{}) *StringCmd

	ACLLog(ctx context.Context, count int64) *ACLLogCmd
	ACLLogReset(ctx context.Context) *StatusCmd

	ACLGenPass(ctx context.Context, bit int) *StringCmd

	ACLSetUser(ctx context.Context, username string, rules ...string) *StatusCmd
	ACLDelUser(ctx context.Context, username string) *IntCmd
	ACLUsers(ctx context.Context) *StringSliceCmd
	ACLWhoAmI(ctx context.Context) *StringCmd
	ACLList(ctx context.Context) *StringSliceCmd

	ACLCat(ctx context.Context) *StringSliceCmd
	ACLCatArgs(ctx context.Context, options *ACLCatArgs) *StringSliceCmd
}

type ACLCatArgs struct {
	Category string
}

func (c cmdable) ACLDryRun(ctx context.Context, username string, command ...interface{}) *StringCmd {
	args := make([]interface{}, 0, 3+len(command))
	args = append(args, "acl", "dryrun", username)
	args = append(args, command...)
	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLLog(ctx context.Context, count int64) *ACLLogCmd {
	args := make([]interface{}, 0, 3)
	args = append(args, "acl", "log")
	if count > 0 {
		args = append(args, count)
	}
	cmd := NewACLLogCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLLogReset(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "acl", "log", "reset")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLDelUser(ctx context.Context, username string) *IntCmd {
	cmd := NewIntCmd(ctx, "acl", "deluser", username)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLSetUser(ctx context.Context, username string, rules ...string) *StatusCmd {
	args := make([]interface{}, 3+len(rules))
	args[0] = "acl"
	args[1] = "setuser"
	args[2] = username
	for i, rule := range rules {
		args[i+3] = rule
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLGenPass(ctx context.Context, bit int) *StringCmd {
	args := make([]interface{}, 0, 3)
	args = append(args, "acl", "genpass")
	if bit > 0 {
		args = append(args, bit)
	}
	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLUsers(ctx context.Context) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "acl", "users")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLWhoAmI(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "acl", "whoami")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLList(ctx context.Context) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "acl", "list")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLCat(ctx context.Context) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "acl", "cat")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ACLCatArgs(ctx context.Context, options *ACLCatArgs) *StringSliceCmd {
	// if there is a category passed, build new cmd, if there isn't - use the ACLCat method
	if options != nil && options.Category != "" {
		cmd := NewStringSliceCmd(ctx, "acl", "cat", options.Category)
		_ = c(ctx, cmd)
		return cmd
	}

	return c.ACLCat(ctx)
}
