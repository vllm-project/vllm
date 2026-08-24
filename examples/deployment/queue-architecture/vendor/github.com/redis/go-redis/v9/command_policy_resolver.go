package redis

import (
	"context"
	"strings"

	"github.com/redis/go-redis/v9/internal/routing"
)

type (
	module      = string
	commandName = string
)

var defaultPolicies = map[module]map[commandName]*routing.CommandPolicy{
	"ft": {
		"create": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"search": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"aggregate": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"dictadd": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"dictdump": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"dictdel": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"suglen": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultHashSlot,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"cursor": {
			Request:  routing.ReqSpecial,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"sugadd": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultHashSlot,
		},
		"sugget": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultHashSlot,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"sugdel": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultHashSlot,
		},
		"spellcheck": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"explain": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"explaincli": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"aliasadd": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"aliasupdate": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"aliasdel": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"aliaslist": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"info": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"tagvals": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"syndump": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"synupdate": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"profile": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
			Tips: map[string]string{
				routing.ReadOnlyCMD: "",
			},
		},
		"alter": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"dropindex": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
		"drop": {
			Request:  routing.ReqDefault,
			Response: routing.RespDefaultKeyless,
		},
	},
}

// defaultPolicyKeyless reports whether name (e.g. "ft.aliaslist") is registered
// in the static policy table as a plain keyless command: default request
// routing with a keyless response policy. Commands whose slot comes from a key
// (RespDefaultHashSlot, e.g. ft.suglen) or with special request routing
// (ReqSpecial, e.g. ft.cursor) are excluded — their key position must still be
// resolved. cmdFirstKeyPosWithInfo consults this so the initial slot
// computation on a cold command-info cache matches the policy the router
// applies once the command reaches routeAndRun.
func defaultPolicyKeyless(name string) bool {
	i := strings.IndexByte(name, '.')
	if i < 0 {
		return false
	}
	policy, ok := defaultPolicies[name[:i]][name[i+1:]]
	if !ok {
		return false
	}
	return policy.Request == routing.ReqDefault && policy.Response == routing.RespDefaultKeyless
}

type CommandInfoResolveFunc func(ctx context.Context, cmd Cmder) *routing.CommandPolicy

type commandInfoResolver struct {
	resolveFunc      CommandInfoResolveFunc
	fallBackResolver *commandInfoResolver
}

func NewCommandInfoResolver(resolveFunc CommandInfoResolveFunc) *commandInfoResolver {
	return &commandInfoResolver{
		resolveFunc: resolveFunc,
	}
}

func NewDefaultCommandPolicyResolver() *commandInfoResolver {
	return NewCommandInfoResolver(func(ctx context.Context, cmd Cmder) *routing.CommandPolicy {
		module := "core"
		command := cmd.Name()
		cmdParts := strings.Split(command, ".")
		if len(cmdParts) == 2 {
			module = cmdParts[0]
			command = cmdParts[1]
		}

		if policy, ok := defaultPolicies[module][command]; ok {
			return policy
		}

		return nil
	})
}

func (r *commandInfoResolver) GetCommandPolicy(ctx context.Context, cmd Cmder) *routing.CommandPolicy {
	if r.resolveFunc == nil {
		return nil
	}

	policy := r.resolveFunc(ctx, cmd)
	if policy != nil {
		return policy
	}

	if r.fallBackResolver != nil {
		return r.fallBackResolver.GetCommandPolicy(ctx, cmd)
	}

	return nil
}

func (r *commandInfoResolver) SetFallbackResolver(fallbackResolver *commandInfoResolver) {
	r.fallBackResolver = fallbackResolver
}
