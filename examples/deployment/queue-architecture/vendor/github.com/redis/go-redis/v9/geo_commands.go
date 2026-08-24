package redis

import (
	"context"
	"errors"
)

type GeoCmdable interface {
	GeoAdd(ctx context.Context, key string, geoLocation ...*GeoLocation) *IntCmd
	GeoPos(ctx context.Context, key string, members ...string) *GeoPosCmd
	GeoRadius(ctx context.Context, key string, longitude, latitude float64, query *GeoRadiusQuery) *GeoLocationCmd
	GeoRadiusStore(ctx context.Context, key string, longitude, latitude float64, query *GeoRadiusQuery) *IntCmd
	GeoRadiusByMember(ctx context.Context, key, member string, query *GeoRadiusQuery) *GeoLocationCmd
	GeoRadiusByMemberStore(ctx context.Context, key, member string, query *GeoRadiusQuery) *IntCmd
	GeoSearch(ctx context.Context, key string, q *GeoSearchQuery) *StringSliceCmd
	GeoSearchLocation(ctx context.Context, key string, q *GeoSearchLocationQuery) *GeoSearchLocationCmd
	GeoSearchStore(ctx context.Context, key, store string, q *GeoSearchStoreQuery) *IntCmd
	GeoDist(ctx context.Context, key string, member1, member2, unit string) *FloatCmd
	GeoHash(ctx context.Context, key string, members ...string) *StringSliceCmd
}

func (c cmdable) GeoAdd(ctx context.Context, key string, geoLocation ...*GeoLocation) *IntCmd {
	args := make([]interface{}, 2+3*len(geoLocation))
	args[0] = "geoadd"
	args[1] = key
	for i, eachLoc := range geoLocation {
		args[2+3*i] = eachLoc.Longitude
		args[2+3*i+1] = eachLoc.Latitude
		args[2+3*i+2] = eachLoc.Name
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// GeoRadius queries a geospatial index for members within a distance from a coordinate.
// This is a read-only variant that does not support Store or StoreDist options.
//
// Deprecated: Use GeoSearch with BYRADIUS argument instead as of Redis 6.2.0.
func (c cmdable) GeoRadius(
	ctx context.Context, key string, longitude, latitude float64, query *GeoRadiusQuery,
) *GeoLocationCmd {
	cmd := NewGeoLocationCmd(ctx, query, "georadius_ro", key, longitude, latitude)
	if query.Store != "" || query.StoreDist != "" {
		cmd.SetErr(errors.New("GeoRadius does not support Store or StoreDist"))
		return cmd
	}
	_ = c(ctx, cmd)
	return cmd
}

// GeoRadiusStore is a writing GEORADIUS command.
func (c cmdable) GeoRadiusStore(
	ctx context.Context, key string, longitude, latitude float64, query *GeoRadiusQuery,
) *IntCmd {
	args := geoLocationArgs(query, "georadius", key, longitude, latitude)
	cmd := NewIntCmd(ctx, args...)
	if query.Store == "" && query.StoreDist == "" {
		cmd.SetErr(errors.New("GeoRadiusStore requires Store or StoreDist"))
		return cmd
	}
	_ = c(ctx, cmd)
	return cmd
}

// GeoRadiusByMember queries a geospatial index for members within a distance from a member.
// This is a read-only variant that does not support Store or StoreDist options.
//
// Deprecated: Use GeoSearch with BYRADIUS and FROMMEMBER arguments instead as of Redis 6.2.0.
func (c cmdable) GeoRadiusByMember(
	ctx context.Context, key, member string, query *GeoRadiusQuery,
) *GeoLocationCmd {
	cmd := NewGeoLocationCmd(ctx, query, "georadiusbymember_ro", key, member)
	if query.Store != "" || query.StoreDist != "" {
		cmd.SetErr(errors.New("GeoRadiusByMember does not support Store or StoreDist"))
		return cmd
	}
	_ = c(ctx, cmd)
	return cmd
}

// GeoRadiusByMemberStore is a writing GEORADIUSBYMEMBER command.
func (c cmdable) GeoRadiusByMemberStore(
	ctx context.Context, key, member string, query *GeoRadiusQuery,
) *IntCmd {
	args := geoLocationArgs(query, "georadiusbymember", key, member)
	cmd := NewIntCmd(ctx, args...)
	if query.Store == "" && query.StoreDist == "" {
		cmd.SetErr(errors.New("GeoRadiusByMemberStore requires Store or StoreDist"))
		return cmd
	}
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) GeoSearch(ctx context.Context, key string, q *GeoSearchQuery) *StringSliceCmd {
	args := make([]interface{}, 0, 13)
	args = append(args, "geosearch", key)
	args = geoSearchArgs(q, args)
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) GeoSearchLocation(
	ctx context.Context, key string, q *GeoSearchLocationQuery,
) *GeoSearchLocationCmd {
	args := make([]interface{}, 0, 16)
	args = append(args, "geosearch", key)
	args = geoSearchLocationArgs(q, args)
	cmd := NewGeoSearchLocationCmd(ctx, q, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) GeoSearchStore(ctx context.Context, key, store string, q *GeoSearchStoreQuery) *IntCmd {
	args := make([]interface{}, 0, 15)
	args = append(args, "geosearchstore", store, key)
	args = geoSearchArgs(&q.GeoSearchQuery, args)
	if q.StoreDist {
		args = append(args, "storedist")
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) GeoDist(
	ctx context.Context, key string, member1, member2, unit string,
) *FloatCmd {
	if unit == "" {
		unit = "km"
	}
	cmd := NewFloatCmd(ctx, "geodist", key, member1, member2, unit)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) GeoHash(ctx context.Context, key string, members ...string) *StringSliceCmd {
	args := make([]interface{}, 2+len(members))
	args[0] = "geohash"
	args[1] = key
	for i, member := range members {
		args[2+i] = member
	}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) GeoPos(ctx context.Context, key string, members ...string) *GeoPosCmd {
	args := make([]interface{}, 2+len(members))
	args[0] = "geopos"
	args[1] = key
	for i, member := range members {
		args[2+i] = member
	}
	cmd := NewGeoPosCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}
