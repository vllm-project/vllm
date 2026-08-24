package redis

import "time"

// NewCmdResult returns a Cmd initialised with val and err for testing.
func NewCmdResult(val interface{}, err error) *Cmd {
	var cmd Cmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewSliceResult returns a SliceCmd initialised with val and err for testing.
func NewSliceResult(val []interface{}, err error) *SliceCmd {
	var cmd SliceCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewStatusResult returns a StatusCmd initialised with val and err for testing.
func NewStatusResult(val string, err error) *StatusCmd {
	var cmd StatusCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewIntResult returns an IntCmd initialised with val and err for testing.
func NewIntResult(val int64, err error) *IntCmd {
	var cmd IntCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewDurationResult returns a DurationCmd initialised with val and err for testing.
func NewDurationResult(val time.Duration, err error) *DurationCmd {
	var cmd DurationCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewBoolResult returns a BoolCmd initialised with val and err for testing.
func NewBoolResult(val bool, err error) *BoolCmd {
	var cmd BoolCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewStringResult returns a StringCmd initialised with val and err for testing.
func NewStringResult(val string, err error) *StringCmd {
	var cmd StringCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewFloatResult returns a FloatCmd initialised with val and err for testing.
func NewFloatResult(val float64, err error) *FloatCmd {
	var cmd FloatCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewStringSliceResult returns a StringSliceCmd initialised with val and err for testing.
func NewStringSliceResult(val []string, err error) *StringSliceCmd {
	var cmd StringSliceCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewBoolSliceResult returns a BoolSliceCmd initialised with val and err for testing.
func NewBoolSliceResult(val []bool, err error) *BoolSliceCmd {
	var cmd BoolSliceCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewFloatSliceResult returns a FloatSliceCmd initialised with val and err for testing.
func NewFloatSliceResult(val []float64, err error) *FloatSliceCmd {
	var cmd FloatSliceCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewMapStringStringResult returns a MapStringStringCmd initialised with val and err for testing.
func NewMapStringStringResult(val map[string]string, err error) *MapStringStringCmd {
	var cmd MapStringStringCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewMapStringIntCmdResult returns a MapStringIntCmd initialised with val and err for testing.
func NewMapStringIntCmdResult(val map[string]int64, err error) *MapStringIntCmd {
	var cmd MapStringIntCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewTimeCmdResult returns a TimeCmd initialised with val and err for testing.
func NewTimeCmdResult(val time.Time, err error) *TimeCmd {
	var cmd TimeCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewZSliceCmdResult returns a ZSliceCmd initialised with val and err for testing.
func NewZSliceCmdResult(val []Z, err error) *ZSliceCmd {
	var cmd ZSliceCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewZWithKeyCmdResult returns a ZWithKeyCmd initialised with val and err for testing.
func NewZWithKeyCmdResult(val *ZWithKey, err error) *ZWithKeyCmd {
	var cmd ZWithKeyCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewScanCmdResult returns a ScanCmd initialised with val and err for testing.
func NewScanCmdResult(keys []string, cursor uint64, err error) *ScanCmd {
	var cmd ScanCmd
	cmd.page = keys
	cmd.cursor = cursor
	cmd.SetErr(err)
	return &cmd
}

// NewClusterSlotsCmdResult returns a ClusterSlotsCmd initialised with val and err for testing.
func NewClusterSlotsCmdResult(val []ClusterSlot, err error) *ClusterSlotsCmd {
	var cmd ClusterSlotsCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewGeoLocationCmdResult returns a GeoLocationCmd initialised with val and err for testing.
func NewGeoLocationCmdResult(val []GeoLocation, err error) *GeoLocationCmd {
	var cmd GeoLocationCmd
	cmd.locations = val
	cmd.SetErr(err)
	return &cmd
}

// NewGeoPosCmdResult returns a GeoPosCmd initialised with val and err for testing.
func NewGeoPosCmdResult(val []*GeoPos, err error) *GeoPosCmd {
	var cmd GeoPosCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewCommandsInfoCmdResult returns a CommandsInfoCmd initialised with val and err for testing.
func NewCommandsInfoCmdResult(val map[string]*CommandInfo, err error) *CommandsInfoCmd {
	var cmd CommandsInfoCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewXMessageSliceCmdResult returns a XMessageSliceCmd initialised with val and err for testing.
func NewXMessageSliceCmdResult(val []XMessage, err error) *XMessageSliceCmd {
	var cmd XMessageSliceCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewXStreamSliceCmdResult returns a XStreamSliceCmd initialised with val and err for testing.
func NewXStreamSliceCmdResult(val []XStream, err error) *XStreamSliceCmd {
	var cmd XStreamSliceCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}

// NewXPendingResult returns a XPendingCmd initialised with val and err for testing.
func NewXPendingResult(val *XPending, err error) *XPendingCmd {
	var cmd XPendingCmd
	cmd.val = val
	cmd.SetErr(err)
	return &cmd
}
