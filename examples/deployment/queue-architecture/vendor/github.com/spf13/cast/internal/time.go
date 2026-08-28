package internal

import (
	"fmt"
	"time"
)

//go:generate stringer -type=TimeFormatType

type TimeFormatType int

const (
	TimeFormatNoTimezone TimeFormatType = iota
	TimeFormatNamedTimezone
	TimeFormatNumericTimezone
	TimeFormatNumericAndNamedTimezone
	TimeFormatTimeOnly
)

type TimeFormat struct {
	Format string
	Typ    TimeFormatType
}

func (f TimeFormat) HasTimezone() bool {
	// We don't include the formats with only named timezones, see
	// https://github.com/golang/go/issues/19694#issuecomment-289103522
	return f.Typ >= TimeFormatNumericTimezone && f.Typ <= TimeFormatNumericAndNamedTimezone
}

var TimeFormats = []TimeFormat{
	// Keep common formats at the top.
	{"2006-01-02", TimeFormatNoTimezone},
	{time.RFC3339, TimeFormatNumericTimezone},
	{"2006-01-02T15:04:05", TimeFormatNoTimezone}, // iso8601 without timezone
	{time.RFC1123Z, TimeFormatNumericTimezone},
	{time.RFC1123, TimeFormatNamedTimezone},
	{time.RFC822Z, TimeFormatNumericTimezone},
	{time.RFC822, TimeFormatNamedTimezone},
	{time.RFC850, TimeFormatNamedTimezone},
	{"2006-01-02 15:04:05.999999999 -0700 MST", TimeFormatNumericAndNamedTimezone}, // Time.String()
	{"2006-01-02T15:04:05-0700", TimeFormatNumericTimezone},                        // RFC3339 without timezone hh:mm colon
	{"2006-01-02 15:04:05Z0700", TimeFormatNumericTimezone},                        // RFC3339 without T or timezone hh:mm colon
	{"2006-01-02 15:04:05", TimeFormatNoTimezone},
	{time.ANSIC, TimeFormatNoTimezone},
	{time.UnixDate, TimeFormatNamedTimezone},
	{time.RubyDate, TimeFormatNumericTimezone},
	{"2006-01-02 15:04:05Z07:00", TimeFormatNumericTimezone},
	{"02 Jan 2006", TimeFormatNoTimezone},
	{"2006-01-02 15:04:05 -07:00", TimeFormatNumericTimezone},
	{"2006-01-02 15:04:05 -0700", TimeFormatNumericTimezone},
	{time.Kitchen, TimeFormatTimeOnly},
	{time.Stamp, TimeFormatTimeOnly},
	{time.StampMilli, TimeFormatTimeOnly},
	{time.StampMicro, TimeFormatTimeOnly},
	{time.StampNano, TimeFormatTimeOnly},
}

func ParseDateWith(s string, location *time.Location, formats []TimeFormat) (d time.Time, e error) {
	for _, format := range formats {
		if d, e = time.Parse(format.Format, s); e == nil {

			// Some time formats have a zone name, but no offset, so it gets
			// put in that zone name (not the default one passed in to us), but
			// without that zone's offset. So set the location manually.
			if format.Typ <= TimeFormatNamedTimezone {
				if location == nil {
					location = time.Local
				}
				year, month, day := d.Date()
				hour, min, sec := d.Clock()
				d = time.Date(year, month, day, hour, min, sec, d.Nanosecond(), location)
			}

			return
		}
	}
	return d, fmt.Errorf("unable to parse date: %s", s)
}
