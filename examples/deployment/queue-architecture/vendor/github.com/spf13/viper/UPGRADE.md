# Update Log

**This document details any major updates required to use new features or improvements in Viper.**

## v1.20.x

### New file searching API

Viper now includes a new file searching API that allows users to customize how Viper looks for config files.

Viper accepts a custom [`Finder`](https://pkg.go.dev/github.com/spf13/viper#Finder) interface implementation:

```go
// Finder looks for files and directories in an [afero.Fs] filesystem.
type Finder interface {
	Find(fsys afero.Fs) ([]string, error)
}
```

It is supposed to return a list of paths to config files.

The default implementation uses [github.com/sagikazarmark/locafero](https://github.com/sagikazarmark/locafero) under the hood.

You can supply your own implementation using `WithFinder`:

```go
v := viper.NewWithOptions(
    viper.WithFinder(&MyFinder{}),
)
```

For more information, check out the [Finder examples](https://pkg.go.dev/github.com/spf13/viper#Finder)
and the [documentation](https://pkg.go.dev/github.com/sagikazarmark/locafero) for the locafero package.

### New encoding API

Viper now allows customizing the encoding layer by providing an API for encoding and decoding configuration data:

```go
// Encoder encodes Viper's internal data structures into a byte representation.
// It's primarily used for encoding a map[string]any into a file format.
type Encoder interface {
	Encode(v map[string]any) ([]byte, error)
}

// Decoder decodes the contents of a byte slice into Viper's internal data structures.
// It's primarily used for decoding contents of a file into a map[string]any.
type Decoder interface {
	Decode(b []byte, v map[string]any) error
}

// Codec combines [Encoder] and [Decoder] interfaces.
type Codec interface {
	Encoder
	Decoder
}
```

By default, Viper includes the following codecs:

- JSON
- TOML
- YAML
- Dotenv

The rest of the codecs are moved to [github.com/go-viper/encoding](https://github.com/go-viper/encoding)

Customizing the encoding layer is possible by providing a custom registry of codecs:

- [Encoder](https://pkg.go.dev/github.com/spf13/viper#Encoder) -> [EncoderRegistry](https://pkg.go.dev/github.com/spf13/viper#EncoderRegistry)
- [Decoder](https://pkg.go.dev/github.com/spf13/viper#Decoder) -> [DecoderRegistry](https://pkg.go.dev/github.com/spf13/viper#DecoderRegistry)
- [Codec](https://pkg.go.dev/github.com/spf13/viper#Codec) -> [CodecRegistry](https://pkg.go.dev/github.com/spf13/viper#CodecRegistry)

You can supply the registry of codecs to Viper using the appropriate `With*Registry` function:

```go
codecRegistry := viper.NewCodecRegistry()

codecRegistry.RegisterCodec("myformat", &MyCodec{})

v := viper.NewWithOptions(
    viper.WithCodecRegistry(codecRegistry),
)
```

### BREAKING: "github.com/mitchellh/mapstructure" depedency replaced

The original [mapstructure](https://github.com/mitchellh/mapstructure) has been [archived](https://github.com/mitchellh/mapstructure/issues/349) and was replaced with a [fork](https://github.com/go-viper/mapstructure) maintained by Viper ([#1723](https://github.com/spf13/viper/pull/1723)).

As a result, the package import path needs to be changed in cases where `mapstructure` is directly referenced in your code.

For example, when providing a custom decoder config:

```go
err := viper.Unmarshal(&appConfig, func(config *mapstructure.DecoderConfig) {
	config.TagName = "yaml"
})
```

The change is fairly straightforward, just replace all occurrences of the import path `github.com/mitchellh/mapstructure` with `github.com/go-viper/mapstructure/v2`:

```diff
- import "github.com/mitchellh/mapstructure"
+ import "github.com/go-viper/mapstructure/v2"
```

### BREAKING: HCL, Java properties, INI removed from core

In order to reduce third-party dependencies, Viper dropped support for the following formats from the core:

- HCL
- Java properties
- INI

You can still use these formats though by importing them from [github.com/go-viper/encoding](https://github.com/go-viper/encoding):

```go
import (
    "github.com/go-viper/encoding/hcl"
    "github.com/go-viper/encoding/javaproperties"
    "github.com/go-viper/encoding/ini"
)

codecRegistry := viper.NewCodecRegistry()

{
    codec := hcl.Codec{}

    codecRegistry.RegisterCodec("hcl", codec)
    codecRegistry.RegisterCodec("tfvars", codec)

}

{
    codec := &javaproperties.Codec{}

    codecRegistry.RegisterCodec("properties", codec)
    codecRegistry.RegisterCodec("props", codec)
    codecRegistry.RegisterCodec("prop", codec)
}

codecRegistry.RegisterCodec("ini", ini.Codec{})

v := viper.NewWithOptions(
    viper.WithCodecRegistry(codecRegistry),
)
```
