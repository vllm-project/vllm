# External Dependencies

This file lists the dependencies used in this repository.

{{/* compress has actually a BSD 3-Clause license, but the License file in the repo confuses go-license tooling, hence the manual exception */}}
| Dependency                                       | License                                 |
|--------------------------------------------------|-----------------------------------------|
{{ range . }}| {{ .Name }} | {{ if eq .Name "github.com/klauspost/compress/flate" }}BSD 3-Clause{{ else }}{{ .LicenseName }}{{ end }} |
{{ end }}
