target "test-ci" {
  # Bake tags override every image exporter's name; each format needs its own.
  tags = []
  output = [
    {
      type              = "image"
      name              = join(",", compact([IMAGE_TAG, IMAGE_TAG_LATEST]))
      push              = true
      compression       = "gzip"
      force-compression = true
    },
    {
      type              = "image"
      name              = join(",", compact(["${IMAGE_TAG}-zstd", IMAGE_TAG_LATEST != "" ? "${IMAGE_TAG_LATEST}-zstd" : ""]))
      push              = true
      compression       = "zstd"
      compression-level = 3
      force-compression = true
      oci-mediatypes    = true
    },
  ]
}
