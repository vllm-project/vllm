target "test-ci" {
  output = ["type=registry,compression=zstd,compression-level=3,force-compression=true,oci-mediatypes=true"]
}
