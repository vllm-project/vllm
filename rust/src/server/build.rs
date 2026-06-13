fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_dir = format!("{manifest_dir}/../../proto");

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .protoc_arg("--experimental_allow_proto3_optional") // be compatible with old compilers
        .compile_protos(&[format!("{proto_dir}/vllm_grpc.proto")], &[proto_dir])?;

    Ok(())
}
