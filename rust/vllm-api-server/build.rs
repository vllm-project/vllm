fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)  // We only need client
        .build_client(true)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &["../proto/vllm_engine.proto"],
            &["../proto"],
        )?;
    Ok(())
}
