// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_dir = std::path::PathBuf::from(format!("{manifest_dir}/../../proto"));
    let protos = [
        proto_dir.join("control.proto"),
        proto_dir.join("inference.proto"),
    ];

    for proto in &protos {
        println!("cargo:rerun-if-changed={}", proto.display());
    }

    let file_descriptor_set = protox::compile(&protos, [&proto_dir])?;

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_fds(file_descriptor_set)?;

    Ok(())
}
