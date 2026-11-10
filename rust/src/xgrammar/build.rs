// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Builds the vendored xgrammar C++ sources plus the C ABI shim into a static
//! library, mirroring the upstream CMake `xgrammar` target in a Release build
//! (`-O3 -DNDEBUG`, C++17, no cpptrace, no internal checks).
//!
//! `vendor/xgrammar` is https://github.com/mlc-ai/xgrammar at tag `v0.2.8`
//! (archive sha256
//! `16c06f7cef8f13ae597b007cce515991725603bded6ebce5f1828e6a9c9685be`):
//! `cpp/` without the `tvm_ffi` Python bindings, `include/`, `LICENSE`, and
//! `NOTICE` (Apache-2.0), plus the `3rdparty` headers it compiles against:
//! `picojson/picojson.h` (BSD-2-Clause, in the archive) and
//! `dlpack/include/dlpack/dlpack.h` (Apache-2.0, the tag's submodule commit
//! `bbd2f4d`). Keep the version in lockstep with the `xgrammar` Python
//! requirement (`requirements/common.txt`) so both engines agree.

use std::path::Path;

fn main() {
    let root = Path::new("vendor/xgrammar");
    let mut sources: Vec<_> = Vec::new();
    collect_sources(&root.join("cpp"), &mut sources);
    sources.sort();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .define("NDEBUG", None)
        .define("XGRAMMAR_ENABLE_CPPTRACE", "0")
        .define("XGRAMMAR_ENABLE_INTERNAL_CHECK", "0")
        .include(root.join("include"))
        .include(root.join("3rdparty/picojson"))
        .include(root.join("3rdparty/dlpack/include"))
        // `__FILE__` in error messages reads like the wheel's (`/project/cpp/...`).
        .flag_if_supported("-fmacro-prefix-map=vendor/xgrammar=/project")
        .warnings(false)
        .files(&sources)
        .file("cpp/shim.cc");
    build.compile("xgrammar");

    println!("cargo:rerun-if-changed=cpp/shim.cc");
    println!("cargo:rerun-if-changed=vendor/xgrammar");
}

fn collect_sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read vendored xgrammar sources") {
        let path = entry.expect("read dir entry").path();
        if path.is_dir() {
            collect_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "cc") {
            out.push(path);
        }
    }
}
