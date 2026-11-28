use std::path::PathBuf;

fn package_version(lock: &str, package: &str) -> String {
    lock.split("[[package]]")
        .find_map(|section| {
            let mut name = None;
            let mut version = None;
            for line in section.lines() {
                if let Some(value) = line.strip_prefix("name = \"") {
                    name = value.strip_suffix('\"');
                }
                if let Some(value) = line.strip_prefix("version = \"") {
                    version = value.strip_suffix('\"');
                }
            }
            (name == Some(package)).then(|| version.map(str::to_owned))?
        })
        .unwrap_or_else(|| panic!("missing {package} from Cargo.lock"))
}

fn main() {
    let lock_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../..")
        .join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    let lock = std::fs::read_to_string(&lock_path).unwrap();
    for (package, variable) in [
        ("fastokens", "VLLM_FASTOKENS_VERSION"),
        ("tokenizers", "VLLM_TOKENIZERS_VERSION"),
        ("riptoken", "VLLM_RIPTOKEN_VERSION"),
        ("tiktoken-rs", "VLLM_TIKTOKEN_RS_VERSION"),
        ("tekken", "VLLM_TEKKEN_VERSION"),
    ] {
        println!(
            "cargo:rustc-env={variable}={}",
            package_version(&lock, package)
        );
    }
}
