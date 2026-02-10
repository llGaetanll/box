use std::error::Error;
use std::process::Command;

use spirv_builder::MetadataPrintout;
use spirv_builder::SpirvBuilder;

fn build_shader(path_to_crate: &str) -> Result<(), Box<dyn Error>> {
    let builder = SpirvBuilder::new(path_to_crate, "spirv-unknown-vulkan1.2")
        .print_metadata(MetadataPrintout::Full);

    let _result = builder.build()?;
    Ok(())
}

fn set_git_sha() {
    // Re-run if HEAD changes (e.g., branch switch)
    println!("cargo::rerun-if-changed=../../.git/HEAD");

    // Also watch the ref that HEAD points to (e.g., refs/heads/master)
    // This ensures we re-run when new commits are made on the current branch
    if let Ok(head_contents) = std::fs::read_to_string("../../.git/HEAD") {
        if let Some(ref_path) = head_contents.trim().strip_prefix("ref: ") {
            println!("cargo::rerun-if-changed=../../.git/{}", ref_path);
        }
    }

    let sha = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo::rustc-env=GIT_SHA={}", sha);
}

fn main() -> Result<(), Box<dyn Error>> {
    build_shader("../shader")?;
    set_git_sha();

    Ok(())
}
