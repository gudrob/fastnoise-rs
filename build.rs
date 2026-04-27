//! Build script that detects CPU features and sets cfg flags.
//! No runtime detection needed – Rust's `#[cfg(target_feature)]` handles this.
//! Runtime SIMD level detection happens in `simd/mod.rs`.

fn main() {
    // Detect target architecture and features
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    match target_arch.as_str() {
        "x86" | "x86_64" => {
            // Enable x86 SIMD features based on target features
            println!("cargo:rustc-cfg=x86_arch");

            // SSE2 is baseline for x86_64, but we still check
            // These are set by rustc based on -C target-cpu/target-feature
        }
        "aarch64" => {
            println!("cargo:rustc-cfg=arm_arch");
            // NEON is baseline on AArch64
        }
        "arm" => {
            println!("cargo:rustc-cfg=arm_arch");
        }
        _ => {
            // Unknown architecture – scalar fallback only
            println!("cargo:rustc-cfg=unknown_arch");
        }
    }

    // Re-run if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}