//! Build script that wires Cargo features to cfg flags for SIMD modules.
//!
//! The Cargo features (`sse2`, `sse41`, `avx2`, `avx512`, `neon`) are used
//! to conditionally compile the corresponding SIMD backend modules.
//! Without any feature, only the scalar fallback is compiled.
//!
//! All SIMD backends (SSE2, SSE4.1, AVX2, AVX-512F, NEON) have real implementations
//! using `std::arch` intrinsics. When the corresponding feature is not enabled,
//! the type aliases fall back to scalar.

fn main() {
    // Declare custom cfg flags so rustc doesn't warn about them
    println!("cargo:rustc-check-cfg=cfg(has_sse2)");
    println!("cargo:rustc-check-cfg=cfg(has_sse41)");
    println!("cargo:rustc-check-cfg=cfg(has_avx2)");
    println!("cargo:rustc-check-cfg=cfg(has_avx512)");
    println!("cargo:rustc-check-cfg=cfg(has_neon)");

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // Read Cargo features
    // We use compile-time cfg flags: `has_sse2`, `has_sse41`, `has_avx2`, `has_avx512`, `has_neon`

    #[cfg(feature = "sse2")]
    println!("cargo:rustc-cfg=has_sse2");
    #[cfg(feature = "sse41")]
    println!("cargo:rustc-cfg=has_sse41");
    #[cfg(feature = "avx2")]
    println!("cargo:rustc-cfg=has_avx2");
    #[cfg(feature = "avx512")]
    println!("cargo:rustc-cfg=has_avx512");
    #[cfg(feature = "neon")]
    println!("cargo:rustc-cfg=has_neon");

    match target_arch.as_str() {
        "x86" | "x86_64" => {
            // Auto-enable features based on compiler target features
            // (set by -C target-cpu or target-feature)
            #[cfg(target_feature = "sse2")]
            println!("cargo:rustc-cfg=has_sse2");
            #[cfg(target_feature = "sse4.1")]
            println!("cargo:rustc-cfg=has_sse41");
            #[cfg(target_feature = "avx2")]
            println!("cargo:rustc-cfg=has_avx2");
            #[cfg(target_feature = "avx512f")]
            println!("cargo:rustc-cfg=has_avx512");
        }
        "aarch64" | "arm" => {
            // NEON is baseline on AArch64; check for armv7+neon
            #[cfg(target_feature = "neon")]
            println!("cargo:rustc-cfg=has_neon");
        }
        _ => {}
    }

    // Re-run if build.rs or Cargo.toml changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
}
