//! SIMD abstraction layer for FastNoiseSIMD.
//!
//! Provides traits `SimdFloat` and `SimdInt` that abstract over
//! different SIMD instruction sets (scalar, SSE2, SSE4.1, AVX2, AVX-512F, NEON).
//!
//! ## Runtime dispatch
//! `SimdLevel::detect()` probes CPU features at runtime and returns the best level.
//! The code then dispatches to the appropriate implementation.
//!
//! ## Compile-time features
//! Feature flags (`sse2`, `sse41`, `avx2`, `avx512`, `neon`) can restrict
//! which SIMD paths are compiled into the binary.

/// SIMD level / instruction set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// Scalar fallback (1 lane, no SIMD).
    Scalar = 0,
    /// SSE2 (4 lanes, 128-bit).
    Sse2 = 1,
    /// SSE4.1 (4 lanes, 128-bit, with blend/floor instructions).
    Sse41 = 2,
    /// AVX2 + FMA (8 lanes, 256-bit).
    Avx2 = 3,
    /// AVX-512F (16 lanes, 512-bit).
    Avx512 = 4,
    /// ARM NEON (4 lanes, 128-bit).
    Neon = 5,
}

impl SimdLevel {
    /// The number of f32 lanes per SIMD vector for this level.
    #[must_use]
    #[allow(dead_code)]
    const fn vector_size(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse2 | SimdLevel::Sse41 | SimdLevel::Neon => 4,
            SimdLevel::Avx2 => 8,
            SimdLevel::Avx512 => 16,
        }
    }
}

/// Detect the best SIMD level available at runtime.
///
/// Probes CPU features and returns the highest supported level.
/// Falls back to `Scalar` if no SIMD is available.
#[must_use]
pub(crate) fn detect_simd_level() -> SimdLevel {
    detect_simd_level_impl()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_simd_level_impl() -> SimdLevel {
    if std::is_x86_feature_detected!("avx512f") {
        return SimdLevel::Avx512;
    }
    if std::is_x86_feature_detected!("avx2") {
        return SimdLevel::Avx2;
    }
    if std::is_x86_feature_detected!("sse4.1") {
        return SimdLevel::Sse41;
    }
    if std::is_x86_feature_detected!("sse2") {
        return SimdLevel::Sse2;
    }
    SimdLevel::Scalar
}

#[cfg(target_arch = "aarch64")]
fn detect_simd_level_impl() -> SimdLevel {
    // NEON is mandatory on ARMv8+. The `is_aarch64_feature_detected` macro
    // is available on aarch64 targets.
    if std::arch::is_aarch64_feature_detected!("neon") {
        return SimdLevel::Neon;
    }
    // Fallback: NEON is guaranteed on aarch64.
    SimdLevel::Neon
}

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64"
)))]
fn detect_simd_level_impl() -> SimdLevel {
    SimdLevel::Scalar
}

/// Trait for SIMD floating-point operations.
///
/// All operations correspond to the C++ `SIMDf_*` macros.
#[allow(dead_code)]
pub trait SimdFloat: Sized + Copy + Clone {
    /// Number of f32 lanes.
    const VECTOR_SIZE: usize;

    /// Undefined / uninitialized value.
    unsafe fn undefined() -> Self;

    /// Broadcast a scalar to all lanes.
    fn set(value: f32) -> Self;

    /// Set all lanes to 1.0.
    fn set1() -> Self;

    /// Load from aligned memory.
    unsafe fn load(ptr: *const f32) -> Self;

    /// Store to aligned memory.
    unsafe fn store(ptr: *mut f32, value: Self);

    // --- Arithmetic ---

    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    /// Fused multiply-add: `self * b + c`
    fn mul_add(self, b: Self, c: Self) -> Self;
    /// Fused multiply-sub: `self * b - c`
    fn mul_sub(self, b: Self, c: Self) -> Self;

    // --- Comparison ---

    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;
    /// Returns a mask (all bits 1 = true, all bits 0 = false) where self < rhs.
    fn less_than(self, rhs: Self) -> Self;
    /// Blend based on mask: `(mask & a) | (!mask & b)`
    fn blendv(self, b: Self, mask: Self) -> Self;

    // --- Math ---

    fn abs(self) -> Self;
    /// Reciprocal square root approximation (with optional Newton refinement).
    fn inv_sqrt(self) -> Self;
    /// Floor.
    fn floor(self) -> Self;
}

/// Trait for SIMD integer operations.
#[allow(dead_code)]
pub trait SimdInt: Sized + Copy + Clone {
    /// Number of i32 lanes.
    const VECTOR_SIZE: usize;

    /// Broadcast a scalar to all lanes.
    fn set(value: i32) -> Self;
    /// Set all lanes to 1.
    fn set1() -> Self;

    // --- Arithmetic ---

    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;

    // --- Bitwise ---

    fn and(self, rhs: Self) -> Self;
    fn xor(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
    fn shift_right(self, rhs: i32) -> Self;

    // --- Conversion ---

    /// The corresponding SIMD float type.
    type FloatType: SimdFloat;

    /// Convert to SIMD float (truncates, same as C-style cast).
    fn convert_to_float(self) -> Self::FloatType;
}

// Platform-specific implementations
pub(crate) mod scalar;

// Using `has_sse2` cfg set by build.rs (from feature flag or target_feature detection).
#[cfg(all(target_arch = "x86_64", has_sse2))]
pub(crate) mod sse2;
#[cfg(not(all(target_arch = "x86_64", has_sse2)))]
pub(crate) mod sse2 {
    #[allow(unused_imports)]
    pub(crate) use super::scalar::ScalarFloat as Sse2Float;
    #[allow(unused_imports)]
    pub(crate) use super::scalar::ScalarInt as Sse2Int;
}

#[cfg(all(target_arch = "x86_64", has_sse41))]
pub(crate) mod sse41;
#[cfg(not(all(target_arch = "x86_64", has_sse41)))]
pub(crate) mod sse41 {
    #[allow(unused_imports)]
    pub(crate) use super::scalar::ScalarFloat as Sse41Float;
    #[allow(unused_imports)]
    pub(crate) use super::scalar::ScalarInt as Sse41Int;
}

// Note: avx2.rs is a stub (scalar fallback) until full AVX2 implementation.
#[cfg(all(target_arch = "x86_64", has_avx2))]
pub(crate) mod avx2;
#[cfg(not(all(target_arch = "x86_64", has_avx2)))]
pub(crate) mod avx2 {
    #[allow(unused_imports)]
    pub(crate) use super::scalar::ScalarFloat as Avx2Float;
    #[allow(unused_imports)]
    pub(crate) use super::scalar::ScalarInt as Avx2Int;
}

// Note: avx512.rs is a stub (scalar fallback) until full AVX-512F implementation.
#[cfg(all(target_arch = "x86_64", has_avx512))]
pub(crate) mod avx512;
#[cfg(not(all(target_arch = "x86_64", has_avx512)))]
pub(crate) mod avx512 {
    #[allow(unused_imports)]
    pub(crate) use super::scalar::ScalarFloat as Avx512Float;
    #[allow(unused_imports)]
    pub(crate) use super::scalar::ScalarInt as Avx512Int;
}

// Note: neon.rs is a stub (scalar fallback) until full NEON implementation.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    has_neon
))]
pub(crate) mod neon;
#[cfg(not(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    has_neon
)))]
#[allow(dead_code)]
pub(crate) mod neon {
    pub(crate) use super::scalar::ScalarFloat as NeonFloat;
    pub(crate) use super::scalar::ScalarInt as NeonInt;
}
