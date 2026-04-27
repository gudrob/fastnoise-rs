//! AVX2 + FMA SIMD implementation (8-lane, 256-bit).
//!
//! Stub – falls back to scalar. Full implementation requires `std::arch::x86_64`.

use super::scalar::{ScalarFloat, ScalarInt};

pub type Avx2Float = ScalarFloat;
pub type Avx2Int = ScalarInt;
