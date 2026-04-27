//! AVX-512F SIMD implementation (16-lane, 512-bit).
//!
//! Stub – falls back to scalar. Full implementation requires `std::arch::x86_64`.

use super::scalar::{ScalarFloat, ScalarInt};

pub type Avx512Float = ScalarFloat;
pub type Avx512Int = ScalarInt;
