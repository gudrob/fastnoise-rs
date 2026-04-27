//! ARM NEON SIMD implementation (4-lane, 128-bit).
//!
//! Stub – falls back to scalar. Full implementation requires `std::arch::aarch64`.

use super::scalar::{ScalarFloat, ScalarInt};

pub type NeonFloat = ScalarFloat;
pub type NeonInt = ScalarInt;
