//! Scalar (1-lane) SIMD implementation – the reference fallback.
//!
//! This implements `SimdFloat` and `SimdInt` using plain `f32` and `i32`.

use super::{SimdFloat, SimdInt};

/// Scalar float wrapper (1 lane, just wraps f32).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarFloat(pub f32);

/// Scalar int wrapper (1 lane, just wraps i32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarInt(pub i32);

impl SimdFloat for ScalarFloat {
    const VECTOR_SIZE: usize = 1;

    #[inline]
    unsafe fn undefined() -> Self {
        // In scalar mode, use 0.0 as placeholder (same as C++ SIMD_ZERO_ALL)
        Self(0.0)
    }

    #[inline]
    fn set(value: f32) -> Self {
        Self(value)
    }

    #[inline]
    fn set1() -> Self {
        Self(1.0)
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        Self(*ptr)
    }

    #[inline]
    unsafe fn store(ptr: *mut f32, value: Self) {
        *ptr = value.0;
    }

    // --- Arithmetic ---

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }

    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // Use mul_add for precision matching FMA
        Self(self.0.mul_add(b.0, c.0))
    }

    #[inline]
    fn mul_sub(self, b: Self, c: Self) -> Self {
        Self(self.0 * b.0 - c.0)
    }

    // --- Comparison ---

    #[inline]
    fn min(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }

    #[inline]
    fn less_than(self, rhs: Self) -> Self {
        // Use bit pattern: all 1s for true, all 0s for false
        let mask: u32 = if self.0 < rhs.0 { u32::MAX } else { 0 };
        Self(f32::from_bits(mask))
    }

    #[inline]
    fn blendv(self, b: Self, mask: Self) -> Self {
        // mask bits: all 1s = pick self, all 0s = pick b
        if mask.0.to_bits() == u32::MAX {
            self
        } else {
            b
        }
    }

    // --- Math ---

    #[inline]
    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    #[inline]
    fn inv_sqrt(self) -> Self {
        // Reciprocal square root: 1.0 / sqrt(x)
        // With Newton-Raphson refinement for precision (matches C++ SIMD)
        let x = self.0;
        if x <= 0.0 {
            return Self(0.0);
        }
        let rsqrt = 1.0 / x.sqrt();
        #[cfg(feature = "fma")]
        {
            // Newton-Raphson: rsqrt * (1.5 - 0.5 * x * rsqrt * rsqrt)
            let half = 0.5;
            let r = x.mul_add(-half * rsqrt * rsqrt, 1.5) * rsqrt;
            Self(r)
        }
        #[cfg(not(feature = "fma"))]
        {
            Self(rsqrt)
        }
    }

    #[inline]
    fn floor(self) -> Self {
        Self(self.0.floor())
    }
}

impl SimdInt for ScalarInt {
    const VECTOR_SIZE: usize = 1;
    type FloatType = ScalarFloat;

    #[inline]
    fn set(value: i32) -> Self {
        Self(value)
    }

    #[inline]
    fn set1() -> Self {
        Self(1)
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }

    #[inline]
    fn and(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }

    #[inline]
    fn xor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    #[inline]
    fn or(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }

    #[inline]
    fn shift_right(self, rhs: i32) -> Self {
        Self(self.0 >> rhs)
    }

    #[inline]
    fn convert_to_float(self) -> Self::FloatType {
        ScalarFloat(self.0 as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_mul_add() {
        let a = ScalarFloat(2.0);
        let b = ScalarFloat(3.0);
        let c = ScalarFloat(4.0);
        let result = a.mul_add(b, c);
        assert!((result.0 - 10.0).abs() < 1e-6); // 2*3+4 = 10
    }

    #[test]
    fn test_scalar_blendv() {
        let a = ScalarFloat(5.0);
        let b = ScalarFloat(9.0);
        let mask_true = ScalarFloat(f32::from_bits(u32::MAX));
        let mask_false = ScalarFloat(0.0);

        assert_eq!(a.blendv(b, mask_true).0, 5.0);
        assert_eq!(a.blendv(b, mask_false).0, 9.0);
    }

    #[test]
    fn test_scalar_inv_sqrt() {
        let x = ScalarFloat(4.0);
        let r = x.inv_sqrt().0;
        assert!((r - 0.5).abs() < 1e-4); // 1/sqrt(4) = 0.5
    }

    #[test]
    fn test_scalar_int_convert() {
        let i = ScalarInt::set(42);
        let f = i.convert_to_float();
        assert!((f.0 - 42.0).abs() < 1e-6);
    }
}
