//! ARM NEON SIMD implementation (4-lane, 128-bit).
//!
//! Uses `std::arch::aarch64` intrinsics. NEON is mandatory on ARMv8+.
//! Provides native FMA, blend, floor, and integer operations.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(not(target_arch = "aarch64"))]
use super::scalar::{ScalarFloat as NeonFloat, ScalarInt as NeonInt};

use super::{SimdFloat, SimdInt};

/// NEON float vector (4 lanes, float32x4_t).
#[derive(Clone, Copy, Debug)]
#[cfg(target_arch = "aarch64")]
pub struct NeonFloat(float32x4_t);

/// NEON integer vector (4 lanes, int32x4_t).
#[derive(Clone, Copy, Debug)]
#[cfg(target_arch = "aarch64")]
pub struct NeonInt(int32x4_t);

#[cfg(target_arch = "aarch64")]
impl SimdFloat for NeonFloat {
    const VECTOR_SIZE: usize = 4;

    #[inline]
    unsafe fn undefined() -> Self {
        // NEON has no "undefined" intrinsic; use zero as placeholder.
        Self(vdupq_n_f32(0.0))
    }

    #[inline]
    fn set(value: f32) -> Self {
        unsafe { Self(vdupq_n_f32(value)) }
    }

    #[inline]
    fn set1() -> Self {
        unsafe { Self(vdupq_n_f32(1.0)) }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        Self(vld1q_f32(ptr))
    }

    #[inline]
    unsafe fn store(ptr: *mut f32, value: Self) {
        vst1q_f32(ptr, value.0);
    }

    // --- Arithmetic ---

    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }

    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // NEON has native FMA: vfmaq_f32
        unsafe { Self(vfmaq_f32(c.0, self.0, b.0)) }
    }

    #[inline]
    fn mul_sub(self, b: Self, c: Self) -> Self {
        // NEON has native FMS: vfmsq_f32
        unsafe { Self(vfmsq_f32(c.0, self.0, b.0)) }
    }

    // --- Comparison ---

    #[inline]
    fn min(self, rhs: Self) -> Self {
        unsafe { Self(vminq_f32(self.0, rhs.0)) }
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        unsafe { Self(vmaxq_f32(self.0, rhs.0)) }
    }

    #[inline]
    fn less_than(self, rhs: Self) -> Self {
        // vcltq_f32 returns uint32x4_t (all-ones for true, all-zeros for false).
        // Reinterpret to float32x4_t for the trait.
        unsafe {
            let mask_u32 = vcltq_f32(self.0, rhs.0);
            Self(vreinterpretq_f32_u32(mask_u32))
        }
    }

    #[inline]
    fn blendv(self, b: Self, mask: Self) -> Self {
        // vbslq_f32(mask: uint32x4_t, a, b) → a where mask bits set, b otherwise.
        // Our mask is stored as float32x4_t; reinterpret.
        unsafe {
            let mask_u32 = vreinterpretq_u32_f32(mask.0);
            Self(vbslq_f32(mask_u32, self.0, b.0))
        }
    }

    // --- Math ---

    #[inline]
    fn abs(self) -> Self {
        unsafe { Self(vabsq_f32(self.0)) }
    }

    #[inline]
    fn inv_sqrt(self) -> Self {
        unsafe {
            let approx = vrsqrteq_f32(self.0);
            // Newton-Raphson refinement
            let half = vdupq_n_f32(0.5);
            let three_halfs = vdupq_n_f32(1.5);
            let tmp = vmulq_f32(vmulq_f32(half, self.0), vmulq_f32(approx, approx));
            Self(vmulq_f32(approx, vsubq_f32(three_halfs, tmp)))
        }
    }

    #[inline]
    fn floor(self) -> Self {
        // NEON native round-toward-minus-infinity (floor).
        // vrndmq_f32 rounds toward -infinity.
        unsafe { Self(vrndmq_f32(self.0)) }
    }
}

#[cfg(target_arch = "aarch64")]
impl SimdInt for NeonInt {
    const VECTOR_SIZE: usize = 4;
    type FloatType = NeonFloat;

    #[inline]
    fn set(value: i32) -> Self {
        unsafe { Self(vdupq_n_s32(value)) }
    }

    #[inline]
    fn set1() -> Self {
        unsafe { Self(vdupq_n_s32(1)) }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(vaddq_s32(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(vsubq_s32(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(vmulq_s32(self.0, rhs.0)) }
    }

    #[inline]
    fn and(self, rhs: Self) -> Self {
        unsafe { Self(vandq_s32(self.0, rhs.0)) }
    }

    #[inline]
    fn xor(self, rhs: Self) -> Self {
        unsafe { Self(veorq_s32(self.0, rhs.0)) }
    }

    #[inline]
    fn or(self, rhs: Self) -> Self {
        unsafe { Self(vorrq_s32(self.0, rhs.0)) }
    }

    #[inline]
    fn shift_right(self, rhs: i32) -> Self {
        // vshrq_n_s32 requires compile-time constant N.
        // Emulate via float: multiply by 2^{-rhs} and truncate.
        unsafe {
            let f = vcvtq_f32_s32(self.0);
            let factor = 2.0_f32.powi(-rhs);
            let shifted = vmulq_f32(f, vdupq_n_f32(factor));
            Self(vcvtq_s32_f32(shifted))
        }
    }

    #[inline]
    fn convert_to_float(self) -> Self::FloatType {
        unsafe { NeonFloat(vcvtq_f32_s32(self.0)) }
    }
}
