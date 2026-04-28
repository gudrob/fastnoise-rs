//! SSE4.1 SIMD implementation (4-lane, 128-bit).
//!
//! Adds _mm_blendv_ps and _mm_floor_ps over SSE2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{SimdFloat, SimdInt};

/// SSE4.1 float vector (4 lanes, __m128) – reuses SSE2 with blendv/floor improvements.
#[derive(Clone, Copy, Debug)]
pub struct Sse41Float(__m128);

/// SSE4.1 integer vector (4 lanes, __m128i).
#[derive(Clone, Copy, Debug)]
pub struct Sse41Int(__m128i);

impl SimdFloat for Sse41Float {
    const VECTOR_SIZE: usize = 4;

    #[inline]
    unsafe fn undefined() -> Self {
        Self(_mm_undefined_ps())
    }

    #[inline]
    fn set(value: f32) -> Self {
        unsafe { Self(_mm_set1_ps(value)) }
    }

    #[inline]
    fn set1() -> Self {
        unsafe { Self(_mm_set1_ps(1.0)) }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        Self(_mm_loadu_ps(ptr))
    }

    #[inline]
    unsafe fn store(ptr: *mut f32, value: Self) {
        _mm_storeu_ps(ptr, value.0);
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm_add_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm_sub_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm_mul_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // SSE4.1 does not have FMA. Emulate: a * b + c
        unsafe {
            let tmp = _mm_mul_ps(self.0, b.0);
            Self(_mm_add_ps(tmp, c.0))
        }
    }

    #[inline]
    fn mul_sub(self, b: Self, c: Self) -> Self {
        unsafe {
            let tmp = _mm_mul_ps(self.0, b.0);
            Self(_mm_sub_ps(tmp, c.0))
        }
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        unsafe { Self(_mm_min_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        unsafe { Self(_mm_max_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn less_than(self, rhs: Self) -> Self {
        unsafe { Self(_mm_cmplt_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn blendv(self, b: Self, mask: Self) -> Self {
        // SSE4.1 native blend
        unsafe { Self(_mm_blendv_ps(b.0, self.0, mask.0)) }
    }

    #[inline]
    fn abs(self) -> Self {
        unsafe {
            let sign_mask = _mm_set1_ps(-0.0_f32);
            Self(_mm_andnot_ps(sign_mask, self.0))
        }
    }

    #[inline]
    fn inv_sqrt(self) -> Self {
        unsafe {
            let approx = _mm_rsqrt_ps(self.0);
            let half = _mm_set1_ps(0.5);
            let three_halfs = _mm_set1_ps(1.5);
            let tmp = _mm_mul_ps(_mm_mul_ps(half, self.0), _mm_mul_ps(approx, approx));
            Self(_mm_mul_ps(approx, _mm_sub_ps(three_halfs, tmp)))
        }
    }

    #[inline]
    fn floor(self) -> Self {
        // SSE4.1 native floor
        unsafe { Self(_mm_floor_ps(self.0)) }
    }
}

impl SimdInt for Sse41Int {
    const VECTOR_SIZE: usize = 4;
    type FloatType = Sse41Float;

    #[inline]
    fn set(value: i32) -> Self {
        unsafe { Self(_mm_set1_epi32(value)) }
    }

    #[inline]
    fn set1() -> Self {
        unsafe { Self(_mm_set1_epi32(1)) }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm_add_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm_sub_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // SSE4.1 has _mm_mullo_epi32
        unsafe { Self(_mm_mullo_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn and(self, rhs: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, rhs.0)) }
    }

    #[inline]
    fn xor(self, rhs: Self) -> Self {
        unsafe { Self(_mm_xor_si128(self.0, rhs.0)) }
    }

    #[inline]
    fn or(self, rhs: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, rhs.0)) }
    }

    #[inline]
    fn shift_right(self, rhs: i32) -> Self {
        // _mm_srai_epi32 requires a compile-time constant.
        // Workaround: convert to f32, shift via multiplication by 2^{-rhs}, convert back.
        unsafe {
            let f = _mm_cvtepi32_ps(self.0);
            let factor = 2.0_f32.powi(-rhs);
            let shifted = _mm_mul_ps(f, _mm_set1_ps(factor));
            Self(_mm_cvttps_epi32(shifted))
        }
    }

    #[inline]
    fn convert_to_float(self) -> Self::FloatType {
        unsafe { Sse41Float(_mm_cvtepi32_ps(self.0)) }
    }
}
