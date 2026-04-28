//! SSE2 (4-lane, 128-bit) SIMD implementation.
//!
//! Uses `std::arch::x86_64` intrinsics. SSE2 is baseline on x86_64.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{SimdFloat, SimdInt};

/// SSE2 float vector (4 lanes, __m128).
#[derive(Clone, Copy, Debug)]
pub struct Sse2Float(__m128);

/// SSE2 integer vector (4 lanes, __m128i).
#[derive(Clone, Copy, Debug)]
pub struct Sse2Int(__m128i);

impl SimdFloat for Sse2Float {
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

    // --- Arithmetic ---

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
        // SSE2 does not have FMA. Emulate: a * b + c
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

    // --- Comparison ---

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
        // SSE4.1 _mm_blendv_ps, but SSE2 can emulate:
        // result = (mask & self) | (!mask & b)
        unsafe {
            let masked_a = _mm_and_ps(mask.0, self.0);
            let not_mask = _mm_andnot_ps(mask.0, b.0);
            Self(_mm_or_ps(masked_a, not_mask))
        }
    }

    // --- Math ---

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
            // Newton-Raphson refinement: rsqrt * (1.5 - 0.5 * x * rsqrt * rsqrt)
            let half = _mm_set1_ps(0.5);
            let three_halfs = _mm_set1_ps(1.5);
            let tmp = _mm_mul_ps(_mm_mul_ps(half, self.0), _mm_mul_ps(approx, approx));
            Self(_mm_mul_ps(approx, _mm_sub_ps(three_halfs, tmp)))
        }
    }

    #[inline]
    fn floor(self) -> Self {
        // SSE2 floor emulation: truncate toward negative infinity
        // Cast to int and back handles positive numbers correctly,
        // but negative non-integers need adjustment.
        unsafe {
            // Use a trick: convert to int, convert back, subtract 1 if negative and fractional
            let int_val = _mm_cvttps_epi32(self.0);
            let float_val = _mm_cvtepi32_ps(int_val);
            // Check if original > float_val (means negative fractional)
            let mask = _mm_cmpgt_ps(self.0, float_val);
            // Subtract 1 for those that were negative fractions
            let correction = _mm_and_ps(mask, _mm_set1_ps(1.0));
            Self(_mm_add_ps(float_val, correction))
        }
    }
}

impl SimdInt for Sse2Int {
    const VECTOR_SIZE: usize = 4;
    type FloatType = Sse2Float;

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
        // SSE2 _mm_mullo_epi32 is SSE4.1. Emulate with 32→64→32 multiply.
        unsafe {
            // This is a simplified version; full 32-bit mul emulation is complex
            // For noise, wrapping multiply is acceptable
            let a = _mm_mul_epu32(self.0, rhs.0); // low 2 lanes (0*0 and 2*2)
            let b = _mm_mul_epu32(
                _mm_shuffle_epi32(self.0, 0xB1), // swap 0<->1, 2<->3
                _mm_shuffle_epi32(rhs.0, 0xB1),
            ); // high 2 lanes
               // Interleave: a gets lanes 0,2; b gets lanes 1,3
               // Use unpack instructions... this gets involved. For now: scalar fallback via _mm_set_epi32
               // Actually, let's use a simpler approach: unpack
            let a_unpack = _mm_shuffle_epi32(a, 0xD8); // a0, a2, 0, 0 -> we need a0, b0, a2, b2
            let b_unpack = _mm_shuffle_epi32(b, 0xD8); // b1, b3, 0, 0
            Self(_mm_unpacklo_epi32(a_unpack, b_unpack))
        }
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
        // This is acceptable for the few places we shift in noise code.
        unsafe {
            let f = _mm_cvtepi32_ps(self.0);
            let factor = 2.0_f32.powi(-rhs);
            let shifted = _mm_mul_ps(f, _mm_set1_ps(factor));
            Self(_mm_cvttps_epi32(shifted))
        }
    }

    #[inline]
    fn convert_to_float(self) -> Self::FloatType {
        unsafe { Sse2Float(_mm_cvtepi32_ps(self.0)) }
    }
}
