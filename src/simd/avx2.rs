//! AVX2 + FMA SIMD implementation (8-lane, 256-bit).
//!
//! Uses `std::arch::x86_64` intrinsics. Provides native FMA,
//! blend, and integer multiply support.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{SimdFloat, SimdInt};

/// AVX2 float vector (8 lanes, __m256).
#[derive(Clone, Copy, Debug)]
pub struct Avx2Float(__m256);

/// AVX2 integer vector (8 lanes, __m256i).
#[derive(Clone, Copy, Debug)]
pub struct Avx2Int(__m256i);

impl SimdFloat for Avx2Float {
    const VECTOR_SIZE: usize = 8;

    #[inline]
    unsafe fn undefined() -> Self {
        Self(_mm256_undefined_ps())
    }

    #[inline]
    fn set(value: f32) -> Self {
        unsafe { Self(_mm256_set1_ps(value)) }
    }

    #[inline]
    fn set1() -> Self {
        unsafe { Self(_mm256_set1_ps(1.0)) }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        Self(_mm256_load_ps(ptr))
    }

    #[inline]
    unsafe fn store(ptr: *mut f32, value: Self) {
        _mm256_store_ps(ptr, value.0);
    }

    // --- Arithmetic ---

    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_add_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_sub_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_mul_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // AVX2 has native FMA via _mm256_fmadd_ps (FMA3)
        unsafe { Self(_mm256_fmadd_ps(self.0, b.0, c.0)) }
    }

    #[inline]
    fn mul_sub(self, b: Self, c: Self) -> Self {
        unsafe { Self(_mm256_fmsub_ps(self.0, b.0, c.0)) }
    }

    // --- Comparison ---

    #[inline]
    fn min(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_min_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_max_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn less_than(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_cmp_ps(self.0, rhs.0, _CMP_LT_OQ)) }
    }

    #[inline]
    fn blendv(self, b: Self, mask: Self) -> Self {
        // AVX2 native blend: (mask & self) | (!mask & b)
        unsafe { Self(_mm256_blendv_ps(b.0, self.0, mask.0)) }
    }

    // --- Math ---

    #[inline]
    fn abs(self) -> Self {
        unsafe {
            let sign_mask = _mm256_set1_ps(-0.0_f32);
            Self(_mm256_andnot_ps(sign_mask, self.0))
        }
    }

    #[inline]
    fn inv_sqrt(self) -> Self {
        unsafe {
            let approx = _mm256_rsqrt_ps(self.0);
            // Newton-Raphson refinement
            let half = _mm256_set1_ps(0.5);
            let three_halfs = _mm256_set1_ps(1.5);
            let tmp = _mm256_mul_ps(_mm256_mul_ps(half, self.0), _mm256_mul_ps(approx, approx));
            Self(_mm256_mul_ps(approx, _mm256_sub_ps(three_halfs, tmp)))
        }
    }

    #[inline]
    fn floor(self) -> Self {
        // AVX2 has no native floor. Emulate via truncation:
        // 1. truncate via cvttps_epi32 + cvtepi32_ps
        // 2. if original > truncated (negative fraction), subtract 1
        unsafe {
            let int_val = _mm256_cvttps_epi32(self.0);
            let float_val = _mm256_cvtepi32_ps(int_val);
            let mask = _mm256_cmp_ps(self.0, float_val, _CMP_GT_OQ);
            let correction = _mm256_and_ps(mask, _mm256_set1_ps(1.0));
            Self(_mm256_add_ps(float_val, correction))
        }
    }
}

impl SimdInt for Avx2Int {
    const VECTOR_SIZE: usize = 8;
    type FloatType = Avx2Float;

    #[inline]
    fn set(value: i32) -> Self {
        unsafe { Self(_mm256_set1_epi32(value)) }
    }

    #[inline]
    fn set1() -> Self {
        unsafe { Self(_mm256_set1_epi32(1)) }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_add_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_sub_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // AVX2 has _mm256_mullo_epi32
        unsafe { Self(_mm256_mullo_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn and(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_and_si256(self.0, rhs.0)) }
    }

    #[inline]
    fn xor(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_xor_si256(self.0, rhs.0)) }
    }

    #[inline]
    fn or(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_or_si256(self.0, rhs.0)) }
    }

    #[inline]
    fn shift_right(self, rhs: i32) -> Self {
        // _mm256_srai_epi32 requires a compile-time constant in older Rust.
        // Since this module only compiles with has_avx2 (target_feature enabled),
        // the intrinsic is available. Use it via inline as workaround if needed.
        unsafe { Self(_mm256_srai_epi32(self.0, rhs)) }
    }

    #[inline]
    fn convert_to_float(self) -> Self::FloatType {
        unsafe { Avx2Float(_mm256_cvtepi32_ps(self.0)) }
    }
}
