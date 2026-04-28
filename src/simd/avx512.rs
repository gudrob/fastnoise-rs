//! AVX-512F SIMD implementation (16-lane, 512-bit).
//!
//! Uses `std::arch::x86_64` intrinsics with EVEX-encoded instructions.
//! Provides native FMA, blend, floor, and integer multiply.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{SimdFloat, SimdInt};

/// AVX-512F float vector (16 lanes, __m512).
#[derive(Clone, Copy, Debug)]
pub struct Avx512Float(__m512);

/// AVX-512F integer vector (16 lanes, __m512i).
#[derive(Clone, Copy, Debug)]
pub struct Avx512Int(__m512i);

impl SimdFloat for Avx512Float {
    const VECTOR_SIZE: usize = 16;

    #[inline]
    unsafe fn undefined() -> Self {
        Self(_mm512_undefined_ps())
    }

    #[inline]
    fn set(value: f32) -> Self {
        unsafe { Self(_mm512_set1_ps(value)) }
    }

    #[inline]
    fn set1() -> Self {
        unsafe { Self(_mm512_set1_ps(1.0)) }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        Self(_mm512_load_ps(ptr))
    }

    #[inline]
    unsafe fn store(ptr: *mut f32, value: Self) {
        _mm512_store_ps(ptr, value.0);
    }

    // --- Arithmetic ---

    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_add_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_sub_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_mul_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // AVX-512F native FMA
        unsafe { Self(_mm512_fmadd_ps(self.0, b.0, c.0)) }
    }

    #[inline]
    fn mul_sub(self, b: Self, c: Self) -> Self {
        unsafe { Self(_mm512_fmsub_ps(self.0, b.0, c.0)) }
    }

    // --- Comparison ---

    #[inline]
    fn min(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_min_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_max_ps(self.0, rhs.0)) }
    }

    #[inline]
    fn less_than(self, rhs: Self) -> Self {
        // AVX-512F uses k-mask from _mm512_cmp_ps_mask
        // We need to convert the k-mask to an __m512 all-ones/all-zeros mask.
        unsafe {
            let kmask = _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LT_OQ);
            // _mm512_maskz_mov_ps: zero-masked move with kmask → all-1s where kmask bit is set
            // Generate a vector of all-ones.
            let ones = _mm512_set1_ps(f32::from_bits(u32::MAX));
            Self(_mm512_maskz_mov_ps(kmask, ones))
        }
    }

    #[inline]
    fn blendv(self, b: Self, mask: Self) -> Self {
        // AVX-512F: use k-mask from the mask vector (bit 31 of each lane).
        unsafe {
            let kmask = _mm512_movepi32_mask(_mm512_castps_si512(mask.0));
            Self(_mm512_mask_blend_ps(kmask, b.0, self.0))
        }
    }

    // --- Math ---

    #[inline]
    fn abs(self) -> Self {
        unsafe {
            let sign_mask = _mm512_set1_ps(-0.0_f32);
            Self(_mm512_andnot_ps(sign_mask, self.0))
        }
    }

    #[inline]
    fn inv_sqrt(self) -> Self {
        unsafe {
            let approx = _mm512_rsqrt14_ps(self.0);
            // Newton-Raphson refinement
            let half = _mm512_set1_ps(0.5);
            let three_halfs = _mm512_set1_ps(1.5);
            let tmp = _mm512_mul_ps(
                _mm512_mul_ps(half, self.0),
                _mm512_mul_ps(approx, approx),
            );
            Self(_mm512_mul_ps(approx, _mm512_sub_ps(three_halfs, tmp)))
        }
    }

    #[inline]
    fn floor(self) -> Self {
        // AVX-512F has native floor via _mm512_floor_ps
        unsafe { Self(_mm512_floor_ps(self.0)) }
    }
}

impl SimdInt for Avx512Int {
    const VECTOR_SIZE: usize = 16;
    type FloatType = Avx512Float;

    #[inline]
    fn set(value: i32) -> Self {
        unsafe { Self(_mm512_set1_epi32(value)) }
    }

    #[inline]
    fn set1() -> Self {
        unsafe { Self(_mm512_set1_epi32(1)) }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_add_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_sub_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // AVX-512F has _mm512_mullo_epi32
        unsafe { Self(_mm512_mullo_epi32(self.0, rhs.0)) }
    }

    #[inline]
    fn and(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, rhs.0)) }
    }

    #[inline]
    fn xor(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_xor_si512(self.0, rhs.0)) }
    }

    #[inline]
    fn or(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_or_si512(self.0, rhs.0)) }
    }

    #[inline]
    fn shift_right(self, rhs: i32) -> Self {
        // _mm512_srai_epi32 requires compile-time constant shift in some Rust versions.
        // For AVX-512F with target_feature enabled this should work.
        unsafe { Self(_mm512_srai_epi32(self.0, rhs)) }
    }

    #[inline]
    fn convert_to_float(self) -> Self::FloatType {
        unsafe { Avx512Float(_mm512_cvtepi32_ps(self.0)) }
    }
}