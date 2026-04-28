//! Fractal noise overlay (FBM, Billow, RigidMulti).
//!
//! Applies multiple octaves of a base noise function, weighted by
//! lacunarity and gain, to produce fractal noise.

use crate::settings::{FractalType, Settings};
use crate::simd::{SimdFloat, SimdInt};

/// Generate 3D fractal noise by combining multiple octaves of a base noise function.
pub fn fractal_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_noise: fn(i32, f32, f32, f32) -> f32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    match settings.fractal_type {
        FractalType::FBM => fbm_3d::<F, I>(settings, base_noise, x, y, z),
        FractalType::Billow => billow_3d::<F, I>(settings, base_noise, x, y, z),
        FractalType::RigidMulti => rigid_multi_3d::<F, I>(settings, base_noise, x, y, z),
    }
}

/// Generate 2D fractal noise by combining multiple octaves.
pub fn fractal_2d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_noise: fn(i32, f32, f32) -> f32,
    x: f32,
    y: f32,
) -> f32 {
    match settings.fractal_type {
        FractalType::FBM => fbm_2d::<F, I>(settings, base_noise, x, y),
        FractalType::Billow => billow_2d::<F, I>(settings, base_noise, x, y),
        FractalType::RigidMulti => rigid_multi_2d::<F, I>(settings, base_noise, x, y),
    }
}

// ============================================================================
// Fractal Brownian Motion (FBM)
// ============================================================================

pub fn fbm_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_noise: fn(i32, f32, f32, f32) -> f32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;

    for i in 0..settings.octaves.max(1) {
        let noise = base_noise(settings.seed + i, x * freq, y * freq, z * freq);
        sum += noise * amp;
        max += amp;
        freq *= settings.lacunarity;
        amp *= settings.gain;
    }

    sum / max
}

pub fn fbm_2d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_noise: fn(i32, f32, f32) -> f32,
    x: f32,
    y: f32,
) -> f32 {
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;

    for i in 0..settings.octaves.max(1) {
        let noise = base_noise(settings.seed + i, x * freq, y * freq);
        sum += noise * amp;
        max += amp;
        freq *= settings.lacunarity;
        amp *= settings.gain;
    }

    sum / max
}

// ============================================================================
// Billow (abs-based FBM)
// ============================================================================

pub fn billow_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_noise: fn(i32, f32, f32, f32) -> f32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;

    for i in 0..settings.octaves.max(1) {
        let noise = base_noise(settings.seed + i, x * freq, y * freq, z * freq);
        sum += (noise.abs() * 2.0 - 1.0) * amp;
        max += amp;
        freq *= settings.lacunarity;
        amp *= settings.gain;
    }

    sum / max
}

pub fn billow_2d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_noise: fn(i32, f32, f32) -> f32,
    x: f32,
    y: f32,
) -> f32 {
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;

    for i in 0..settings.octaves.max(1) {
        let noise = base_noise(settings.seed + i, x * freq, y * freq);
        sum += (noise.abs() * 2.0 - 1.0) * amp;
        max += amp;
        freq *= settings.lacunarity;
        amp *= settings.gain;
    }

    sum / max
}

// ============================================================================
// Rigid Multi-Fractal
// ============================================================================

pub fn rigid_multi_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_noise: fn(i32, f32, f32, f32) -> f32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;

    for i in 0..settings.octaves.max(1) {
        let noise = base_noise(settings.seed + i, x * freq, y * freq, z * freq);
        sum += (1.0 - noise.abs()) * amp;
        max += amp;
        freq *= settings.lacunarity;
        amp *= settings.gain;
    }

    (sum / max) * 2.0 - 1.0
}

pub fn rigid_multi_2d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_noise: fn(i32, f32, f32) -> f32,
    x: f32,
    y: f32,
) -> f32 {
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;

    for i in 0..settings.octaves.max(1) {
        let noise = base_noise(settings.seed + i, x * freq, y * freq);
        sum += (1.0 - noise.abs()) * amp;
        max += amp;
        freq *= settings.lacunarity;
        amp *= settings.gain;
    }

    (sum / max) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar::{ScalarFloat, ScalarInt};

    fn test_noise_3d(_seed: i32, x: f32, y: f32, z: f32) -> f32 {
        // Simple value-like noise for testing
        (x * 0.5 + y * 0.3 + z * 0.2).sin()
    }

    fn test_noise_2d(_seed: i32, x: f32, y: f32) -> f32 {
        (x * 0.5 + y * 0.3).sin()
    }

    #[test]
    fn test_fbm_3d_range() {
        let settings = Settings::new(42);
        for i in 0..5 {
            for j in 0..5 {
                let v = fbm_3d::<ScalarFloat, ScalarInt>(
                    &settings,
                    test_noise_3d,
                    i as f32 * 0.1,
                    j as f32 * 0.1,
                    0.5,
                );
                assert!(v.is_finite(), "fbm value should be finite: {v}");
            }
        }
    }

    #[test]
    fn test_billow_2d_range() {
        let settings = Settings::new(42);
        for i in 0..5 {
            for j in 0..5 {
                let v = billow_2d::<ScalarFloat, ScalarInt>(
                    &settings,
                    test_noise_2d,
                    i as f32 * 0.1,
                    j as f32 * 0.1,
                );
                assert!(v.is_finite(), "billow value should be finite: {v}");
            }
        }
    }

    #[test]
    fn test_rigid_multi_3d_range() {
        let settings = Settings::new(42);
        let v = rigid_multi_3d::<ScalarFloat, ScalarInt>(&settings, test_noise_3d, 0.5, 0.5, 0.5);
        assert!(v.is_finite(), "rigid multi value should be finite: {v}");
        assert!(
            (-1.0..=1.0).contains(&v),
            "rigid multi should be in [-1,1]: {v}"
        );
    }
}
