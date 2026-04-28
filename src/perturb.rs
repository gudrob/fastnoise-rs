//! Domain warping / perturbation.
//!
//! Applies a secondary noise field to warp the input coordinates before
//! evaluating the primary noise. Perturbation functions call the raw
//! single-noise generators directly to avoid recursion through `generate_3d`.

use crate::settings::{NoiseType, PerturbType, Settings};
use crate::simd::{SimdFloat, SimdInt};

/// Perturb 3D coordinates only — returns the warped (x, y, z) without evaluating noise.
///
/// Used by the SIMD batch kernel to compute perturbed coordinates per-lane,
/// then feed them into the batched Value/Perlin kernel for speed.
pub fn perturb_coords<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    match settings.perturb_type {
        PerturbType::Gradient => perturb_gradient_3d::<F, I>(settings, x, y, z),
        PerturbType::GradientFractal => perturb_gradient_fractal_3d::<F, I>(settings, x, y, z),
        PerturbType::Normalise => perturb_normalise_3d::<F, I>(settings, x, y, z),
        PerturbType::GradientNormalise => perturb_gradient_normalise_3d::<F, I>(settings, x, y, z),
        PerturbType::GradientFractalNormalise => {
            perturb_gradient_normalise_fractal_3d::<F, I>(settings, x, y, z)
        }
        _ => (x, y, z),
    }
}

/// Perturb 3D coordinates and evaluate noise at the perturbed location.
/// Uses the `single_*_3d` functions directly to avoid infinite recursion.
pub fn perturb_3d<F: SimdFloat, I: SimdInt>(settings: &Settings, x: f32, y: f32, z: f32) -> f32 {
    let (nx, ny, nz) = match settings.perturb_type {
        PerturbType::Gradient => perturb_gradient_3d::<F, I>(settings, x, y, z),
        PerturbType::GradientFractal => perturb_gradient_fractal_3d::<F, I>(settings, x, y, z),
        PerturbType::Normalise => perturb_normalise_3d::<F, I>(settings, x, y, z),
        PerturbType::GradientNormalise => perturb_gradient_normalise_3d::<F, I>(settings, x, y, z),
        PerturbType::GradientFractalNormalise => {
            perturb_gradient_normalise_fractal_3d::<F, I>(settings, x, y, z)
        }
        _ => (x, y, z),
    };

    match settings.noise_type {
        NoiseType::Value | NoiseType::ValueFractal => {
            crate::noise::single_value_3d::<F, I>(settings.seed, nx, ny, nz)
        }
        NoiseType::Perlin | NoiseType::PerlinFractal => {
            crate::noise::single_perlin_3d::<F, I>(settings.seed, nx, ny, nz)
        }
        NoiseType::Simplex | NoiseType::SimplexFractal => {
            crate::noise::single_simplex_3d::<F, I>(settings.seed, nx, ny, nz)
        }
        NoiseType::Cubic | NoiseType::CubicFractal => {
            crate::noise::single_cubic_3d::<F, I>(settings.seed, nx, ny, nz)
        }
        NoiseType::WhiteNoise => {
            crate::noise::single_white_noise_3d::<F, I>(settings.seed, nx, ny, nz)
        }
        NoiseType::Cellular => crate::noise::single_cellular_3d::<F, I>(settings, nx, ny, nz),
    }
}

/// Evaluate a single noise sample (3D) for perturbation offsets — does NOT recurse.
/// This is a helper that calls the raw single-noise function based on settings.
fn perturb_sample_3d<F: SimdFloat, I: SimdInt>(settings: &Settings, x: f32, y: f32, z: f32) -> f32 {
    match settings.noise_type {
        NoiseType::Value | NoiseType::ValueFractal => {
            crate::noise::single_value_3d::<F, I>(settings.seed, x, y, z)
        }
        NoiseType::Perlin | NoiseType::PerlinFractal => {
            crate::noise::single_perlin_3d::<F, I>(settings.seed, x, y, z)
        }
        NoiseType::Simplex | NoiseType::SimplexFractal => {
            crate::noise::single_simplex_3d::<F, I>(settings.seed, x, y, z)
        }
        NoiseType::Cubic | NoiseType::CubicFractal => {
            crate::noise::single_cubic_3d::<F, I>(settings.seed, x, y, z)
        }
        NoiseType::WhiteNoise => {
            crate::noise::single_white_noise_3d::<F, I>(settings.seed, x, y, z)
        }
        NoiseType::Cellular => crate::noise::single_cellular_3d::<F, I>(settings, x, y, z),
    }
}

fn perturb_gradient_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    let amp = settings.perturb_amplitude;
    let x_off = perturb_sample_3d::<F, I>(settings, x + 0.1, y, z) * amp;
    let y_off = perturb_sample_3d::<F, I>(settings, x, y + 0.1, z) * amp;
    let z_off = perturb_sample_3d::<F, I>(settings, x, y, z + 0.1) * amp;
    (x + x_off, y + y_off, z + z_off)
}

fn perturb_gradient_fractal_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    let amp = settings.perturb_amplitude;
    let mut fractal = settings.clone();
    fractal.frequency = settings.perturb_frequency;
    let x_off = perturb_sample_3d::<F, I>(&fractal, x + 0.1, y, z) * amp;
    let y_off = perturb_sample_3d::<F, I>(&fractal, x, y + 0.1, z) * amp;
    let z_off = perturb_sample_3d::<F, I>(&fractal, x, y, z + 0.1) * amp;
    (x + x_off, y + y_off, z + z_off)
}

fn perturb_normalise_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    let amp = settings.perturb_amplitude;
    let nx = perturb_sample_3d::<F, I>(settings, x + 0.1, y, z);
    let ny = perturb_sample_3d::<F, I>(settings, x, y + 0.1, z);
    let nz = perturb_sample_3d::<F, I>(settings, x, y, z + 0.1);
    let len = (nx * nx + ny * ny + nz * nz).sqrt().max(0.0001);
    (x + nx / len * amp, y + ny / len * amp, z + nz / len * amp)
}

fn perturb_gradient_normalise_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    let amp = settings.perturb_amplitude;
    let (px, py, pz) = perturb_gradient_3d::<F, I>(settings, x, y, z);
    let len = (px * px + py * py + pz * pz).sqrt().max(0.0001);
    (x + px / len * amp, y + py / len * amp, z + pz / len * amp)
}

fn perturb_gradient_normalise_fractal_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    let amp = settings.perturb_amplitude;
    let (px, py, pz) = perturb_gradient_fractal_3d::<F, I>(settings, x, y, z);
    let len = (px * px + py * py + pz * pz).sqrt().max(0.0001);
    (x + px / len * amp, y + py / len * amp, z + pz / len * amp)
}
