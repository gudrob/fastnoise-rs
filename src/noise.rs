//! Core noise generation functions.
//!
//! Each noise type supports:
//! - Single-sample evaluation (`get_noise_3d`, `get_noise_2d`)
//! - Grid generation (`fill_noise_set_3d`, `fill_noise_set_2d`)
//!
//! All functions are generic over `SimdFloat + SimdInt` SIMD types.

use crate::hash;
use crate::settings::{CellularDistanceFunction, NoiseType, Settings};
use crate::simd::{SimdFloat, SimdInt};

// ============================================================================
// Public API – Noise generation using Settings
// ============================================================================

/// Generate 3D noise using the given settings.
pub fn generate_3d<F: SimdFloat, I: SimdInt>(settings: &Settings, x: f32, y: f32, z: f32) -> f32 {
    let x = x * settings.x_scale * settings.frequency;
    let y = y * settings.y_scale * settings.frequency;
    let z = z * settings.z_scale * settings.frequency;

    let value = match settings.noise_type {
        NoiseType::Value => single_value_3d::<F, I>(settings.seed, x, y, z),
        NoiseType::ValueFractal => crate::fractal::fractal_3d::<F, I>(
            settings,
            |seed, x, y, z| single_value_3d::<F, I>(seed, x, y, z),
            x,
            y,
            z,
        ),
        NoiseType::Perlin => single_perlin_3d::<F, I>(settings.seed, x, y, z),
        NoiseType::PerlinFractal => crate::fractal::fractal_3d::<F, I>(
            settings,
            |seed, x, y, z| single_perlin_3d::<F, I>(seed, x, y, z),
            x,
            y,
            z,
        ),
        NoiseType::Simplex => single_simplex_3d::<F, I>(settings.seed, x, y, z),
        NoiseType::SimplexFractal => crate::fractal::fractal_3d::<F, I>(
            settings,
            |seed, x, y, z| single_simplex_3d::<F, I>(seed, x, y, z),
            x,
            y,
            z,
        ),
        NoiseType::Cellular => single_cellular_3d::<F, I>(settings, x, y, z),
        NoiseType::WhiteNoise => single_white_noise_3d::<F, I>(settings.seed, x, y, z),
        NoiseType::Cubic => single_cubic_3d::<F, I>(settings.seed, x, y, z),
        NoiseType::CubicFractal => crate::fractal::fractal_3d::<F, I>(
            settings,
            |seed, x, y, z| single_cubic_3d::<F, I>(seed, x, y, z),
            x,
            y,
            z,
        ),
    };

    // Apply perturb if configured
    if settings.perturb_type != crate::settings::PerturbType::None {
        crate::perturb::perturb_3d::<F, I>(settings, x, y, z)
    } else {
        value
    }
}

/// Generate 2D noise using the given settings.
pub fn generate_2d<F: SimdFloat, I: SimdInt>(settings: &Settings, x: f32, y: f32) -> f32 {
    let x = x * settings.x_scale * settings.frequency;
    let y = y * settings.y_scale * settings.frequency;

    let value = match settings.noise_type {
        NoiseType::Value => single_value_2d::<F, I>(settings.seed, x, y),
        NoiseType::ValueFractal => crate::fractal::fractal_2d::<F, I>(
            settings,
            |seed, x, y| single_value_2d::<F, I>(seed, x, y),
            x,
            y,
        ),
        NoiseType::Perlin => single_perlin_2d::<F, I>(settings.seed, x, y),
        NoiseType::PerlinFractal => crate::fractal::fractal_2d::<F, I>(
            settings,
            |seed, x, y| single_perlin_2d::<F, I>(seed, x, y),
            x,
            y,
        ),
        NoiseType::Simplex => single_simplex_2d::<F, I>(settings.seed, x, y),
        NoiseType::SimplexFractal => crate::fractal::fractal_2d::<F, I>(
            settings,
            |seed, x, y| single_simplex_2d::<F, I>(seed, x, y),
            x,
            y,
        ),
        NoiseType::Cellular => single_cellular_2d::<F, I>(settings, x, y),
        NoiseType::WhiteNoise => single_white_noise_2d::<F, I>(settings.seed, x, y),
        NoiseType::Cubic => single_cubic_2d::<F, I>(settings.seed, x, y),
        NoiseType::CubicFractal => crate::fractal::fractal_2d::<F, I>(
            settings,
            |seed, x, y| single_cubic_2d::<F, I>(seed, x, y),
            x,
            y,
        ),
    };

    value
}

// ============================================================================
// Single-sample Value Noise
// ============================================================================

/// Value noise 3D – single sample.
pub(crate) fn single_value_3d<F: SimdFloat, I: SimdInt>(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let fz = z - iz as f32;

    // Hash the 8 corners
    let v000 = hash::val_coord_f32(seed, ix, iy, iz);
    let v100 = hash::val_coord_f32(seed, ix + 1, iy, iz);
    let v010 = hash::val_coord_f32(seed, ix, iy + 1, iz);
    let v110 = hash::val_coord_f32(seed, ix + 1, iy + 1, iz);
    let v001 = hash::val_coord_f32(seed, ix, iy, iz + 1);
    let v101 = hash::val_coord_f32(seed, ix + 1, iy, iz + 1);
    let v011 = hash::val_coord_f32(seed, ix, iy + 1, iz + 1);
    let v111 = hash::val_coord_f32(seed, ix + 1, iy + 1, iz + 1);

    // Smoothstep interpolation weights
    let sx = smoothstep(fx);
    let sy = smoothstep(fy);
    let sz = smoothstep(fz);

    // Trilinear interpolation
    let i0 = lerp(v000, v100, sx);
    let i1 = lerp(v010, v110, sx);
    let i2 = lerp(v001, v101, sx);
    let i3 = lerp(v011, v111, sx);

    let j0 = lerp(i0, i1, sy);
    let j1 = lerp(i2, i3, sy);

    lerp(j0, j1, sz)
}

/// Value noise 2D – single sample.
fn single_value_2d<F: SimdFloat, I: SimdInt>(seed: i32, x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let v00 = hash::val_coord_2d_f32(seed, ix, iy);
    let v10 = hash::val_coord_2d_f32(seed, ix + 1, iy);
    let v01 = hash::val_coord_2d_f32(seed, ix, iy + 1);
    let v11 = hash::val_coord_2d_f32(seed, ix + 1, iy + 1);

    let sx = smoothstep(fx);
    let sy = smoothstep(fy);

    let i0 = lerp(v00, v10, sx);
    let i1 = lerp(v01, v11, sx);

    lerp(i0, i1, sy)
}

// ============================================================================
// Single-sample Perlin Noise
// ============================================================================

/// Perlin noise 3D – single sample.
pub(crate) fn single_perlin_3d<F: SimdFloat, I: SimdInt>(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let fz = z - iz as f32;

    // Gradient vectors for 8 corners (from 12-direction cube)
    let g000 = grad_coord_3d(seed, ix, iy, iz, fx, fy, fz);
    let g100 = grad_coord_3d(seed, ix + 1, iy, iz, fx - 1.0, fy, fz);
    let g010 = grad_coord_3d(seed, ix, iy + 1, iz, fx, fy - 1.0, fz);
    let g110 = grad_coord_3d(seed, ix + 1, iy + 1, iz, fx - 1.0, fy - 1.0, fz);
    let g001 = grad_coord_3d(seed, ix, iy, iz + 1, fx, fy, fz - 1.0);
    let g101 = grad_coord_3d(seed, ix + 1, iy, iz + 1, fx - 1.0, fy, fz - 1.0);
    let g011 = grad_coord_3d(seed, ix, iy + 1, iz + 1, fx, fy - 1.0, fz - 1.0);
    let g111 = grad_coord_3d(seed, ix + 1, iy + 1, iz + 1, fx - 1.0, fy - 1.0, fz - 1.0);

    let sx = smoothstep(fx);
    let sy = smoothstep(fy);
    let sz = smoothstep(fz);

    let i0 = lerp(g000, g100, sx);
    let i1 = lerp(g010, g110, sx);
    let i2 = lerp(g001, g101, sx);
    let i3 = lerp(g011, g111, sx);

    let j0 = lerp(i0, i1, sy);
    let j1 = lerp(i2, i3, sy);

    lerp(j0, j1, sz)
}

/// Perlin noise 2D – single sample.
fn single_perlin_2d<F: SimdFloat, I: SimdInt>(seed: i32, x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let g00 = grad_coord_2d(seed, ix, iy, fx, fy);
    let g10 = grad_coord_2d(seed, ix + 1, iy, fx - 1.0, fy);
    let g01 = grad_coord_2d(seed, ix, iy + 1, fx, fy - 1.0);
    let g11 = grad_coord_2d(seed, ix + 1, iy + 1, fx - 1.0, fy - 1.0);

    let sx = smoothstep(fx);
    let sy = smoothstep(fy);

    let i0 = lerp(g00, g10, sx);
    let i1 = lerp(g01, g11, sx);

    lerp(i0, i1, sy)
}

// ============================================================================
// Single-sample Simplex Noise
// ============================================================================

/// Simplex noise 3D – single sample.
pub(crate) fn single_simplex_3d<F: SimdFloat, I: SimdInt>(
    seed: i32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    // Standard 3D simplex skew/unskew from the original paper:
    // F3 = 1/3, G3 = 1/6
    const SKEW_3D: f32 = 1.0 / 3.0;
    const UNSKEW_3D: f32 = 1.0 / 6.0;
    let s = (x + y + z) * SKEW_3D;
    let ix = (x + s).floor() as i32;
    let iy = (y + s).floor() as i32;
    let iz = (z + s).floor() as i32;

    let t = (ix + iy + iz) as f32 * UNSKEW_3D;
    let x0 = x - (ix as f32 - t);
    let y0 = y - (iy as f32 - t);
    let z0 = z - (iz as f32 - t);

    // Determine simplex ordering
    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0 {
            (1, 0, 0, 1, 1, 0) // X Y Z
        } else if x0 >= z0 {
            (1, 0, 0, 1, 0, 1) // X Z Y
        } else {
            (0, 0, 1, 1, 0, 1) // Z X Y
        }
    } else {
        if y0 < z0 {
            (0, 0, 1, 0, 1, 1) // Z Y X
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1) // Y Z X
        } else {
            (0, 1, 0, 1, 1, 0) // Y X Z
        }
    };

    let x1 = x0 - i1 as f32 + UNSKEW_3D;
    let y1 = y0 - j1 as f32 + UNSKEW_3D;
    let z1 = z0 - k1 as f32 + UNSKEW_3D;
    let x2 = x0 - i2 as f32 + 2.0 * UNSKEW_3D;
    let y2 = y0 - j2 as f32 + 2.0 * UNSKEW_3D;
    let z2 = z0 - k2 as f32 + 2.0 * UNSKEW_3D;
    let x3 = x0 - 1.0 + 3.0 * UNSKEW_3D;
    let y3 = y0 - 1.0 + 3.0 * UNSKEW_3D;
    let z3 = z0 - 1.0 + 3.0 * UNSKEW_3D;

    let ix = ix;
    let iy = iy;
    let iz = iz;

    let mut n = 0.0_f32;

    // Contribution from each corner
    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if t0 > 0.0 {
        let t0 = t0 * t0;
        n += t0 * t0 * grad_coord_3d(seed, ix, iy, iz, x0, y0, z0);
    }

    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if t1 > 0.0 {
        let t1 = t1 * t1;
        n += t1 * t1 * grad_coord_3d(seed, ix + i1, iy + j1, iz + k1, x1, y1, z1);
    }

    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if t2 > 0.0 {
        let t2 = t2 * t2;
        n += t2 * t2 * grad_coord_3d(seed, ix + i2, iy + j2, iz + k2, x2, y2, z2);
    }

    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if t3 > 0.0 {
        let t3 = t3 * t3;
        n += t3 * t3 * grad_coord_3d(seed, ix + 1, iy + 1, iz + 1, x3, y3, z3);
    }

    // Scale to [-1, 1]
    n * 32.0
}

/// Simplex noise 2D – single sample.
fn single_simplex_2d<F: SimdFloat, I: SimdInt>(seed: i32, x: f32, y: f32) -> f32 {
    const SKEW_2D: f32 = (1.732_050_8 - 1.0) / 2.0; // (sqrt(3) - 1) / 2 ≈ 0.366
    const UNSKEW_2D: f32 = (3.0 - 1.732_050_8) / 6.0; // (3 - sqrt(3)) / 6 ≈ 0.211

    let s = (x + y) * SKEW_2D;
    let ix = (x + s).floor() as i32;
    let iy = (y + s).floor() as i32;

    let t = (ix + iy) as f32 * UNSKEW_2D;
    let x0 = x - (ix as f32 - t);
    let y0 = y - (iy as f32 - t);

    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

    let x1 = x0 - i1 as f32 + UNSKEW_2D;
    let y1 = y0 - j1 as f32 + UNSKEW_2D;
    let x2 = x0 - 1.0 + 2.0 * UNSKEW_2D;
    let y2 = y0 - 1.0 + 2.0 * UNSKEW_2D;

    let mut n = 0.0_f32;

    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if t0 > 0.0 {
        let t0 = t0 * t0;
        n += t0 * t0 * grad_coord_2d(seed, ix, iy, x0, y0);
    }

    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if t1 > 0.0 {
        let t1 = t1 * t1;
        n += t1 * t1 * grad_coord_2d(seed, ix + i1, iy + j1, x1, y1);
    }

    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if t2 > 0.0 {
        let t2 = t2 * t2;
        n += t2 * t2 * grad_coord_2d(seed, ix + 1, iy + 1, x2, y2);
    }

    n * 70.0
}

// ============================================================================
// Single-sample Cellular Noise
// ============================================================================

/// Cellular noise 3D – single sample.
pub(crate) fn single_cellular_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let _fx = x - ix as f32;
    let _fy = y - iy as f32;
    let _fz = z - iz as f32;

    let mut f1 = f32::MAX;
    let mut f2 = f32::MAX;
    let mut closest_hash = 0_i32;

    let jitter = settings.cellular_jitter;

    // Search 3x3x3 neighborhood
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let nx = ix + dx;
                let ny = iy + dy;
                let nz = iz + dz;
                let h = hash::hash_hb(settings.seed, nx, ny, nz);

                let cell_x = nx as f32 + jitter * ((h & 0xFF) as f32 / 255.0 - 0.5);
                let cell_y = ny as f32 + jitter * (((h >> 8) & 0xFF) as f32 / 255.0 - 0.5);
                let cell_z = nz as f32 + jitter * (((h >> 16) & 0xFF) as f32 / 255.0 - 0.5);

                let dx_f = x - cell_x;
                let dy_f = y - cell_y;
                let dz_f = z - cell_z;

                let dist = match settings.cellular_distance_function {
                    CellularDistanceFunction::Euclidean => {
                        (dx_f * dx_f + dy_f * dy_f + dz_f * dz_f).sqrt()
                    }
                    CellularDistanceFunction::Manhattan => dx_f.abs() + dy_f.abs() + dz_f.abs(),
                    CellularDistanceFunction::Natural => {
                        (dx_f * dx_f + dy_f * dy_f + dz_f * dz_f).sqrt()
                            + dx_f.abs()
                            + dy_f.abs()
                            + dz_f.abs()
                    }
                };

                if dist < f1 {
                    f2 = f1;
                    f1 = dist;
                    closest_hash = h;
                } else if dist < f2 {
                    f2 = dist;
                }
            }
        }
    }

    match settings.cellular_return_type {
        crate::settings::CellularReturnType::CellValue => {
            (closest_hash & 0x3FF) as f32 / 1000.0 - 1.0
        }
        crate::settings::CellularReturnType::Distance => f1 - 1.0,
        crate::settings::CellularReturnType::Distance2 => f2 - 1.0,
        crate::settings::CellularReturnType::Distance2Add => (f2 + f1) * 0.5 - 1.0,
        crate::settings::CellularReturnType::Distance2Sub => (f2 - f1) - 1.0,
        crate::settings::CellularReturnType::Distance2Mul => (f2 * f1) * 0.5 - 1.0,
        crate::settings::CellularReturnType::Distance2Div => (f1 / (f2 + 0.000_001)) - 1.0,
        crate::settings::CellularReturnType::NoiseLookup => {
            // Look up noise at closest cell point
            let cx = ix as f32 + jitter * ((closest_hash & 0xFF) as f32 / 255.0 - 0.5);
            let cy = iy as f32 + jitter * (((closest_hash >> 8) & 0xFF) as f32 / 255.0 - 0.5);
            let cz = iz as f32 + jitter * (((closest_hash >> 16) & 0xFF) as f32 / 255.0 - 0.5);
            single_value_3d::<F, I>(
                settings.seed,
                cx * settings.cellular_noise_lookup_frequency,
                cy * settings.cellular_noise_lookup_frequency,
                cz * settings.cellular_noise_lookup_frequency,
            )
        }
        crate::settings::CellularReturnType::Distance2Cave => (f2 - f1).clamp(0.0, 1.0) * 2.0 - 1.0,
    }
}

/// Cellular noise 2D – single sample.
fn single_cellular_2d<F: SimdFloat, I: SimdInt>(settings: &Settings, x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let _fx = x - ix as f32;
    let _fy = y - iy as f32;

    let mut f1 = f32::MAX;
    let mut f2 = f32::MAX;
    let mut closest_hash = 0_i32;

    let jitter = settings.cellular_jitter;

    for dx in -1..=1 {
        for dy in -1..=1 {
            let nx = ix + dx;
            let ny = iy + dy;
            let h = hash::hash_hb(settings.seed, nx, ny, 0);

            let cell_x = nx as f32 + jitter * ((h & 0xFF) as f32 / 255.0 - 0.5);
            let cell_y = ny as f32 + jitter * (((h >> 8) & 0xFF) as f32 / 255.0 - 0.5);

            let dx_f = x - cell_x;
            let dy_f = y - cell_y;

            let dist = match settings.cellular_distance_function {
                CellularDistanceFunction::Euclidean => (dx_f * dx_f + dy_f * dy_f).sqrt(),
                CellularDistanceFunction::Manhattan => dx_f.abs() + dy_f.abs(),
                CellularDistanceFunction::Natural => {
                    (dx_f * dx_f + dy_f * dy_f).sqrt() + dx_f.abs() + dy_f.abs()
                }
            };

            if dist < f1 {
                f2 = f1;
                f1 = dist;
                closest_hash = h;
            } else if dist < f2 {
                f2 = dist;
            }
        }
    }

    match settings.cellular_return_type {
        crate::settings::CellularReturnType::CellValue => {
            (closest_hash & 0x3FF) as f32 / 1000.0 - 1.0
        }
        crate::settings::CellularReturnType::Distance => f1 - 1.0,
        crate::settings::CellularReturnType::Distance2 => f2 - 1.0,
        crate::settings::CellularReturnType::Distance2Add => (f2 + f1) * 0.5 - 1.0,
        crate::settings::CellularReturnType::Distance2Sub => (f2 - f1) - 1.0,
        crate::settings::CellularReturnType::Distance2Mul => (f2 * f1) * 0.5 - 1.0,
        crate::settings::CellularReturnType::Distance2Div => (f1 / (f2 + 0.000_001)) - 1.0,
        crate::settings::CellularReturnType::NoiseLookup => {
            let cx = ix as f32 + jitter * ((closest_hash & 0xFF) as f32 / 255.0 - 0.5);
            let cy = iy as f32 + jitter * (((closest_hash >> 8) & 0xFF) as f32 / 255.0 - 0.5);
            single_value_2d::<F, I>(
                settings.seed,
                cx * settings.cellular_noise_lookup_frequency,
                cy * settings.cellular_noise_lookup_frequency,
            )
        }
        crate::settings::CellularReturnType::Distance2Cave => (f2 - f1).clamp(0.0, 1.0) * 2.0 - 1.0,
    }
}

// ============================================================================
// Single-sample White Noise
// ============================================================================

pub(crate) fn single_white_noise_3d<F: SimdFloat, I: SimdInt>(
    seed: i32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    hash::val_coord_f32(seed, x.floor() as i32, y.floor() as i32, z.floor() as i32)
}

fn single_white_noise_2d<F: SimdFloat, I: SimdInt>(seed: i32, x: f32, y: f32) -> f32 {
    hash::val_coord_2d_f32(seed, x.floor() as i32, y.floor() as i32)
}

// ============================================================================
// Single-sample Cubic Noise
// ============================================================================

/// Cubic noise 3D – Catmull-Rom style interpolation over a 4x4x4 grid.
pub(crate) fn single_cubic_3d<F: SimdFloat, I: SimdInt>(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let fz = z - iz as f32;

    let mut vals = [[[0.0_f32; 4]; 4]; 4];
    for dx in 0..4 {
        for dy in 0..4 {
            for dz in 0..4 {
                vals[dx][dy][dz] = hash::val_coord_f32(
                    seed,
                    ix + dx as i32 - 1,
                    iy + dy as i32 - 1,
                    iz + dz as i32 - 1,
                );
            }
        }
    }

    cubic_interp_3d(&vals, fx, fy, fz)
}

/// Cubic noise 2D – Catmull-Rom style interpolation over a 4x4 grid.
fn single_cubic_2d<F: SimdFloat, I: SimdInt>(seed: i32, x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let mut vals = [[0.0_f32; 4]; 4];
    for dx in 0..4 {
        for dy in 0..4 {
            vals[dx][dy] = hash::val_coord_2d_f32(seed, ix + dx as i32 - 1, iy + dy as i32 - 1);
        }
    }

    cubic_interp_2d(&vals, fx, fy)
}

// ============================================================================
// Interpolation Helpers
// ============================================================================

/// Smoothstep (Hermite): 3t^2 - 2t^3
#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Catmull-Rom spline weight.
#[inline]
fn cubic_hermite(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    a * (-0.5 * t3 + t2 - 0.5 * t)
        + b * (1.5 * t3 - 2.5 * t2 + 1.0)
        + c * (-1.5 * t3 + 2.0 * t2 + 0.5 * t)
        + d * (0.5 * t3 - 0.5 * t2)
}

fn cubic_interp_2d(vals: &[[f32; 4]; 4], fx: f32, fy: f32) -> f32 {
    let mut row = [0.0_f32; 4];
    for y in 0..4 {
        row[y] = cubic_hermite(vals[0][y], vals[1][y], vals[2][y], vals[3][y], fx);
    }
    cubic_hermite(row[0], row[1], row[2], row[3], fy)
}

fn cubic_interp_3d(vals: &[[[f32; 4]; 4]; 4], fx: f32, fy: f32, fz: f32) -> f32 {
    let mut plane = [[0.0_f32; 4]; 4];
    for y in 0..4 {
        for z in 0..4 {
            plane[y][z] = cubic_hermite(
                vals[0][y][z],
                vals[1][y][z],
                vals[2][y][z],
                vals[3][y][z],
                fx,
            );
        }
    }
    let mut row = [0.0_f32; 4];
    for z in 0..4 {
        row[z] = cubic_hermite(plane[0][z], plane[1][z], plane[2][z], plane[3][z], fy);
    }
    cubic_hermite(row[0], row[1], row[2], row[3], fz)
}

// ============================================================================
// Gradient Direction Helpers
// ============================================================================

/// 3D gradient dot product: hash determines direction from 12-sided cube,
/// then dot with offset vector.
#[inline]
fn grad_coord_3d(seed: i32, x: i32, y: i32, z: i32, dx: f32, dy: f32, dz: f32) -> f32 {
    let h = (hash::hash_hb(seed, x, y, z) & 0xF) as usize;
    let u = if h < 8 { dx } else { dy };
    let v = if h < 4 {
        dy
    } else {
        if h == 12 || h == 14 {
            dx
        } else {
            dz
        }
    };
    let u = if h & 1 == 0 { u } else { -u };
    let v = if h & 2 == 0 { v } else { -v };
    u + v
}

/// 2D gradient dot product: 8-directional grad from hash.
#[inline]
fn grad_coord_2d(seed: i32, x: i32, y: i32, dx: f32, dy: f32) -> f32 {
    let h = (hash::hash_hb(seed, x, y, 0) & 7) as usize;
    let (u, v) = match h {
        0 => (dx, dy),
        1 => (-dx, dy),
        2 => (dx, -dy),
        3 => (-dx, -dy),
        4 => (dx, dx),
        5 => (-dx, dx),
        6 => (dx, -dx),
        _ => (-dx, -dx),
    };
    u + v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar::{ScalarFloat, ScalarInt};

    #[test]
    fn test_value_noise_3d_range() {
        let s = Settings::new(42);
        for i in 0..10 {
            for j in 0..10 {
                let v = single_value_3d::<ScalarFloat, ScalarInt>(
                    s.seed,
                    i as f32 * 0.1,
                    j as f32 * 0.1,
                    0.5,
                );
                assert!(v.is_finite(), "value should be finite: {v}");
            }
        }
    }

    #[test]
    fn test_simplex_2d_range() {
        let s = Settings::new(42);
        for i in 0..10 {
            for j in 0..10 {
                let v = single_simplex_2d::<ScalarFloat, ScalarInt>(
                    s.seed,
                    i as f32 * 0.1,
                    j as f32 * 0.1,
                );
                assert!(v.is_finite(), "simplex value should be finite: {v}");
            }
        }
    }

    #[test]
    fn test_cellular_3d() {
        let mut s = Settings::new(42);
        s.cellular_distance_function = CellularDistanceFunction::Euclidean;
        s.cellular_return_type = crate::settings::CellularReturnType::Distance;
        let v = single_cellular_3d::<ScalarFloat, ScalarInt>(&s, 0.5, 0.5, 0.5);
        assert!(v.is_finite(), "cellular value should be finite: {v}");
    }
}
