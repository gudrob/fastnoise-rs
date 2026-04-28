//! Batched SIMD noise kernels.
//!
//! These functions process `VECTOR_SIZE` samples in parallel,
//! where coordinates are loaded as SIMD vectors and all arithmetic
//! is done with SIMD trait operations.
//!
//! This is the key to matching FastNoiseSIMD's performance.

use crate::hash;
use crate::settings::{NoiseType, Settings};
use crate::simd::{SimdFloat, SimdInt};

// ============================================================================
// SIMD Coordinate Helpers
// ============================================================================

/// Build `VECTOR_SIZE` x-coordinates spaced by `stride` from `base_x`.
#[inline]
fn coord_x<F: SimdFloat>(base_x: f32, stride: f32) -> F {
    unsafe {
        let mut arr = [0.0_f32; 16];
        for (i, elem) in arr.iter_mut().enumerate().take(F::VECTOR_SIZE) {
            *elem = base_x + stride * i as f32;
        }
        F::load(arr.as_ptr())
    }
}

/// SIMD smoothstep (Hermite): 3t² - 2t³
#[inline]
fn smoothstep_simd<F: SimdFloat>(t: F) -> F {
    // t * t * (3 - 2*t) = 3t² - 2t³
    let two = F::set(2.0);
    let three = F::set(3.0);
    // 3*t*t - 2*t*t*t
    let t2 = t.mul(t);
    let t3 = t2.mul(t);
    let part1 = t2.mul(three);
    let part2 = t3.mul(two);
    part1.sub(part2)
}

/// SIMD linear interpolation
#[inline]
fn lerp_simd<F: SimdFloat>(a: F, b: F, t: F) -> F {
    // a + (b - a) * t
    t.mul_add(b.sub(a), a)
}

// ============================================================================
// Batched Value Noise 3D
// ============================================================================

/// Compute value noise for VECTOR_SIZE x-coordinates in parallel.
///
/// All lanes share the same y, z, seed. x varies per lane.
#[inline]
fn value_noise_3d_batch<F: SimdFloat, I: SimdInt>(
    seed: i32,
    x: F, // VECTOR_SIZE x coords
    y: f32,
    z: f32,
) -> F {
    // Extract x scalars, compute per-lane, load into SIMD.
    // Trilinear interpolation of hash values is done in SIMD.
    unsafe {
        let mut x_scalars = [0.0_f32; 16];
        F::store(x_scalars.as_mut_ptr(), x);

        let iy = y.floor() as i32;
        let iz = z.floor() as i32;
        let fy = y - iy as f32;
        let fz = z - iz as f32;

        let mut v000 = [0.0_f32; 16];
        let mut v100 = [0.0_f32; 16];
        let mut v010 = [0.0_f32; 16];
        let mut v110 = [0.0_f32; 16];
        let mut v001 = [0.0_f32; 16];
        let mut v101 = [0.0_f32; 16];
        let mut v011 = [0.0_f32; 16];
        let mut v111 = [0.0_f32; 16];
        let mut fx_arr = [0.0_f32; 16];

        for i in 0..F::VECTOR_SIZE {
            let xi = x_scalars[i];
            let ix = xi.floor() as i32;
            fx_arr[i] = xi - ix as f32;

            v000[i] = hash::val_coord_f32(seed, ix, iy, iz);
            v100[i] = hash::val_coord_f32(seed, ix + 1, iy, iz);
            v010[i] = hash::val_coord_f32(seed, ix, iy + 1, iz);
            v110[i] = hash::val_coord_f32(seed, ix + 1, iy + 1, iz);
            v001[i] = hash::val_coord_f32(seed, ix, iy, iz + 1);
            v101[i] = hash::val_coord_f32(seed, ix + 1, iy, iz + 1);
            v011[i] = hash::val_coord_f32(seed, ix, iy + 1, iz + 1);
            v111[i] = hash::val_coord_f32(seed, ix + 1, iy + 1, iz + 1);
        }

        let v000_s = F::load(v000.as_ptr());
        let v100_s = F::load(v100.as_ptr());
        let v010_s = F::load(v010.as_ptr());
        let v110_s = F::load(v110.as_ptr());
        let v001_s = F::load(v001.as_ptr());
        let v101_s = F::load(v101.as_ptr());
        let v011_s = F::load(v011.as_ptr());
        let v111_s = F::load(v111.as_ptr());

        let fx_simd = F::load(fx_arr.as_ptr());
        let sx = smoothstep_simd(fx_simd);
        let sy = F::set(smoothstep(fy));
        let sz = F::set(smoothstep(fz));

        // Trilinear interpolation
        let i0 = lerp_simd(v000_s, v100_s, sx);
        let i1 = lerp_simd(v010_s, v110_s, sx);
        let i2 = lerp_simd(v001_s, v101_s, sx);
        let i3 = lerp_simd(v011_s, v111_s, sx);

        let j0 = lerp_simd(i0, i1, sy);
        let j1 = lerp_simd(i2, i3, sy);

        lerp_simd(j0, j1, sz)
    }
}

// ============================================================================
// Efficient SIMD Grid Filling
// ============================================================================

/// Fill a 3D noise grid using batched SIMD along the x-axis.
///
/// Processes `VECTOR_SIZE` consecutive x values at once per y,z iteration.
/// Then falls back to scalar for any remainder.
#[allow(clippy::too_many_arguments)]
pub fn fill_noise_set_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    start_x: i32,
    start_y: i32,
    start_z: i32,
    width: i32,
    height: i32,
    depth: i32,
    noise_set_out: &mut [f32],
) {
    let vs = F::VECTOR_SIZE as i32;
    let mut idx = 0;

    for z in start_z..start_z + depth {
        for y in start_y..start_y + height {
            let mut x = start_x;
            // Process full SIMD chunks
            while x + vs <= start_x + width {
                let batch = noise_batch_3d::<F, I>(
                    settings, x as f32, y as f32, z as f32,
                    1.0, // unit stride for consecutive x
                );

                unsafe {
                    // Store results
                    let mut out_slice = [0.0_f32; 16];
                    F::store(out_slice.as_mut_ptr(), batch);
                    for &val in out_slice.iter().take(F::VECTOR_SIZE) {
                        noise_set_out[idx] = val;
                        idx += 1;
                    }
                }
                x += vs;
            }
            // Scalar remainder
            while x < start_x + width {
                noise_set_out[idx] =
                    noise_generate_sample_3d::<F, I>(settings, x as f32, y as f32, z as f32);
                idx += 1;
                x += 1;
            }
        }
    }
}

/// Compute a single sample by delegating to `noise::generate_3d`.
#[inline]
fn noise_generate_sample_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    super::noise::generate_3d::<F, I>(settings, x, y, z)
}

/// Compute a single sample by delegating to `noise::generate_2d`.
#[inline]
fn noise_generate_sample_2d<F: SimdFloat, I: SimdInt>(settings: &Settings, x: f32, y: f32) -> f32 {
    super::noise::generate_2d::<F, I>(settings, x, y)
}

/// Fill a 2D noise grid using batched SIMD along the x-axis.
///
/// Processes `VECTOR_SIZE` consecutive x values at once per y iteration.
/// Then falls back to scalar for any remainder.
pub fn fill_noise_set_2d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    start_x: i32,
    start_y: i32,
    width: i32,
    height: i32,
    noise_set_out: &mut [f32],
) {
    let vs = F::VECTOR_SIZE as i32;
    let mut idx = 0;

    for y in start_y..start_y + height {
        let mut x = start_x;
        // Process full SIMD chunks
        while x + vs <= start_x + width {
            let batch = noise_batch_2d::<F, I>(
                settings, x as f32, y as f32, 1.0, // unit stride for consecutive x
            );

            unsafe {
                let mut out_slice = [0.0_f32; 16];
                F::store(out_slice.as_mut_ptr(), batch);
                for &val in out_slice.iter().take(F::VECTOR_SIZE) {
                    noise_set_out[idx] = val;
                    idx += 1;
                }
            }
            x += vs;
        }
        // Scalar remainder
        while x < start_x + width {
            noise_set_out[idx] = noise_generate_sample_2d::<F, I>(settings, x as f32, y as f32);
            idx += 1;
            x += 1;
        }
    }
}

/// Compute a batch of VECTOR_SIZE noise samples at (x + stride*i, y).
///
/// Uses per-lane scalar evaluation for now (2D batch kernels TBD).
fn noise_batch_2d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_x: f32,
    y: f32,
    stride: f32,
) -> F {
    unsafe {
        let mut values = [0.0_f32; 16];
        for (i, elem) in values.iter_mut().enumerate().take(F::VECTOR_SIZE) {
            let x = base_x + stride * i as f32;
            *elem = noise_generate_sample_2d::<F, I>(settings, x, y);
        }
        F::load(values.as_ptr())
    }
}

/// Compute a batch of VECTOR_SIZE noise samples at (x + stride*i, y, z).
///
/// This is the core SIMD kernel. It parallelizes the innermost dimension.
fn noise_batch_3d<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_x: f32,
    y: f32,
    z: f32,
    stride: f32,
) -> F {
    let x = coord_x::<F>(base_x, stride);
    // Scale coordinates
    let x = x.mul(F::set(settings.x_scale * settings.frequency));
    let y_scaled = y * settings.y_scale * settings.frequency;
    let z_scaled = z * settings.z_scale * settings.frequency;

    let _y_simd = F::set(y_scaled);
    let _z_simd = F::set(z_scaled);

    let value = match settings.noise_type {
        NoiseType::Value => value_noise_3d_batch::<F, I>(settings.seed, x, y_scaled, z_scaled),
        NoiseType::Perlin => perlin_noise_3d_batch::<F, I>(settings.seed, x, y_scaled, z_scaled),
        NoiseType::Simplex => simplex_noise_3d_batch::<F, I>(settings.seed, x, y_scaled, z_scaled),
        _ => {
            // Non-batched noise types fall back to scalar per-lane
            // For fractal/cellular/cubic/white noise, we build per-lane
            build_batch_scalar::<F, I>(settings, base_x, y, z, stride)
        }
    };

    // Apply perturb (scalar per-lane for now)
    if settings.perturb_type != crate::settings::PerturbType::None {
        // Perturb is complex; fall back to per-lane
        build_batch_perturbed::<F, I>(settings, base_x, y, z, stride)
    } else {
        value
    }
}

/// Build a batch via per-lane scalar evaluation (fallback for non-batched types).
fn build_batch_scalar<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_x: f32,
    y: f32,
    z: f32,
    stride: f32,
) -> F {
    unsafe {
        let mut values = [0.0_f32; 16];
        for (i, elem) in values.iter_mut().enumerate().take(F::VECTOR_SIZE) {
            let x = base_x + stride * i as f32;
            *elem = noise_generate_sample_3d::<F, I>(settings, x, y, z);
        }
        F::load(values.as_ptr())
    }
}

/// Build a batch with perturb applied per-lane.
///
/// Scales coordinates before calling `perturb::perturb_3d`,
/// matching the coordinate scaling that `noise::generate_3d` performs
/// before its perturb dispatch.
fn build_batch_perturbed<F: SimdFloat, I: SimdInt>(
    settings: &Settings,
    base_x: f32,
    y: f32,
    z: f32,
    stride: f32,
) -> F {
    unsafe {
        let mut values = [0.0_f32; 16];
        for (i, elem) in values.iter_mut().enumerate().take(F::VECTOR_SIZE) {
            let sx = (base_x + stride * i as f32) * settings.x_scale * settings.frequency;
            let sy = y * settings.y_scale * settings.frequency;
            let sz = z * settings.z_scale * settings.frequency;
            *elem = super::perturb::perturb_3d::<F, I>(settings, sx, sy, sz);
        }
        F::load(values.as_ptr())
    }
}

// ============================================================================
// Batched Perlin Noise 3D
// ============================================================================

fn perlin_noise_3d_batch<F: SimdFloat, I: SimdInt>(seed: i32, x: F, y: f32, z: f32) -> F {
    unsafe {
        let mut g000 = [0.0_f32; 16];
        let mut g100 = [0.0_f32; 16];
        let mut g010 = [0.0_f32; 16];
        let mut g110 = [0.0_f32; 16];
        let mut g001 = [0.0_f32; 16];
        let mut g101 = [0.0_f32; 16];
        let mut g011 = [0.0_f32; 16];
        let mut g111 = [0.0_f32; 16];

        let mut x_scalars = [0.0_f32; 16];
        F::store(x_scalars.as_mut_ptr(), x);

        for i in 0..F::VECTOR_SIZE {
            let xi = x_scalars[i];
            let ix = xi.floor() as i32;
            let fx = xi - ix as f32;
            let fy = y - y.floor();
            let fz = z - z.floor();
            let iy = y.floor() as i32;
            let iz = z.floor() as i32;

            g000[i] = grad_coord_batch(seed, ix, iy, iz, fx, fy, fz);
            g100[i] = grad_coord_batch(seed, ix + 1, iy, iz, fx - 1.0, fy, fz);
            g010[i] = grad_coord_batch(seed, ix, iy + 1, iz, fx, fy - 1.0, fz);
            g110[i] = grad_coord_batch(seed, ix + 1, iy + 1, iz, fx - 1.0, fy - 1.0, fz);
            g001[i] = grad_coord_batch(seed, ix, iy, iz + 1, fx, fy, fz - 1.0);
            g101[i] = grad_coord_batch(seed, ix + 1, iy, iz + 1, fx - 1.0, fy, fz - 1.0);
            g011[i] = grad_coord_batch(seed, ix, iy + 1, iz + 1, fx, fy - 1.0, fz - 1.0);
            g111[i] = grad_coord_batch(seed, ix + 1, iy + 1, iz + 1, fx - 1.0, fy - 1.0, fz - 1.0);
        }

        let g000_s = F::load(g000.as_ptr());
        let g100_s = F::load(g100.as_ptr());
        let g010_s = F::load(g010.as_ptr());
        let g110_s = F::load(g110.as_ptr());
        let g001_s = F::load(g001.as_ptr());
        let g101_s = F::load(g101.as_ptr());
        let g011_s = F::load(g011.as_ptr());
        let g111_s = F::load(g111.as_ptr());

        let mut fx_arr = [0.0_f32; 16];
        let mut fy_arr = [0.0_f32; 16];
        let mut fz_arr = [0.0_f32; 16];
        for i in 0..F::VECTOR_SIZE {
            fx_arr[i] = x_scalars[i] - x_scalars[i].floor();
            fy_arr[i] = y - y.floor();
            fz_arr[i] = z - z.floor();
        }
        let fx_simd = F::load(fx_arr.as_ptr());
        let fy_simd = F::set(fy_arr[0]);
        let fz_simd = F::set(fz_arr[0]);

        let sx = smoothstep_simd(fx_simd);
        let sy = smoothstep_simd(fy_simd);
        let sz = smoothstep_simd(fz_simd);

        let i0 = lerp_simd(g000_s, g100_s, sx);
        let i1 = lerp_simd(g010_s, g110_s, sx);
        let i2 = lerp_simd(g001_s, g101_s, sx);
        let i3 = lerp_simd(g011_s, g111_s, sx);

        let j0 = lerp_simd(i0, i1, sy);
        let j1 = lerp_simd(i2, i3, sy);

        lerp_simd(j0, j1, sz)
    }
}

#[inline]
fn grad_coord_batch(seed: i32, x: i32, y: i32, z: i32, dx: f32, dy: f32, dz: f32) -> f32 {
    let h = (hash::hash_hb(seed, x, y, z) & 0xF) as usize;
    let u = if h < 8 { dx } else { dy };
    let v = if h < 4 {
        dy
    } else if h == 12 || h == 14 {
        dx
    } else {
        dz
    };
    let u = if h & 1 == 0 { u } else { -u };
    let v = if h & 2 == 0 { v } else { -v };
    u + v
}

// ============================================================================
// Batched Simplex Noise 3D
// ============================================================================

fn simplex_noise_3d_batch<F: SimdFloat, I: SimdInt>(seed: i32, x: F, y: f32, z: f32) -> F {
    // Simplex 3D is inherently scalar due to branching on coordinate ordering.
    // Each lane has a different (x0,y0,z0) skew → different simplex case → different code path.
    // Fall back to per-lane computation, then load results into SIMD register.
    unsafe {
        let mut x_scalars = [0.0_f32; 16];
        F::store(x_scalars.as_mut_ptr(), x);

        let mut values = [0.0_f32; 16];
        for i in 0..F::VECTOR_SIZE {
            values[i] = super::noise::single_simplex_3d::<F, I>(seed, x_scalars[i], y, z);
        }
        F::load(values.as_ptr())
    }
}

/// Scalar smoothstep used in batch helpers.
#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Settings;
    use crate::simd::scalar::{ScalarFloat, ScalarInt};

    #[test]
    fn test_fill_noise_3d_scalar() {
        let settings = Settings::new(42);
        let width = 128;
        let height = 16;
        let depth = 1;
        let mut out = vec![0.0_f32; (width * height * depth) as usize];
        fill_noise_set_3d::<ScalarFloat, ScalarInt>(
            &settings, 0, 0, 0, width, height, depth, &mut out,
        );
        for v in &out {
            assert!(v.is_finite(), "value not finite: {v}");
        }
    }
}