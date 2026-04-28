//! Benchmark suite for fast-noise-simd-rs.
//!
//! Run with: `cargo bench`
//!
//! Compares performance across SIMD backends:
//!   Scalar, SSE4.1, AVX2, AVX-512F
//!
//! To benchmark a specific backend only:
//!   cargo bench --features sse41
//!   cargo bench --features avx2
//!   cargo bench --features avx512

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use fast_noise_simd_rs::{
    CellularDistanceFunction, CellularReturnType, FastNoise, NoiseType, PerturbType, SimdLevel,
};

// ============================================================================
// Helper: Build a FastNoise with forced SIMD level
// ============================================================================

/// Available SIMD backends (scalar always available; others gated by cpu features).
fn simd_backends() -> Vec<(&'static str, SimdLevel)> {
    let mut backends = vec![("scalar", SimdLevel::Scalar)];

    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    backends.push(("sse41", SimdLevel::Sse41));

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    backends.push(("avx2", SimdLevel::Avx2));

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    backends.push(("avx512", SimdLevel::Avx512));

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    backends.push(("neon", SimdLevel::Neon));

    backends
}

// ============================================================================
// Single-point benchmarks (per-backend)
// ============================================================================

fn bench_value_3d(c: &mut Criterion) {
    for (name, level) in simd_backends() {
        let group_name = format!("value_3d/{}", name);
        c.bench_function(&group_name, |b| {
            b.iter_batched(
                || FastNoise::new(1337).with_simd_level(level),
                |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_value_fractal_3d(c: &mut Criterion) {
    for (name, level) in simd_backends() {
        let group_name = format!("value_fractal_3d/{}", name);
        c.bench_function(&group_name, |b| {
            b.iter_batched(
                || {
                    FastNoise::new(1337)
                        .with_simd_level(level)
                        .with_noise_type(NoiseType::ValueFractal)
                        .with_fractal_octaves(5)
                        .with_fractal_gain(0.5)
                        .with_fractal_lacunarity(2.0)
                },
                |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_perlin_3d(c: &mut Criterion) {
    for (name, level) in simd_backends() {
        let group_name = format!("perlin_3d/{}", name);
        c.bench_function(&group_name, |b| {
            b.iter_batched(
                || {
                    FastNoise::new(1337)
                        .with_simd_level(level)
                        .with_noise_type(NoiseType::Perlin)
                },
                |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_simplex_3d(c: &mut Criterion) {
    for (name, level) in simd_backends() {
        let group_name = format!("simplex_3d/{}", name);
        c.bench_function(&group_name, |b| {
            b.iter_batched(
                || {
                    FastNoise::new(1337)
                        .with_simd_level(level)
                        .with_noise_type(NoiseType::Simplex)
                },
                |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_cellular_3d(c: &mut Criterion) {
    for (name, level) in simd_backends() {
        let group_name = format!("cellular_3d/{}", name);
        c.bench_function(&group_name, |b| {
            b.iter_batched(
                || {
                    FastNoise::new(1337)
                        .with_simd_level(level)
                        .with_noise_type(NoiseType::Cellular)
                        .with_cellular_distance_function(CellularDistanceFunction::Euclidean)
                        .with_cellular_return_type(CellularReturnType::Distance2)
                },
                |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_cubic_3d(c: &mut Criterion) {
    for (name, level) in simd_backends() {
        let group_name = format!("cubic_3d/{}", name);
        c.bench_function(&group_name, |b| {
            b.iter_batched(
                || {
                    FastNoise::new(1337)
                        .with_simd_level(level)
                        .with_noise_type(NoiseType::Cubic)
                },
                |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_grid_256x256(c: &mut Criterion) {
    for (name, level) in simd_backends() {
        let group_name = format!("grid_256x256/{}", name);
        c.bench_function(&group_name, |b| {
            b.iter_batched(
                || {
                    FastNoise::new(1337)
                        .with_simd_level(level)
                        .with_frequency(0.01)
                        .with_noise_type(NoiseType::SimplexFractal)
                        .with_fractal_octaves(3)
                },
                |noise| {
                    noise.generate_grid_2d(
                        black_box(0),
                        black_box(0),
                        black_box(256),
                        black_box(256),
                    );
                },
                BatchSize::LargeInput,
            );
        });
    }
}

fn bench_perturb_3d(c: &mut Criterion) {
    for (name, level) in simd_backends() {
        let group_name = format!("perturb_3d/{}", name);
        c.bench_function(&group_name, |b| {
            b.iter_batched(
                || {
                    FastNoise::new(1337)
                        .with_simd_level(level)
                        .with_noise_type(NoiseType::Simplex)
                        .with_perturb_type(PerturbType::GradientFractal)
                        .with_perturb_amp(0.5)
                },
                |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
                BatchSize::SmallInput,
            );
        });
    }
}

criterion_group!(
    benches,
    bench_value_3d,
    bench_value_fractal_3d,
    bench_perlin_3d,
    bench_simplex_3d,
    bench_cellular_3d,
    bench_cubic_3d,
    bench_grid_256x256,
    bench_perturb_3d,
);
criterion_main!(benches);
