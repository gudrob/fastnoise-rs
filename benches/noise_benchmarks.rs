//! Benchmark suite for fast-noise-simd-rs.
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use fast_noise_simd_rs::{
    CellularDistanceFunction, CellularReturnType, FastNoise, NoiseType, PerturbType,
};

fn bench_value_3d(c: &mut Criterion) {
    c.bench_function("value_3d", |b| {
        b.iter_batched(
            || FastNoise::new(1337),
            |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_value_fractal_3d(c: &mut Criterion) {
    c.bench_function("value_fractal_3d", |b| {
        b.iter_batched(
            || {
                FastNoise::new(1337)
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

fn bench_perlin_3d(c: &mut Criterion) {
    c.bench_function("perlin_3d", |b| {
        b.iter_batched(
            || FastNoise::new(1337).with_noise_type(NoiseType::Perlin),
            |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_simplex_3d(c: &mut Criterion) {
    c.bench_function("simplex_3d", |b| {
        b.iter_batched(
            || FastNoise::new(1337).with_noise_type(NoiseType::Simplex),
            |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_cellular_3d(c: &mut Criterion) {
    c.bench_function("cellular_3d", |b| {
        b.iter_batched(
            || {
                FastNoise::new(1337)
                    .with_noise_type(NoiseType::Cellular)
                    .with_cellular_distance_function(CellularDistanceFunction::Euclidean)
                    .with_cellular_return_type(CellularReturnType::Distance2)
            },
            |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_cubic_3d(c: &mut Criterion) {
    c.bench_function("cubic_3d", |b| {
        b.iter_batched(
            || FastNoise::new(1337).with_noise_type(NoiseType::Cubic),
            |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_grid_256x256(c: &mut Criterion) {
    c.bench_function("grid_256x256", |b| {
        b.iter_batched(
            || {
                FastNoise::new(1337)
                    .with_frequency(0.01)
                    .with_noise_type(NoiseType::SimplexFractal)
                    .with_fractal_octaves(3)
            },
            |noise| {
                noise.generate_grid_2d(black_box(0), black_box(0), black_box(256), black_box(256));
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_perturb_3d(c: &mut Criterion) {
    c.bench_function("perturb_3d", |b| {
        b.iter_batched(
            || {
                FastNoise::new(1337)
                    .with_noise_type(NoiseType::Simplex)
                    .with_perturb_type(PerturbType::GradientFractal)
                    .with_perturb_amp(0.5)
            },
            |noise| noise.get_noise_3d(black_box(1.3), black_box(2.7), black_box(3.1)),
            BatchSize::SmallInput,
        );
    });
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
