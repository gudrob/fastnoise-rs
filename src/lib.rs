//! # fast-noise-simd-rs
//!
//! A zero-dependency Rust port of [FastNoiseSIMD](https://github.com/Auburn/FastNoiseSIMD),
//! a fast SIMD-accelerated coherent noise generation library.
//!
//! ## Features
//! - **Noise Types**: Value, ValueFractal, Perlin, PerlinFractal, Simplex, SimplexFractal,
//!   WhiteNoise, Cellular, Cubic, CubicFractal
//! - **Fractal Types**: FBM, Billow, RigidMulti
//! - **Cellular Distance Functions**: Euclidean, Manhattan, Natural
//! - **Cellular Return Types**: CellValue, Distance, Distance2, Distance2Add,
//!   Distance2Sub, Distance2Mul, Distance2Div, NoiseLookup, Distance2Cave
//! - **Domain Warping**: Gradient, GradientFractal, Normalise, combined variants
//! - **SIMD Acceleration**: SSE2, SSE4.1, AVX2+FMA, AVX-512F, ARM NEON
//!   (all via `std::arch`, no external dependencies)
//!
//! ## Usage
//! ```ignore
//! use fast_noise_simd_rs::FastNoise;
//!
//! let noise = FastNoise::new(1337)
//!     .with_frequency(0.01)
//!     .with_noise_type(NoiseType::SimplexFractal)
//!     .with_fractal_octaves(5)
//!     .with_fractal_gain(0.5)
//!     .with_fractal_lacunarity(2.0);
//!
//! let grid = noise.generate_grid(0, 0, 0, 256, 256, 1);
//! ```

// Public modules
pub mod hash;
pub mod settings;
pub mod vectorset;

// Internal modules
pub(crate) mod fractal;
pub(crate) mod kernel;
pub(crate) mod noise;
pub(crate) mod perturb;
pub(crate) mod simd;

// Re-exports for convenience
pub use settings::{
    CellularDistanceFunction, CellularReturnType, FractalType, NoiseType, PerturbType,
};
pub use vectorset::NoiseVectorSet;

use simd::scalar::{ScalarFloat, ScalarInt};

// ============================================================================
// SIMD Dispatch Helpers
// ============================================================================

#[allow(unused_macros)]
macro_rules! with_simd {
    ($op:ident, $F:ty, $I:ty, $($args:expr),* $(,)?) => {{
        // Helper that resolves to the best available SIMD path at runtime.
        // For now, always uses scalar; feature-gated modules can override.
        $op::<$F, $I>($($args),*)
    }};
}

// ============================================================================
// FastNoise – Main entry point
// ============================================================================

/// Main noise generator struct.
///
/// Holds all settings and provides single-sample and grid generation methods.
#[derive(Debug, Clone)]
pub struct FastNoise {
    settings: settings::Settings,
}

impl FastNoise {
    /// Create a new `FastNoise` with the given seed.
    pub fn new(seed: i32) -> Self {
        Self {
            settings: settings::Settings::new(seed),
        }
    }

    // ------------------------------------------------------------------
    // Builder methods
    // ------------------------------------------------------------------

    pub fn with_noise_type(mut self, noise_type: NoiseType) -> Self {
        self.settings.noise_type = noise_type;
        self
    }

    pub fn with_frequency(mut self, frequency: f32) -> Self {
        self.settings.frequency = frequency;
        self
    }

    pub fn with_fractal_type(mut self, fractal_type: FractalType) -> Self {
        self.settings.fractal_type = fractal_type;
        self
    }

    pub fn with_fractal_octaves(mut self, octaves: i32) -> Self {
        self.settings.octaves = octaves;
        self
    }

    pub fn with_fractal_lacunarity(mut self, lacunarity: f32) -> Self {
        self.settings.lacunarity = lacunarity;
        self
    }

    pub fn with_fractal_gain(mut self, gain: f32) -> Self {
        self.settings.gain = gain;
        self
    }

    pub fn with_cellular_distance_function(mut self, df: CellularDistanceFunction) -> Self {
        self.settings.cellular_distance_function = df;
        self
    }

    pub fn with_cellular_return_type(mut self, rt: CellularReturnType) -> Self {
        self.settings.cellular_return_type = rt;
        self
    }

    pub fn with_cellular_jitter(mut self, jitter: f32) -> Self {
        self.settings.cellular_jitter = jitter;
        self
    }

    pub fn with_cellular_noise_lookup_frequency(mut self, freq: f32) -> Self {
        self.settings.cellular_noise_lookup_frequency = freq;
        self
    }

    pub fn with_perturb_type(mut self, perturb_type: PerturbType) -> Self {
        self.settings.perturb_type = perturb_type;
        self
    }

    pub fn with_perturb_amp(mut self, amp: f32) -> Self {
        self.settings.perturb_amplitude = amp;
        self
    }

    pub fn with_perturb_frequency(mut self, freq: f32) -> Self {
        self.settings.perturb_frequency = freq;
        self
    }

    pub fn with_x_scale(mut self, scale: f32) -> Self {
        self.settings.x_scale = scale;
        self
    }

    pub fn with_y_scale(mut self, scale: f32) -> Self {
        self.settings.y_scale = scale;
        self
    }

    pub fn with_z_scale(mut self, scale: f32) -> Self {
        self.settings.z_scale = scale;
        self
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Get a reference to the current settings.
    pub fn settings(&self) -> &settings::Settings {
        &self.settings
    }

    /// Set all settings at once.
    pub fn set_settings(&mut self, settings: settings::Settings) {
        self.settings = settings;
    }

    // ------------------------------------------------------------------
    // Single-sample noise
    // ------------------------------------------------------------------

    /// Get noise value at (x, y, z).
    pub fn get_noise_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        noise::generate_3d::<ScalarFloat, ScalarInt>(&self.settings, x, y, z)
    }

    /// Get noise value at (x, y).
    pub fn get_noise_2d(&self, x: f32, y: f32) -> f32 {
        noise::generate_2d::<ScalarFloat, ScalarInt>(&self.settings, x, y)
    }

    // ------------------------------------------------------------------
    // Grid generation
    // ------------------------------------------------------------------

    /// Generate a 3D noise grid.
    ///
    /// Returns a flat `Vec<f32>` ordered x => y => z.
    ///
    /// Currently uses scalar per-pixel evaluation (SIMD kernel integration
    /// is incomplete — see README TODOs).
    pub fn generate_grid(
        &self,
        start_x: i32,
        start_y: i32,
        start_z: i32,
        width: i32,
        height: i32,
        depth: i32,
    ) -> Vec<f32> {
        let count = (width * height * depth) as usize;
        let mut out = Vec::with_capacity(count);

        for z in start_z..start_z + depth {
            for y in start_y..start_y + height {
                for x in start_x..start_x + width {
                    out.push(self.get_noise_3d(x as f32, y as f32, z as f32));
                }
            }
        }

        out
    }

    /// Generate a 2D noise grid.
    ///
    /// Returns a flat `Vec<f32>` ordered x => y.
    pub fn generate_grid_2d(
        &self,
        start_x: i32,
        start_y: i32,
        width: i32,
        height: i32,
    ) -> Vec<f32> {
        let count = (width * height) as usize;
        let mut out = Vec::with_capacity(count);

        for y in start_y..start_y + height {
            for x in start_x..start_x + width {
                out.push(self.get_noise_2d(x as f32, y as f32));
            }
        }

        out
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_value_noise() {
        let noise = FastNoise::new(42)
            .with_frequency(1.0)
            .with_noise_type(NoiseType::Value);
        let v = noise.get_noise_3d(0.5, 0.5, 0.5);
        assert!(v.is_finite(), "value: {v}");
    }

    #[test]
    fn test_simplex_fractal() {
        let noise = FastNoise::new(42)
            .with_frequency(1.0)
            .with_noise_type(NoiseType::SimplexFractal)
            .with_fractal_octaves(3)
            .with_fractal_gain(0.5)
            .with_fractal_lacunarity(2.0);
        let v = noise.get_noise_3d(1.0, 2.0, 3.0);
        assert!(v.is_finite(), "value: {v}");
    }

    #[test]
    fn test_cellular_noise() {
        let noise = FastNoise::new(42)
            .with_noise_type(NoiseType::Cellular)
            .with_cellular_distance_function(CellularDistanceFunction::Euclidean)
            .with_cellular_return_type(CellularReturnType::Distance2);
        let v = noise.get_noise_2d(0.5, 0.5);
        assert!(v.is_finite(), "value: {v}");
    }

    #[test]
    fn test_cubic_noise() {
        let noise = FastNoise::new(42)
            .with_frequency(1.0)
            .with_noise_type(NoiseType::CubicFractal)
            .with_fractal_octaves(2);
        let v = noise.get_noise_3d(1.0, 1.0, 1.0);
        assert!(v.is_finite(), "value: {v}");
    }

    #[test]
    fn test_grid_generation_3d() {
        let noise = FastNoise::new(42).with_frequency(0.1);
        let grid = noise.generate_grid(0, 0, 0, 8, 8, 2);
        assert_eq!(grid.len(), 128);
        for v in &grid {
            assert!(v.is_finite(), "grid value not finite: {v}");
        }
    }

    #[test]
    fn test_grid_generation_2d() {
        let noise = FastNoise::new(42).with_frequency(0.1);
        let grid = noise.generate_grid_2d(0, 0, 16, 16);
        assert_eq!(grid.len(), 256);
        for v in &grid {
            assert!(v.is_finite(), "grid value not finite: {v}");
        }
    }

    #[test]
    fn test_all_noise_types_finite() {
        let types = [
            NoiseType::Value,
            NoiseType::ValueFractal,
            NoiseType::Perlin,
            NoiseType::PerlinFractal,
            NoiseType::Simplex,
            NoiseType::SimplexFractal,
            NoiseType::Cellular,
            NoiseType::WhiteNoise,
            NoiseType::Cubic,
            NoiseType::CubicFractal,
        ];

        for t in types.iter() {
            let noise = FastNoise::new(42).with_noise_type(*t);
            let v = noise.get_noise_3d(1.3, 2.7, 3.1);
            assert!(
                v.is_finite(),
                "noise type {:?} produced non-finite value: {v}",
                t
            );
        }
    }

    #[test]
    fn test_perturb_noise() {
        let noise = FastNoise::new(42)
            .with_noise_type(NoiseType::Simplex)
            .with_perturb_type(PerturbType::Gradient)
            .with_perturb_amp(0.1);
        let v = noise.get_noise_3d(1.0, 2.0, 3.0);
        assert!(v.is_finite(), "perturb value: {v}");
    }
}
