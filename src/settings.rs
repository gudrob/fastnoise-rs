//! Configuration types for FastNoiseSIMD.
//!
//! Mirrors the settings from the C++ `FastNoiseSIMD` class.

/// The type of noise to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseType {
    Value,
    ValueFractal,
    Perlin,
    PerlinFractal,
    Simplex,
    SimplexFractal,
    Cellular,
    WhiteNoise,
    Cubic,
    CubicFractal,
}

/// Fractal type for fractal noise variants (FBM, Billow, RigidMulti).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractalType {
    FBM,
    Billow,
    RigidMulti,
}

/// Distance function for cellular/Voronoi noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellularDistanceFunction {
    Euclidean,
    Manhattan,
    Natural,
}

/// What the cellular noise function returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellularReturnType {
    CellValue,
    Distance,
    Distance2,
    Distance2Add,
    Distance2Sub,
    Distance2Mul,
    Distance2Div,
    NoiseLookup,
    Distance2Cave,
}

/// Type of domain warping (perturb) to apply before noise evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbType {
    None,
    Gradient,
    GradientFractal,
    Normalise,
    GradientNormalise,
    GradientFractalNormalise,
}

/// Builder-style configuration for FastNoise noise generation.
///
/// All C++ `FastNoiseSIMD` parameters are represented here.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    // Seed
    pub seed: i32,

    // General
    pub frequency: f32,
    pub noise_type: NoiseType,

    // Fractal
    pub fractal_type: FractalType,
    pub octaves: i32,
    pub lacunarity: f32,
    pub gain: f32,
    pub fractal_bounding: f32,

    // Cellular
    pub cellular_distance_function: CellularDistanceFunction,
    pub cellular_return_type: CellularReturnType,
    pub cellular_noise_lookup_type: NoiseType,
    pub cellular_noise_lookup_frequency: f32,
    pub cellular_jitter: f32,

    // Perturb
    pub perturb_type: PerturbType,
    pub perturb_frequency: f32,
    pub perturb_amplitude: f32,
    pub perturb_octaves: i32,
    pub perturb_lacunarity: f32,
    pub perturb_gain: f32,
    pub perturb_normalise_length: f32,

    // Axis scales (frequency multipliers per axis)
    pub x_scale: f32,
    pub y_scale: f32,
    pub z_scale: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            seed: 1337,
            frequency: 0.01,
            noise_type: NoiseType::Simplex,

            fractal_type: FractalType::FBM,
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
            fractal_bounding: 0.0, // calculated in init

            cellular_distance_function: CellularDistanceFunction::Euclidean,
            cellular_return_type: CellularReturnType::Distance,
            cellular_noise_lookup_type: NoiseType::Simplex,
            cellular_noise_lookup_frequency: 0.2,
            cellular_jitter: 0.45,

            perturb_type: PerturbType::None,
            perturb_frequency: 0.5,
            perturb_amplitude: 1.0,
            perturb_octaves: 3,
            perturb_lacunarity: 2.0,
            perturb_gain: 0.5,
            perturb_normalise_length: 1.0,

            x_scale: 1.0,
            y_scale: 1.0,
            z_scale: 1.0,
        }
    }
}

impl Settings {
    /// Create a new settings instance with the given seed.
    #[must_use]
    pub fn new(seed: i32) -> Self {
        Self {
            seed,
            ..Self::default()
        }
    }

    /// Calculate the fractal bounding value.
    ///
    /// `fractal_bounding = sum(gain^i for i in 0..octaves)⁻¹`
    /// This normalizes the fractal output to approximately [-1, 1].
    #[must_use]
    pub fn calculate_fractal_bounding(octaves: i32, gain: f32) -> f32 {
        let mut amp = 1.0_f32;
        let mut amp_fractal = 1.0_f32;
        for _ in 1..octaves {
            amp *= gain;
            amp_fractal += amp;
        }
        1.0 / amp_fractal
    }

    /// Initialize computed fields (call after changing octaves/gain).
    pub fn init(&mut self) {
        if self.fractal_bounding == 0.0 {
            self.fractal_bounding = Self::calculate_fractal_bounding(self.octaves, self.gain);
        }
    }
}
