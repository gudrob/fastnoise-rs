//! FastNoiseVectorSet equivalent – a collection of 3D position vectors
//! for generating noise at arbitrary sample points.

/// A set of 3D positions at which to sample noise.
///
/// Equivalent to C++ `FastNoiseVectorSet`.
/// The `size` is the number of SIMD-aligned float entries in each array,
/// i.e. the number of samples.
#[derive(Debug, Clone)]
pub struct NoiseVectorSet {
    /// X coordinates of sample points (size elements).
    pub x_set: Vec<f32>,
    /// Y coordinates of sample points (size elements).
    pub y_set: Vec<f32>,
    /// Z coordinates of sample points (size elements).
    pub z_set: Vec<f32>,
    /// Number of sample points (= length of each coordinate array).
    pub size: usize,
}

impl NoiseVectorSet {
    /// Create a new vector set with the given positions.
    ///
    /// All three coordinate arrays must have the same length.
    #[must_use]
    pub fn new(x_set: Vec<f32>, y_set: Vec<f32>, z_set: Vec<f32>) -> Self {
        let size = x_set.len();
        assert_eq!(y_set.len(), size, "y_set length must match x_set");
        assert_eq!(z_set.len(), size, "z_set length must match x_set");

        Self {
            x_set,
            y_set,
            z_set,
            size,
        }
    }

    /// Create an empty vector set.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            x_set: Vec::new(),
            y_set: Vec::new(),
            z_set: Vec::new(),
            size: 0,
        }
    }

    /// Check if the vector set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Default for NoiseVectorSet {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_matching_lengths() {
        let vs = NoiseVectorSet::new(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        assert_eq!(vs.size, 2);
        assert_eq!(vs.x_set[0], 1.0);
        assert_eq!(vs.y_set[1], 4.0);
    }

    #[test]
    #[should_panic]
    fn test_new_with_mismatched_lengths() {
        let _ = NoiseVectorSet::new(vec![1.0, 2.0], vec![3.0], vec![5.0, 6.0]);
    }

    #[test]
    fn test_empty() {
        let vs = NoiseVectorSet::empty();
        assert!(vs.is_empty());
        assert_eq!(vs.size, 0);
    }
}
