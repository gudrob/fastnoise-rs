//! Hash functions ported from FastNoiseSIMD C++.
//!
//! Uses the same prime multipliers and bit operations as the original
//! to ensure identical output for the same seed/coordinates.

/// Prime used for x-coordinate hashing (501125321).
pub const X_PRIME: i32 = 501125321;
/// Prime used for y-coordinate hashing (1136930381).
pub const Y_PRIME: i32 = 1136930381;
/// Prime used for z-coordinate hashing (1720413743).
pub const Z_PRIME: i32 = 1720413743;

/// 10-bit mask (1023) used to extract direction components from the hash.
pub const BIT_10_MASK: i32 = 0x3FF;

/// Conversion from hash integer to float: multiply by `2^-31`.
/// `1.0 / 2_147_483_648.0`
pub const HASH_TO_FLOAT: f32 = 4.656_613e-10;

/// Hash-based "hash bits" function.
///
/// This is the core hash used by FastNoiseSIMD.
/// Ported from `FN_DECL_CONSTEXPR int HashHB(int seed, int x, int y, int z)`.
///
/// Returns a 32-bit signed integer hash combining all inputs.
#[inline]
#[must_use]
pub fn hash_hb(seed: i32, x: i32, y: i32, z: i32) -> i32 {
    let mut hash = seed;
    hash ^= x.wrapping_mul(X_PRIME);
    hash ^= y.wrapping_mul(Y_PRIME);
    hash ^= z.wrapping_mul(Z_PRIME);

    // Avalanche mixing – same bit operations as C++ HashHB
    hash = hash.wrapping_mul(0x27d4_eb2d);

    let hash2 = hash.wrapping_mul(hash); // hash^2
    hash = hash2
        .wrapping_mul(60493)
        .wrapping_add(hash.wrapping_mul(61379))
        .wrapping_add(hash);

    hash
}

/// Value coordinate hash: returns a float in [-1, 1) range.
///
/// Hash-based random value for a given coordinate.
/// This is `ValCoord` from the C++ source.
#[inline]
#[must_use]
pub fn val_coord_f32(seed: i32, x: i32, y: i32, z: i32) -> f32 {
    let h = val_coord_i32(seed, x, y, z);
    h as f32 * HASH_TO_FLOAT
}

/// Value coordinate hash: returns the raw integer hash.
#[inline]
#[must_use]
pub fn val_coord_i32(seed: i32, x: i32, y: i32, z: i32) -> i32 {
    hash_hb(seed, x, y, z)
}

/// 2D variant: Value coordinate hash (z=0).
#[inline]
#[must_use]
pub fn val_coord_2d_f32(seed: i32, x: i32, y: i32) -> f32 {
    val_coord_f32(seed, x, y, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_determinism() {
        // Same inputs must produce same outputs
        let a = hash_hb(1337, 42, 17, 99);
        let b = hash_hb(1337, 42, 17, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash_different_seeds() {
        // Different seeds should produce different values (with high probability)
        let a = hash_hb(1337, 42, 17, 99);
        let b = hash_hb(1338, 42, 17, 99);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_different_coords() {
        let a = hash_hb(1337, 42, 17, 99);
        let b = hash_hb(1337, 43, 17, 99);
        assert_ne!(a, b);
    }

    #[test]
    fn test_val_coord_range() {
        // val_coord should return values in [-1, 1)
        for x in 0..100 {
            for y in 0..10 {
                let v = val_coord_2d_f32(42, x, y);
                assert!(v >= -1.0, "value {v} < -1.0");
                assert!(v < 1.0, "value {v} >= 1.0");
            }
        }
    }

    #[test]
    fn test_hash_to_float_constant() {
        // 1.0 / 2147483648.0
        let expected = 1.0_f32 / 2_147_483_648.0_f32;
        assert!((HASH_TO_FLOAT - expected).abs() < 1e-10);
    }
}
