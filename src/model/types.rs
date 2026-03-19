/// A three-component single-precision floating-point vector.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Returns the zero vector.
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Returns the vector with all components set to `v`.
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Returns the vector with components `(x, y, z)`.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns the dot product with `other`.
    #[inline(always)]
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns the cross product with `other`.
    #[inline(always)]
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns the squared length of the vector.
    #[inline(always)]
    pub fn len_sq(self) -> f32 {
        self.dot(self)
    }

    /// Returns the length of the vector.
    #[inline(always)]
    pub fn len(self) -> f32 {
        self.len_sq().sqrt()
    }

    /// Returns the unit vector in the same direction. Behaviour is unspecified for
    /// zero-length vectors.
    #[inline(always)]
    pub fn normalize(self) -> Vec3 {
        self * (1.0 / self.len())
    }

    /// Returns the squared Euclidean distance to `other`.
    #[inline(always)]
    pub fn dist_sq(self, other: Vec3) -> f32 {
        (self - other).len_sq()
    }

    /// Returns the Euclidean distance to `other`.
    #[inline(always)]
    pub fn dist(self, other: Vec3) -> f32 {
        self.dist_sq(other).sqrt()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, s: f32) -> Vec3 {
        Vec3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// DREIDING atom-type table index.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeIdx(pub u8);

impl From<TypeIdx> for usize {
    #[inline]
    fn from(t: TypeIdx) -> usize {
        t.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::Vec3;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn zero_is_all_zeros() {
        assert_eq!(Vec3::zero(), v(0.0, 0.0, 0.0));
    }

    #[test]
    fn splat_sets_all_components() {
        assert_eq!(Vec3::splat(3.0), v(3.0, 3.0, 3.0));
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = v(1.0, 2.0, 3.0);
        let b = v(4.0, -1.0, 0.5);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn neg_equals_mul_neg_one() {
        let a = v(1.0, -2.0, 3.0);
        assert_eq!(-a, a * -1.0);
    }

    #[test]
    fn scale_by_zero_gives_zero() {
        assert_eq!(v(5.0, -3.0, 1.0) * 0.0, Vec3::zero());
    }

    #[test]
    fn dot_orthogonal_axes_is_zero() {
        assert_abs_diff_eq!(v(1.0, 0.0, 0.0).dot(v(0.0, 1.0, 0.0)), 0.0);
        assert_abs_diff_eq!(v(0.0, 1.0, 0.0).dot(v(0.0, 0.0, 1.0)), 0.0);
    }

    #[test]
    fn dot_self_equals_len_sq() {
        let a = v(2.0, 3.0, 6.0);
        assert_abs_diff_eq!(a.dot(a), a.len_sq());
    }

    #[test]
    fn dot_is_commutative() {
        let a = v(1.0, 2.0, 3.0);
        let b = v(4.0, 5.0, 6.0);
        assert_abs_diff_eq!(a.dot(b), b.dot(a));
    }

    #[test]
    fn cross_right_hand_rule() {
        assert_eq!(v(1.0, 0.0, 0.0).cross(v(0.0, 1.0, 0.0)), v(0.0, 0.0, 1.0));
    }

    #[test]
    fn cross_anticommutative() {
        let a = v(1.0, 2.0, 3.0);
        let b = v(4.0, 5.0, 6.0);
        assert_eq!(a.cross(b), -(b.cross(a)));
    }

    #[test]
    fn cross_parallel_vectors_is_zero() {
        let a = v(1.0, 2.0, 3.0);
        assert_eq!(a.cross(a * 2.0), Vec3::zero());
    }

    #[test]
    fn cross_result_is_orthogonal_to_both_inputs() {
        let a = v(1.0, 2.0, 3.0);
        let b = v(4.0, 5.0, 6.0);
        let c = a.cross(b);
        assert_abs_diff_eq!(c.dot(a), 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(c.dot(b), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn pythagorean_triple_3_4_5() {
        assert_relative_eq!(v(3.0, 0.0, 4.0).len(), 5.0);
    }

    #[test]
    fn len_sq_consistent_with_len() {
        let a = v(1.0, 2.0, 3.0);
        assert_relative_eq!(a.len() * a.len(), a.len_sq());
    }

    #[test]
    fn normalize_produces_unit_vector() {
        assert_relative_eq!(v(3.0, 0.0, 4.0).normalize().len(), 1.0);
        assert_relative_eq!(v(1.0, 1.0, 1.0).normalize().len(), 1.0);
    }

    #[test]
    fn normalize_preserves_direction() {
        assert_eq!(v(0.0, 0.0, 5.0).normalize(), v(0.0, 0.0, 1.0));
    }

    #[test]
    fn dist_sq_consistent_with_sub_len_sq() {
        let a = v(1.0, 2.0, 3.0);
        let b = v(4.0, 6.0, 3.0);
        assert_abs_diff_eq!(a.dist_sq(b), (a - b).len_sq());
    }

    #[test]
    fn dist_pythagorean_triple() {
        assert_relative_eq!(v(0.0, 0.0, 0.0).dist(v(3.0, 0.0, 4.0)), 5.0);
    }

    #[test]
    fn dist_is_symmetric() {
        let a = v(1.0, 2.0, 3.0);
        let b = v(4.0, 5.0, 6.0);
        assert_abs_diff_eq!(a.dist_sq(b), b.dist_sq(a));
    }
}
