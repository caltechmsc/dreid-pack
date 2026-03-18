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
