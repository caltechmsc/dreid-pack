/// A three-component single-precision floating-point vector.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
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
