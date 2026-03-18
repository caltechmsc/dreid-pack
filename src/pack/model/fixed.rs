use super::spatial::SpatialGrid;
use crate::model::{
    system::FixedAtomPool,
    types::{TypeIdx, Vec3},
};

/// Sentinel stored in [`FixedAtoms::donor_for_h`]: atom is not a polar H,
/// or its donor heavy atom was not recorded.
pub const NO_DONOR: u32 = u32::MAX;

/// Fixed atoms with a spatial index for neighbor queries.
///
/// Borrows per-atom data from a [`FixedAtomPool`] and adds a [`SpatialGrid`].
pub struct FixedAtoms<'a> {
    /// Per-atom coordinates.
    pub positions: &'a [Vec3],
    /// Per-atom DREIDING type.
    pub types: &'a [TypeIdx],
    /// Per-atom partial charges (e).
    pub charges: &'a [f32],
    /// H -> local donor index ([`NO_DONOR`] = not an H, or no donor recorded).
    pub donor_for_h: &'a [u32],
    /// Spatial index over atom indices.
    grid: SpatialGrid<u32>,
}

impl<'a> FixedAtoms<'a> {
    /// Borrows `pool` and builds the spatial index with `cell_size = vdw_cutoff`.
    pub fn build(pool: &'a FixedAtomPool, vdw_cutoff: f32) -> Self {
        let n = pool.positions.len();
        debug_assert_eq!(pool.types.len(), n, "types/positions length mismatch");
        debug_assert_eq!(pool.charges.len(), n, "charges/positions length mismatch");
        debug_assert_eq!(
            pool.donor_for_h.len(),
            n,
            "donor_for_h/positions length mismatch"
        );
        let grid = SpatialGrid::build(
            pool.positions
                .iter()
                .copied()
                .enumerate()
                .map(|(i, pos)| (pos, i as u32)),
            vdw_cutoff,
        );
        Self {
            positions: &pool.positions,
            types: &pool.types,
            charges: &pool.charges,
            donor_for_h: &pool.donor_for_h,
            grid,
        }
    }

    /// Number of fixed atoms.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns `true` if there are no fixed atoms.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Yields `(position, atom_index)` for every fixed atom within `radius` Å
    /// of `center` (exact Euclidean distance ≤ `radius`).
    pub fn neighbors(&self, center: Vec3, radius: f32) -> impl Iterator<Item = (Vec3, u32)> + '_ {
        self.grid.query(center, radius)
    }
}
