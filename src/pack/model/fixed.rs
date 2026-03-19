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
    ///
    /// # Panics
    ///
    /// Panics if atom count exceeds `u32::MAX`, if `vdw_cutoff ≤ 0`, or if
    /// `pool.types`, `pool.charges`, or `pool.donor_for_h` differ in length
    /// from `pool.positions`.
    pub fn build(pool: &'a FixedAtomPool, vdw_cutoff: f32) -> Self {
        let n = pool.positions.len();
        assert!(
            n <= u32::MAX as usize,
            "atom count {n} exceeds u32 capacity"
        );
        assert_eq!(pool.types.len(), n, "types/positions length mismatch");
        assert_eq!(pool.charges.len(), n, "charges/positions length mismatch");
        assert_eq!(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn t(n: u8) -> TypeIdx {
        TypeIdx(n)
    }

    fn linear_pool(n: usize, step: f32) -> FixedAtomPool {
        FixedAtomPool {
            positions: (0..n).map(|i| v(i as f32 * step, 0.0, 0.0)).collect(),
            types: (0..n as u8).map(t).collect(),
            charges: (0..n).map(|i| i as f32 * 0.1).collect(),
            donor_for_h: (0..n as u32)
                .map(|i| if i % 2 == 0 { i + 1 } else { NO_DONOR })
                .collect(),
        }
    }

    fn neighbor_indices(fa: &FixedAtoms<'_>, center: Vec3, radius: f32) -> Vec<u32> {
        let mut out: Vec<u32> = fa.neighbors(center, radius).map(|(_, idx)| idx).collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn build_from_empty_pool_is_empty() {
        let pool = FixedAtomPool {
            positions: vec![],
            types: vec![],
            charges: vec![],
            donor_for_h: vec![],
        };
        let fa = FixedAtoms::build(&pool, 8.0);
        assert!(fa.is_empty());
        assert_eq!(fa.len(), 0);
    }

    #[test]
    fn len_matches_pool_atom_count() {
        let pool = linear_pool(7, 2.0);
        let fa = FixedAtoms::build(&pool, 8.0);
        assert!(!fa.is_empty());
        assert_eq!(fa.len(), 7);
    }

    #[test]
    fn positions_slice_matches_pool() {
        let pool = linear_pool(4, 3.0);
        let fa = FixedAtoms::build(&pool, 8.0);
        assert_eq!(fa.positions, pool.positions.as_slice());
    }

    #[test]
    fn types_slice_matches_pool() {
        let pool = linear_pool(5, 2.0);
        let fa = FixedAtoms::build(&pool, 8.0);
        assert_eq!(fa.types, pool.types.as_slice());
    }

    #[test]
    fn charges_slice_matches_pool() {
        let pool = linear_pool(5, 2.0);
        let fa = FixedAtoms::build(&pool, 8.0);
        assert_eq!(fa.charges, pool.charges.as_slice());
    }

    #[test]
    fn donor_for_h_slice_matches_pool() {
        let pool = linear_pool(6, 2.0);
        let fa = FixedAtoms::build(&pool, 8.0);
        assert_eq!(fa.donor_for_h, pool.donor_for_h.as_slice());
    }

    #[test]
    fn no_donor_sentinel_is_u32_max() {
        assert_eq!(NO_DONOR, u32::MAX);
    }

    #[test]
    fn no_donor_sentinel_round_trips() {
        let pool = FixedAtomPool {
            positions: vec![v(0.0, 0.0, 0.0)],
            types: vec![t(1)],
            charges: vec![0.0],
            donor_for_h: vec![NO_DONOR],
        };
        let fa = FixedAtoms::build(&pool, 8.0);
        assert_eq!(fa.donor_for_h[0], NO_DONOR);
    }

    #[test]
    fn neighbors_returns_all_atoms_with_large_radius() {
        let n = 6usize;
        let pool = linear_pool(n, 1.0);
        let fa = FixedAtoms::build(&pool, 8.0);
        let indices = neighbor_indices(&fa, v(2.5, 0.0, 0.0), 10.0);
        assert_eq!(indices, (0..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn neighbors_returns_correct_subset() {
        let pool = linear_pool(5, 2.0);
        let fa = FixedAtoms::build(&pool, 8.0);
        let indices = neighbor_indices(&fa, v(4.0, 0.0, 0.0), 2.5);
        assert_eq!(indices, vec![1, 2, 3]);
    }

    #[test]
    fn neighbors_includes_atom_at_exact_radius() {
        let pool = FixedAtomPool {
            positions: vec![v(3.0, 0.0, 0.0)],
            types: vec![t(0)],
            charges: vec![0.0],
            donor_for_h: vec![NO_DONOR],
        };
        let fa = FixedAtoms::build(&pool, 8.0);
        assert_eq!(fa.neighbors(Vec3::zero(), 3.0).count(), 1);
    }

    #[test]
    fn neighbors_excludes_atom_just_outside_radius() {
        let pool = FixedAtomPool {
            positions: vec![v(3.01, 0.0, 0.0)],
            types: vec![t(0)],
            charges: vec![0.0],
            donor_for_h: vec![NO_DONOR],
        };
        let fa = FixedAtoms::build(&pool, 8.0);
        assert_eq!(fa.neighbors(Vec3::zero(), 3.0).count(), 0);
    }

    #[test]
    fn empty_fixed_atoms_returns_no_neighbors() {
        let pool = FixedAtomPool {
            positions: vec![],
            types: vec![],
            charges: vec![],
            donor_for_h: vec![],
        };
        let fa = FixedAtoms::build(&pool, 8.0);
        assert_eq!(fa.neighbors(v(0.0, 0.0, 0.0), 100.0).count(), 0);
    }

    #[test]
    fn neighbor_positions_match_stored_positions() {
        let pool = linear_pool(4, 2.0);
        let fa = FixedAtoms::build(&pool, 8.0);
        for (pos, idx) in fa.neighbors(v(3.0, 0.0, 0.0), 10.0) {
            assert_eq!(pos, fa.positions[idx as usize]);
        }
    }

    #[test]
    #[should_panic]
    fn build_panics_on_types_length_mismatch() {
        let pool = FixedAtomPool {
            positions: vec![v(0.0, 0.0, 0.0); 3],
            types: vec![t(0); 2],
            charges: vec![0.0; 3],
            donor_for_h: vec![NO_DONOR; 3],
        };
        FixedAtoms::build(&pool, 8.0);
    }

    #[test]
    #[should_panic]
    fn build_panics_on_charges_length_mismatch() {
        let pool = FixedAtomPool {
            positions: vec![v(0.0, 0.0, 0.0); 3],
            types: vec![t(0); 3],
            charges: vec![0.0; 1],
            donor_for_h: vec![NO_DONOR; 3],
        };
        FixedAtoms::build(&pool, 8.0);
    }

    #[test]
    #[should_panic]
    fn build_panics_on_donor_for_h_length_mismatch() {
        let pool = FixedAtomPool {
            positions: vec![v(0.0, 0.0, 0.0); 3],
            types: vec![t(0); 3],
            charges: vec![0.0; 3],
            donor_for_h: vec![NO_DONOR; 2],
        };
        FixedAtoms::build(&pool, 8.0);
    }
}
