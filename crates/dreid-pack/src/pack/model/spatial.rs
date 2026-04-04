use crate::model::types::Vec3;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Uniform cubic spatial grid.
///
/// Build once with [`SpatialGrid::build`]; query repeatedly with
/// [`SpatialGrid::query`]. All coordinates use [`Vec3`] (`f32` components).
pub struct SpatialGrid<T> {
    cell_size: f32,
    origin: Vec3,
    dims: [usize; 3],
    offsets: Vec<u32>,
    items: Vec<(Vec3, T)>,
}

impl<T: Copy> SpatialGrid<T> {
    /// Constructs a grid from `(position, payload)` pairs.
    ///
    /// `cell_size` should equal the expected query radius for optimal
    /// performance (27-cell worst-case per query at that ratio).
    ///
    /// # Panics
    ///
    /// Panics if `cell_size ≤ 0`, if item count exceeds `u32::MAX`, or if the
    /// resulting grid dimensions overflow `usize` (box too large for `cell_size`).
    pub fn build(items: impl IntoIterator<Item = (Vec3, T)>, cell_size: f32) -> Self
    where
        T: Send + Sync,
    {
        assert!(cell_size > 0.0, "cell_size must be positive");

        let items: Vec<(Vec3, T)> = items.into_iter().collect();
        let n = items.len();

        assert!(
            n <= u32::MAX as usize,
            "item count {n} exceeds u32 capacity"
        );

        if items.is_empty() {
            return Self {
                cell_size,
                origin: Vec3::zero(),
                dims: [0, 0, 0],
                offsets: vec![0],
                items,
            };
        }

        let (min, max) = items.par_iter().map(|(p, _)| (*p, *p)).reduce(
            || (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(a_min, a_max), (b_min, b_max)| {
                (
                    Vec3::new(
                        a_min.x.min(b_min.x),
                        a_min.y.min(b_min.y),
                        a_min.z.min(b_min.z),
                    ),
                    Vec3::new(
                        a_max.x.max(b_max.x),
                        a_max.y.max(b_max.y),
                        a_max.z.max(b_max.z),
                    ),
                )
            },
        );

        let eps = cell_size * 1e-4;
        let dims = [
            (((max.x + eps - min.x) / cell_size).ceil() as usize).max(1),
            (((max.y + eps - min.y) / cell_size).ceil() as usize).max(1),
            (((max.z + eps - min.z) / cell_size).ceil() as usize).max(1),
        ];
        let num_cells = dims[0]
            .checked_mul(dims[1])
            .and_then(|v| v.checked_mul(dims[2]))
            .unwrap_or_else(|| {
                panic!(
                    "spatial grid too large: dims={dims:?} (box extent too large or cell_size={cell_size} too small)"
                )
            });

        let atomic_counts: Vec<AtomicU32> = (0..num_cells).map(|_| AtomicU32::new(0)).collect();
        items.par_iter().for_each(|(pos, _)| {
            if let Some(ci) = cell_index_of(*pos, min, cell_size, dims) {
                atomic_counts[ci].fetch_add(1, Ordering::Relaxed);
            }
        });

        let mut offsets = vec![0u32; num_cells + 1];
        for c in 0..num_cells {
            offsets[c + 1] = offsets[c] + atomic_counts[c].load(Ordering::Relaxed);
        }

        let total = offsets[num_cells] as usize;
        let mut sorted_uninit = Vec::<std::mem::MaybeUninit<(Vec3, T)>>::with_capacity(total);
        // SAFETY: MaybeUninit<_> carries no validity invariant; filling with
        // uninitialized bytes is always sound.
        unsafe { sorted_uninit.set_len(total) };
        for c in 0..num_cells {
            atomic_counts[c].store(offsets[c], Ordering::Relaxed);
        }
        let sorted_ptr = sorted_uninit.as_mut_ptr() as usize;
        items.par_iter().for_each(|(pos, val)| {
            if let Some(ci) = cell_index_of(*pos, min, cell_size, dims) {
                let slot = atomic_counts[ci].fetch_add(1, Ordering::Relaxed) as usize;
                // SAFETY: `fetch_add` gives each thread a unique slot within
                // `[offsets[ci], offsets[ci+1])`. All cell ranges are disjoint,
                // so no two threads write the same index.
                unsafe { (sorted_ptr as *mut (Vec3, T)).add(slot).write((*pos, *val)) };
            }
        });
        // SAFETY: every element has been initialized by the parallel loop above.
        let sorted = unsafe {
            std::vec::Vec::from_raw_parts(
                sorted_uninit.as_mut_ptr() as *mut (Vec3, T),
                total,
                total,
            )
        };
        std::mem::forget(sorted_uninit);

        Self {
            cell_size,
            origin: min,
            dims,
            offsets,
            items: sorted,
        }
    }

    /// Exact sphere query: yields every `(position, payload)` with
    /// `dist(position, center) ≤ radius`.
    pub fn query(&self, center: Vec3, radius: f32) -> impl Iterator<Item = (Vec3, T)> + '_ {
        debug_assert!(
            radius >= 0.0,
            "query radius must be non-negative, got {radius}"
        );
        let r2 = radius * radius;

        if self.items.is_empty() {
            return QueryIter {
                grid: self,
                lo: [0; 3],
                hi: [0; 3],
                cx: 0,
                cy: 0,
                cz: 1,
                cell_pos: 0,
                cell_end: 0,
                center,
                radius_sq: r2,
            };
        }

        let (lo, hi) = self.cell_range(center, radius);
        let first_ci = lo[0] + lo[1] * self.dims[0] + lo[2] * self.dims[0] * self.dims[1];
        QueryIter {
            grid: self,
            lo,
            hi,
            cx: lo[0],
            cy: lo[1],
            cz: lo[2],
            cell_pos: self.offsets[first_ci],
            cell_end: self.offsets[first_ci + 1],
            center,
            radius_sq: r2,
        }
    }

    /// Computes the `[lo, hi]` cell-index range that the AABB
    /// `[center ± radius]` overlaps, clamped to `[0, dims-1]`.
    fn cell_range(&self, center: Vec3, radius: f32) -> ([usize; 3], [usize; 3]) {
        let clamp = |v: isize, dim: usize| -> usize {
            if v < 0 {
                0
            } else if v as usize >= dim {
                dim - 1
            } else {
                v as usize
            }
        };

        let lo = [
            clamp(
                ((center.x - radius - self.origin.x) / self.cell_size).floor() as isize,
                self.dims[0],
            ),
            clamp(
                ((center.y - radius - self.origin.y) / self.cell_size).floor() as isize,
                self.dims[1],
            ),
            clamp(
                ((center.z - radius - self.origin.z) / self.cell_size).floor() as isize,
                self.dims[2],
            ),
        ];
        let hi = [
            clamp(
                ((center.x + radius - self.origin.x) / self.cell_size).floor() as isize,
                self.dims[0],
            ),
            clamp(
                ((center.y + radius - self.origin.y) / self.cell_size).floor() as isize,
                self.dims[1],
            ),
            clamp(
                ((center.z + radius - self.origin.z) / self.cell_size).floor() as isize,
                self.dims[2],
            ),
        ];

        (lo, hi)
    }
}

/// Flat cell index for `pos`, or `None` if `pos` lies outside the grid.
fn cell_index_of(pos: Vec3, origin: Vec3, cell_size: f32, dims: [usize; 3]) -> Option<usize> {
    let d = pos - origin;
    let xi = (d.x / cell_size).floor() as isize;
    let yi = (d.y / cell_size).floor() as isize;
    let zi = (d.z / cell_size).floor() as isize;

    if xi < 0
        || xi as usize >= dims[0]
        || yi < 0
        || yi as usize >= dims[1]
        || zi < 0
        || zi as usize >= dims[2]
    {
        return None;
    }

    Some(xi as usize + yi as usize * dims[0] + zi as usize * dims[0] * dims[1])
}

/// Yields `(position, payload)` pairs within the query sphere.
struct QueryIter<'a, T> {
    grid: &'a SpatialGrid<T>,
    lo: [usize; 3],
    hi: [usize; 3],
    cx: usize,
    cy: usize,
    cz: usize,
    cell_pos: u32,
    cell_end: u32,
    center: Vec3,
    radius_sq: f32,
}

impl<T: Copy> Iterator for QueryIter<'_, T> {
    type Item = (Vec3, T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while self.cell_pos < self.cell_end {
                let (pos, val) = self.grid.items[self.cell_pos as usize];
                self.cell_pos += 1;
                if pos.dist_sq(self.center) <= self.radius_sq {
                    return Some((pos, val));
                }
            }

            self.cx += 1;
            if self.cx > self.hi[0] {
                self.cx = self.lo[0];
                self.cy += 1;
            }
            if self.cy > self.hi[1] {
                self.cy = self.lo[1];
                self.cz += 1;
            }
            if self.cz > self.hi[2] {
                return None;
            }

            let ci = self.cx
                + self.cy * self.grid.dims[0]
                + self.cz * self.grid.dims[0] * self.grid.dims[1];
            self.cell_pos = self.grid.offsets[ci];
            self.cell_end = self.grid.offsets[ci + 1];
        }
    }
}

impl<T: Copy> std::iter::FusedIterator for QueryIter<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    fn query_payloads(grid: &SpatialGrid<u32>, center: Vec3, radius: f32) -> Vec<u32> {
        let mut out: Vec<u32> = grid.query(center, radius).map(|(_, p)| p).collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn all_items_queryable_after_build() {
        let items: Vec<_> = (0u32..10).map(|i| (v(i as f32, 0.0, 0.0), i)).collect();
        let g = SpatialGrid::build(items, 2.0);
        assert_eq!(g.query(v(5.0, 0.0, 0.0), 200.0).count(), 10);
    }

    #[test]
    fn empty_grid_returns_nothing() {
        let g = SpatialGrid::<u32>::build(std::iter::empty(), 5.0);
        assert_eq!(g.query(v(0.0, 0.0, 0.0), 100.0).count(), 0);
    }

    #[test]
    fn single_item_in_range() {
        let g = SpatialGrid::build([(v(1.0, 0.0, 0.0), 42u32)], 5.0);
        let res: Vec<_> = g.query(v(0.0, 0.0, 0.0), 2.0).collect();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].1, 42);
    }

    #[test]
    fn single_item_out_of_range() {
        let g = SpatialGrid::build([(v(10.0, 0.0, 0.0), 1u32)], 5.0);
        assert_eq!(g.query(v(0.0, 0.0, 0.0), 2.0).count(), 0);
    }

    #[test]
    fn item_at_exact_radius_is_included() {
        let g = SpatialGrid::build([(v(3.0, 4.0, 0.0), 7u32)], 5.0);
        assert_eq!(g.query(v(0.0, 0.0, 0.0), 5.0).count(), 1);
    }

    #[test]
    fn item_just_outside_radius_is_excluded() {
        let g = SpatialGrid::build([(v(3.0, 4.0, 0.0), 7u32)], 5.0);
        assert_eq!(g.query(v(0.0, 0.0, 0.0), 4.999).count(), 0);
    }

    #[test]
    fn partial_hit_returns_correct_subset() {
        let items = vec![
            (v(1.0, 0.0, 0.0), 1u32),
            (v(5.0, 0.0, 0.0), 2u32),
            (v(6.0, 0.0, 0.0), 3u32),
            (v(0.0, 0.0, 0.0), 4u32),
        ];
        let g = SpatialGrid::build(items, 5.0);
        assert_eq!(query_payloads(&g, v(0.0, 0.0, 0.0), 5.0), vec![1, 2, 4]);
    }

    #[test]
    fn all_items_found_with_large_radius() {
        let items: Vec<_> = (0u32..50).map(|i| (v(i as f32, 0.0, 0.0), i)).collect();
        let g = SpatialGrid::build(items, 10.0);
        let got = query_payloads(&g, v(25.0, 0.0, 0.0), 1_000.0);
        let expected: Vec<u32> = (0..50).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn items_across_many_cells_all_reachable() {
        let items: Vec<_> = (0u32..27)
            .map(|i| {
                let x = (i % 3) as f32 * 10.0;
                let y = ((i / 3) % 3) as f32 * 10.0;
                let z = (i / 9) as f32 * 10.0;
                (v(x, y, z), i)
            })
            .collect();
        let g = SpatialGrid::build(items, 5.0);

        let result = query_payloads(&g, v(10.0, 10.0, 10.0), 15.0);
        assert!(result.contains(&13));
        assert!(!result.contains(&0));
    }

    #[test]
    fn query_center_outside_grid_still_finds_items() {
        let g = SpatialGrid::build([(v(5.0, 5.0, 5.0), 99u32)], 5.0);
        let res: Vec<_> = g.query(v(-100.0, -100.0, -100.0), 200.0).collect();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].1, 99);
    }

    #[test]
    fn no_duplicates_in_results() {
        let items = vec![(v(0.0, 0.0, 0.0), 1u32), (v(0.1, 0.0, 0.0), 2u32)];
        let g = SpatialGrid::build(items, 1.0);
        let results: Vec<u32> = g.query(v(0.0, 0.0, 0.0), 0.5).map(|(_, v)| v).collect();
        let unique: std::collections::HashSet<u32> = results.iter().copied().collect();
        assert_eq!(results.len(), unique.len());
    }

    #[test]
    fn fused_iterator_returns_none_repeatedly() {
        let g = SpatialGrid::build([(v(0.0, 0.0, 0.0), 1u32)], 1.0);
        let mut it = g.query(v(100.0, 0.0, 0.0), 0.1);
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn works_with_u16_payload() {
        let g = SpatialGrid::build([(v(0.0, 0.0, 0.0), 7u16)], 1.0);
        let res: Vec<u16> = g.query(v(0.0, 0.0, 0.0), 1.0).map(|(_, v)| v).collect();
        assert_eq!(res, vec![7u16]);
    }

    #[test]
    fn atom_at_aabb_maximum_is_stored_and_queryable() {
        let g = SpatialGrid::build([(v(0.0, 0.0, 0.0), 0u32), (v(10.0, 10.0, 10.0), 1u32)], 5.0);
        assert_eq!(g.query(v(10.0, 10.0, 10.0), 0.1).count(), 1);
    }

    #[test]
    fn all_items_collinear_single_axis() {
        let items: Vec<_> = (0u32..5).map(|i| (v(i as f32, 0.0, 0.0), i)).collect();
        let g = SpatialGrid::build(items, 1.5);
        let got = query_payloads(&g, v(2.0, 0.0, 0.0), 1.5);
        assert_eq!(got, vec![1, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn build_panics_on_zero_cell_size() {
        SpatialGrid::<u32>::build(std::iter::empty(), 0.0);
    }

    #[test]
    #[should_panic]
    fn build_panics_on_negative_cell_size() {
        SpatialGrid::<u32>::build(std::iter::empty(), -1.0);
    }

    #[test]
    #[should_panic]
    fn build_panics_on_grid_dim_overflow() {
        SpatialGrid::<u32>::build(
            [
                (v(-f32::MAX, -f32::MAX, -f32::MAX), 0u32),
                (v(f32::MAX, f32::MAX, f32::MAX), 1u32),
            ],
            1.0,
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn query_panics_on_negative_radius() {
        let g = SpatialGrid::build([(v(0.0, 0.0, 0.0), 0u32)], 1.0);
        let _ = g.query(v(0.0, 0.0, 0.0), -1.0).count();
    }
}
