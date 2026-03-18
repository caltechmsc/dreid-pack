use crate::model::types::Vec3;

/// Sentinel marking the end of a linked list.
const SENTINEL: u32 = u32::MAX;

/// Uniform cubic spatial grid.
///
/// Build once with [`SpatialGrid::build`]; query repeatedly with
/// [`SpatialGrid::query`]. All coordinates use [`Vec3`] (`f32` components).
pub struct SpatialGrid<T> {
    cell_size: f32,
    origin: Vec3,
    dims: [usize; 3],
    head: Vec<u32>,
    next: Vec<u32>,
    items: Vec<(Vec3, T)>,
}

impl<T: Copy> SpatialGrid<T> {
    /// Constructs a grid from `(position, payload)` pairs.
    ///
    /// `cell_size` should equal the expected query radius for optimal performance
    /// (27-cell worst-case per query at that ratio).
    ///
    /// # Panics
    ///
    /// Panics if `cell_size ≤ 0`.
    pub fn build(items: impl IntoIterator<Item = (Vec3, T)>, cell_size: f32) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");

        let items: Vec<(Vec3, T)> = items.into_iter().collect();

        if items.is_empty() {
            return Self {
                cell_size,
                origin: Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                dims: [0, 0, 0],
                head: Vec::new(),
                next: Vec::new(),
                items: Vec::new(),
            };
        }

        let mut min = Vec3 {
            x: f32::MAX,
            y: f32::MAX,
            z: f32::MAX,
        };
        let mut max = Vec3 {
            x: f32::MIN,
            y: f32::MIN,
            z: f32::MIN,
        };
        for &(pos, _) in &items {
            if pos.x < min.x {
                min.x = pos.x;
            }
            if pos.y < min.y {
                min.y = pos.y;
            }
            if pos.z < min.z {
                min.z = pos.z;
            }
            if pos.x > max.x {
                max.x = pos.x;
            }
            if pos.y > max.y {
                max.y = pos.y;
            }
            if pos.z > max.z {
                max.z = pos.z;
            }
        }

        let eps = cell_size * 1e-4;
        let dims = [
            (((max.x + eps - min.x) / cell_size).ceil() as usize).max(1),
            (((max.y + eps - min.y) / cell_size).ceil() as usize).max(1),
            (((max.z + eps - min.z) / cell_size).ceil() as usize).max(1),
        ];

        let num_cells = dims[0] * dims[1] * dims[2];
        let num_items = items.len();

        let mut head = vec![SENTINEL; num_cells];
        let mut next = vec![SENTINEL; num_items];
        let mut stored: Vec<(Vec3, T)> = Vec::with_capacity(num_items);

        for (i, (pos, payload)) in items.into_iter().enumerate() {
            if let Some(ci) = cell_index_of(pos, min, cell_size, dims) {
                next[i] = head[ci];
                head[ci] = i as u32;
            }
            stored.push((pos, payload));
        }

        Self {
            cell_size,
            origin: min,
            dims,
            head,
            next,
            items: stored,
        }
    }

    /// Exact sphere query: yields every `(position, payload)` with
    /// `dist(position, center) ≤ radius`.
    pub fn query(&self, center: Vec3, radius: f32) -> impl Iterator<Item = (Vec3, T)> + '_ {
        let r2 = radius * radius;

        let (lo, hi, init_item) = if self.items.is_empty() {
            ([0usize; 3], [0usize; 3], SENTINEL)
        } else {
            let (lo, hi) = self.cell_range(center, radius);
            let ci = lo[0] + lo[1] * self.dims[0] + lo[2] * self.dims[0] * self.dims[1];
            (lo, hi, self.head[ci])
        };

        QueryIter {
            grid: self,
            lo,
            hi,
            cx: lo[0],
            cy: lo[1],
            cz: lo[2],
            item_idx: init_item,
            center,
            radius_sq: r2,
        }
    }

    /// Number of items stored in the grid.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the grid contains no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
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
    item_idx: u32,
    center: Vec3,
    radius_sq: f32,
}

impl<T: Copy> Iterator for QueryIter<'_, T> {
    type Item = (Vec3, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.item_idx != SENTINEL {
                let idx = self.item_idx as usize;
                self.item_idx = self.grid.next[idx];
                let (pos, val) = self.grid.items[idx];
                if pos.dist_sq(self.center) <= self.radius_sq {
                    return Some((pos, val));
                }
                continue;
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
            self.item_idx = self.grid.head[ci];
        }
    }
}

impl<T: Copy> std::iter::FusedIterator for QueryIter<'_, T> {}
