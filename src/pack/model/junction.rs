use arrayvec::ArrayVec;
use std::cmp::Reverse;

/// Junction tree over a connected residue interaction subgraph.
///
/// Bags are in elimination order (index 0 = first eliminated). Each bag
/// eliminates exactly one vertex; its separator (ascending, len ≤ treewidth)
/// lists the vertex's alive neighbors at elimination time.
///
/// Ordering heuristic: MCS (exact for chordal graphs); falls back to
/// min-fill when the graph is non-chordal.
pub struct JunctionTree {
    bags: Vec<Bag>,
    root: u16,
    width: u8,
}

/// Single node in the junction tree.
///
/// Stores the eliminated vertex and its separator. Parent index is always
/// greater than self (root has no parent, empty separator).
pub struct Bag {
    separator: ArrayVec<u32, 5>,
    elim: u32,
    parent: Option<u16>,
}

impl JunctionTree {
    /// Builds a junction tree via elimination on a connected graph.
    ///
    /// `adj[v]` lists neighbors of vertex `v` (0-based, symmetric, no
    /// self-loops). Returns `None` if any separator exceeds `max_width`
    /// entries (caller applies edge-decomposition fallback and retries).
    /// `max_width` must be ≤ 5 (compile-time `ArrayVec` bound).
    ///
    /// # Panics
    ///
    /// Panics if `max_width > 5` or if vertex count exceeds `u16` capacity.
    pub fn build(adj: &[Vec<u32>], max_width: usize) -> Option<Self> {
        assert!(max_width <= 5, "max_width {max_width} exceeds capacity (5)");
        let n = adj.len();
        if n == 0 {
            return Some(Self {
                bags: Vec::new(),
                root: 0,
                width: 0,
            });
        }
        assert!(
            n <= u16::MAX as usize + 1,
            "vertex count {n} exceeds u16 capacity",
        );

        let mut matrix = vec![false; n * n];
        for (v, nbrs) in adj.iter().enumerate() {
            for &u in nbrs {
                debug_assert!((u as usize) < n, "neighbor {u} out of bounds (n={n})",);
                matrix[v * n + u as usize] = true;
            }
        }

        #[cfg(debug_assertions)]
        assert_connected(&matrix, n);

        let order = choose_order(&matrix, n);

        let mut eliminated = vec![false; n];
        let mut bags = Vec::with_capacity(n);
        let mut width: usize = 0;

        for &v in &order {
            let mut sep = ArrayVec::<u32, 5>::new();
            for u in 0..n {
                if u != v && !eliminated[u] && matrix[v * n + u] {
                    if sep.len() >= max_width {
                        return None;
                    }
                    sep.push(u as u32);
                }
            }
            width = width.max(sep.len());

            for i in 0..sep.len() {
                for j in (i + 1)..sep.len() {
                    let a = sep[i] as usize;
                    let b = sep[j] as usize;
                    matrix[a * n + b] = true;
                    matrix[b * n + a] = true;
                }
            }

            eliminated[v] = true;
            bags.push(Bag {
                separator: sep,
                elim: v as u32,
                parent: None,
            });
        }

        let mut vertex_bag = vec![0u16; n];
        for (i, bag) in bags.iter().enumerate() {
            vertex_bag[bag.elim as usize] = i as u16;
        }
        for bag in bags.iter_mut().take(n - 1) {
            let parent = bag
                .separator
                .iter()
                .map(|&v| vertex_bag[v as usize])
                .min()
                .unwrap();
            bag.parent = Some(parent);
        }

        Some(Self {
            root: (n - 1) as u16,
            width: width as u8,
            bags,
        })
    }

    /// Number of bags (= number of vertices).
    pub fn n_bags(&self) -> usize {
        self.bags.len()
    }

    /// Bags in elimination order.
    pub fn bags(&self) -> &[Bag] {
        &self.bags
    }

    /// Root bag index (last eliminated).
    pub fn root(&self) -> u16 {
        self.root
    }

    /// Treewidth (max separator size).
    pub fn width(&self) -> u8 {
        self.width
    }
}

impl Bag {
    /// Separator: alive neighbors at elimination time (ascending).
    pub fn separator(&self) -> &[u32] {
        &self.separator
    }

    /// The vertex eliminated by this bag.
    pub fn elim(&self) -> u32 {
        self.elim
    }

    /// Parent bag index (`None` for root).
    pub fn parent(&self) -> Option<u16> {
        self.parent
    }
}

/// MCS if PEO (chordal -> exact optimal), otherwise min-fill.
fn choose_order(matrix: &[bool], n: usize) -> Vec<usize> {
    let mcs = mcs_order(matrix, n);
    if is_peo(matrix, n, &mcs) {
        mcs
    } else {
        min_fill_order(matrix, n)
    }
}

/// Maximum Cardinality Search.
///
/// At each step, numbers the unnumbered vertex with the most already-numbered
/// neighbors (ties broken by smallest index). Returns the elimination order
/// (sigma=1 first). Produces a PEO iff the graph is chordal.
fn mcs_order(matrix: &[bool], n: usize) -> Vec<usize> {
    let mut numbered = vec![false; n];
    let mut card = vec![0u32; n];
    let mut sigma = vec![0usize; n];

    for i in (1..=n).rev() {
        let v = (0..n)
            .filter(|&v| !numbered[v])
            .min_by_key(|&v| (Reverse(card[v]), v))
            .unwrap();
        sigma[v] = i;
        numbered[v] = true;
        for u in 0..n {
            if !numbered[u] && matrix[v * n + u] {
                card[u] += 1;
            }
        }
    }

    let mut order = vec![0usize; n];
    for v in 0..n {
        order[sigma[v] - 1] = v;
    }
    order
}

/// Returns `true` if `order` is a perfect elimination ordering: for each
/// vertex, the earliest later neighbor is adjacent to all other later
/// neighbors.
fn is_peo(matrix: &[bool], n: usize, order: &[usize]) -> bool {
    let mut pos = vec![0usize; n];
    for (i, &v) in order.iter().enumerate() {
        pos[v] = i;
    }

    for (i, &v) in order.iter().enumerate() {
        let mut f = usize::MAX;
        let mut f_pos = usize::MAX;
        for u in 0..n {
            if matrix[v * n + u] && pos[u] > i && pos[u] < f_pos {
                f_pos = pos[u];
                f = u;
            }
        }
        if f == usize::MAX {
            continue;
        }

        for u in 0..n {
            if u != f && matrix[v * n + u] && pos[u] > i && !matrix[f * n + u] {
                return false;
            }
        }
    }
    true
}

/// Min-fill: always eliminates the vertex whose removal adds the fewest
/// fill-in edges (ties broken by smallest index).
fn min_fill_order(matrix: &[bool], n: usize) -> Vec<usize> {
    let mut adj = matrix.to_vec();
    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);

    for _ in 0..n {
        let v = (0..n)
            .filter(|&v| !eliminated[v])
            .min_by_key(|&v| {
                let mut fill = 0usize;
                for a in 0..n {
                    if a == v || eliminated[a] || !adj[v * n + a] {
                        continue;
                    }
                    for b in (a + 1)..n {
                        if b == v || eliminated[b] || !adj[v * n + b] {
                            continue;
                        }
                        if !adj[a * n + b] {
                            fill += 1;
                        }
                    }
                }
                (fill, v)
            })
            .unwrap();

        for a in 0..n {
            if a == v || eliminated[a] || !adj[v * n + a] {
                continue;
            }
            for b in (a + 1)..n {
                if b == v || eliminated[b] || !adj[v * n + b] {
                    continue;
                }
                adj[a * n + b] = true;
                adj[b * n + a] = true;
            }
        }

        eliminated[v] = true;
        order.push(v);
    }
    order
}

#[cfg(debug_assertions)]
fn assert_connected(matrix: &[bool], n: usize) {
    if n < 2 {
        return;
    }
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    while let Some(v) = stack.pop() {
        for u in 0..n {
            if matrix[v * n + u] && !visited[u] {
                visited[u] = true;
                stack.push(u);
            }
        }
    }
    debug_assert!(visited.iter().all(|&v| v), "input graph is not connected");
}
