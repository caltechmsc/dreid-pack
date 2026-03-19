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

#[cfg(test)]
mod tests {
    use super::*;

    fn adj_from_edges(n: usize, edges: &[(u32, u32)]) -> Vec<Vec<u32>> {
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        }
        adj
    }

    fn complete(n: usize) -> Vec<Vec<u32>> {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i].push(j as u32);
                }
            }
        }
        adj
    }

    fn path(n: usize) -> Vec<Vec<u32>> {
        let edges: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
        adj_from_edges(n, &edges)
    }

    fn cycle(n: usize) -> Vec<Vec<u32>> {
        let mut edges: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
        edges.push((n as u32 - 1, 0));
        adj_from_edges(n, &edges)
    }

    fn star(n: usize) -> Vec<Vec<u32>> {
        let edges: Vec<(u32, u32)> = (1..n as u32).map(|i| (0, i)).collect();
        adj_from_edges(n, &edges)
    }

    fn bag_total(bag: &Bag) -> Vec<u32> {
        let mut total: Vec<u32> = bag.separator().to_vec();
        total.push(bag.elim());
        total
    }

    fn check_running_intersection(tree: &JunctionTree) {
        let n = tree.n_bags();
        for v in 0..n as u32 {
            let bag_ids: Vec<usize> = tree
                .bags()
                .iter()
                .enumerate()
                .filter(|(_, b)| b.elim() == v || b.separator().contains(&v))
                .map(|(i, _)| i)
                .collect();
            if bag_ids.len() <= 1 {
                continue;
            }
            let mut visited = vec![false; n];
            let mut stack = vec![bag_ids[0]];
            visited[bag_ids[0]] = true;
            while let Some(bi) = stack.pop() {
                let b = &tree.bags()[bi];
                if let Some(p) = b.parent() {
                    let p = p as usize;
                    if !visited[p] && bag_ids.contains(&p) {
                        visited[p] = true;
                        stack.push(p);
                    }
                }
                for (ci, cb) in tree.bags().iter().enumerate() {
                    if cb.parent() == Some(bi as u16) && !visited[ci] && bag_ids.contains(&ci) {
                        visited[ci] = true;
                        stack.push(ci);
                    }
                }
            }
            let reachable = bag_ids.iter().filter(|&&i| visited[i]).count();
            assert_eq!(reachable, bag_ids.len(), "vertex {v}: bags not connected");
        }
    }

    #[test]
    fn empty_graph_yields_empty_tree() {
        let tree = JunctionTree::build(&[], 5).unwrap();
        assert_eq!(tree.n_bags(), 0);
        assert_eq!(tree.width(), 0);
    }

    #[test]
    fn single_vertex_has_width_zero() {
        let tree = JunctionTree::build(&[vec![]], 5).unwrap();
        assert_eq!(tree.n_bags(), 1);
        assert_eq!(tree.width(), 0);
        assert_eq!(tree.bags()[0].elim(), 0);
        assert!(tree.bags()[0].separator().is_empty());
    }

    #[test]
    fn two_vertices_have_width_one() {
        let tree = JunctionTree::build(&adj_from_edges(2, &[(0, 1)]), 5).unwrap();
        assert_eq!(tree.n_bags(), 2);
        assert_eq!(tree.width(), 1);
    }

    #[test]
    fn path_of_five_has_width_one() {
        let tree = JunctionTree::build(&path(5), 5).unwrap();
        assert_eq!(tree.width(), 1);
    }

    #[test]
    fn triangle_has_width_two() {
        let tree = JunctionTree::build(&complete(3), 5).unwrap();
        assert_eq!(tree.width(), 2);
    }

    #[test]
    fn cycle_of_four_has_width_two() {
        let tree = JunctionTree::build(&cycle(4), 5).unwrap();
        assert_eq!(tree.width(), 2);
    }

    #[test]
    fn cycle_of_six_has_width_two() {
        let tree = JunctionTree::build(&cycle(6), 5).unwrap();
        assert_eq!(tree.width(), 2);
    }

    #[test]
    fn complete_k4_has_width_three() {
        let tree = JunctionTree::build(&complete(4), 5).unwrap();
        assert_eq!(tree.width(), 3);
    }

    #[test]
    fn complete_k5_has_width_four() {
        let tree = JunctionTree::build(&complete(5), 5).unwrap();
        assert_eq!(tree.width(), 4);
    }

    #[test]
    fn complete_k6_has_width_five() {
        let tree = JunctionTree::build(&complete(6), 5).unwrap();
        assert_eq!(tree.width(), 5);
    }

    #[test]
    fn complete_k7_returns_none() {
        assert!(JunctionTree::build(&complete(7), 5).is_none());
    }

    #[test]
    fn star_of_five_has_width_one() {
        let tree = JunctionTree::build(&star(5), 5).unwrap();
        assert_eq!(tree.width(), 1);
    }

    #[test]
    fn stricter_max_width_rejects_earlier() {
        assert!(JunctionTree::build(&complete(5), 4).is_some());
        assert!(JunctionTree::build(&complete(6), 4).is_none());
    }

    #[test]
    #[should_panic]
    fn max_width_above_five_panics() {
        let _ = JunctionTree::build(&[], 6);
    }

    #[test]
    fn root_is_last_bag() {
        let tree = JunctionTree::build(&path(4), 5).unwrap();
        assert_eq!(tree.root() as usize, tree.n_bags() - 1);
    }

    #[test]
    fn root_has_no_parent_and_empty_separator() {
        let tree = JunctionTree::build(&complete(4), 5).unwrap();
        let root = &tree.bags()[tree.root() as usize];
        assert!(root.parent().is_none());
        assert!(root.separator().is_empty());
    }

    #[test]
    fn non_root_bags_have_parent() {
        let tree = JunctionTree::build(&cycle(5), 5).unwrap();
        for (i, bag) in tree.bags().iter().enumerate() {
            if i != tree.root() as usize {
                assert!(bag.parent().is_some(), "bag {i} has no parent");
            }
        }
    }

    #[test]
    fn parent_index_exceeds_self() {
        let tree = JunctionTree::build(&cycle(6), 5).unwrap();
        for (i, bag) in tree.bags().iter().enumerate() {
            if let Some(p) = bag.parent() {
                assert!((p as usize) > i, "bag {i} parent {p} not greater than self",);
            }
        }
    }

    #[test]
    fn all_vertices_eliminated_exactly_once() {
        let tree = JunctionTree::build(&complete(5), 5).unwrap();
        let mut seen = vec![false; 5];
        for bag in tree.bags() {
            let v = bag.elim() as usize;
            assert!(!seen[v], "vertex {v} eliminated twice");
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn separators_are_ascending() {
        let tree = JunctionTree::build(&complete(5), 5).unwrap();
        for bag in tree.bags() {
            let sep = bag.separator();
            for w in sep.windows(2) {
                assert!(w[0] < w[1], "separator not ascending: {:?}", sep);
            }
        }
    }

    #[test]
    fn separator_subset_of_parent_total() {
        let tree = JunctionTree::build(&cycle(5), 5).unwrap();
        for bag in tree.bags() {
            if let Some(p) = bag.parent() {
                let parent = &tree.bags()[p as usize];
                for &v in bag.separator() {
                    assert!(
                        parent.separator().contains(&v) || parent.elim() == v,
                        "separator vertex {v} not in parent bag",
                    );
                }
            }
        }
    }

    #[test]
    fn all_edges_covered_by_some_bag() {
        let adj = cycle(6);
        let tree = JunctionTree::build(&adj, 5).unwrap();
        for (u, nbrs) in adj.iter().enumerate() {
            for &v in nbrs {
                if (u as u32) < v {
                    let covered = tree.bags().iter().any(|bag| {
                        let t = bag_total(bag);
                        t.contains(&(u as u32)) && t.contains(&v)
                    });
                    assert!(covered, "edge ({u}, {v}) not covered");
                }
            }
        }
    }

    #[test]
    fn all_edges_covered_in_k5() {
        let adj = complete(5);
        let tree = JunctionTree::build(&adj, 5).unwrap();
        for (u, nbrs) in adj.iter().enumerate() {
            for &v in nbrs {
                if (u as u32) < v {
                    let covered = tree.bags().iter().any(|bag| {
                        let t = bag_total(bag);
                        t.contains(&(u as u32)) && t.contains(&v)
                    });
                    assert!(covered, "edge ({u}, {v}) not covered");
                }
            }
        }
    }

    #[test]
    fn running_intersection_holds() {
        check_running_intersection(&JunctionTree::build(&cycle(6), 5).unwrap());
    }

    #[test]
    fn running_intersection_holds_on_k5() {
        check_running_intersection(&JunctionTree::build(&complete(5), 5).unwrap());
    }

    #[test]
    fn chordal_graph_gets_optimal_width() {
        let adj = adj_from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]);
        let tree = JunctionTree::build(&adj, 5).unwrap();
        assert_eq!(tree.width(), 2);
    }

    #[test]
    fn non_chordal_cycle_four_gets_optimal_width() {
        let tree = JunctionTree::build(&cycle(4), 5).unwrap();
        assert_eq!(tree.width(), 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn build_panics_on_out_of_bounds_neighbor() {
        let _ = JunctionTree::build(&[vec![5]], 5);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn build_panics_on_disconnected_graph() {
        let _ = JunctionTree::build(&[vec![], vec![]], 5);
    }
}
