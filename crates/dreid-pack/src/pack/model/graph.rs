/// Residue–residue contact graph for packable slots.
///
/// An edge `(a, b)` with `a < b` means slots `a` and `b` have potential
/// non-zero pair energy.
pub struct ContactGraph {
    adj_offsets: Vec<u32>,
    adj_list: Vec<u32>,
    adj_edge_idx: Vec<u32>,
    edges: Vec<(u32, u32)>,
}

impl ContactGraph {
    /// Constructs a contact graph from raw pairwise contacts.
    ///
    /// Self-loops are silently dropped. Each pair is canonicalized so the
    /// smaller index comes first; duplicates and reversed copies are removed.
    ///
    /// # Panics
    ///
    /// Panics if `n_slots` exceeds `u32::MAX + 1`.
    pub fn build(n_slots: usize, raw_edges: impl IntoIterator<Item = (u32, u32)>) -> Self {
        assert!(
            n_slots <= u32::MAX as usize + 1,
            "n_slots {n_slots} exceeds u32 capacity",
        );

        let mut edges: Vec<(u32, u32)> = raw_edges
            .into_iter()
            .filter(|&(a, b)| a != b)
            .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
            .collect();

        edges.sort_unstable();
        edges.dedup();

        debug_assert!(
            edges
                .iter()
                .all(|&(a, b)| (a as usize) < n_slots && (b as usize) < n_slots),
            "edge slot index out of bounds for n_slots={n_slots}",
        );

        let mut degree = vec![0u32; n_slots];
        for &(a, b) in &edges {
            degree[a as usize] += 1;
            degree[b as usize] += 1;
        }

        let mut adj_offsets = vec![0u32; n_slots + 1];
        for s in 0..n_slots {
            adj_offsets[s + 1] = adj_offsets[s] + degree[s];
        }

        let total_neighbors = adj_offsets[n_slots] as usize;
        let mut adj_list = vec![0u32; total_neighbors];
        let mut adj_edge_idx = vec![0u32; total_neighbors];
        let mut cursor = degree;
        cursor[..n_slots].copy_from_slice(&adj_offsets[..n_slots]);
        for (edge_idx, &(a, b)) in edges.iter().enumerate() {
            let pa = cursor[a as usize] as usize;
            adj_list[pa] = b;
            adj_edge_idx[pa] = edge_idx as u32;
            cursor[a as usize] += 1;

            let pb = cursor[b as usize] as usize;
            adj_list[pb] = a;
            adj_edge_idx[pb] = edge_idx as u32;
            cursor[b as usize] += 1;
        }

        Self {
            adj_offsets,
            adj_list,
            adj_edge_idx,
            edges,
        }
    }

    /// Number of packable slots.
    pub fn n_slots(&self) -> usize {
        self.adj_offsets.len() - 1
    }

    /// Number of contact edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Returns `(neighbor, edge_idx, s_is_left)` for each slot adjacent to `s`,
    /// where `s_is_left` is `true` when `s < neighbor`.
    pub fn neighbor_edges(&self, s: usize) -> impl Iterator<Item = (u32, u32, bool)> + '_ {
        debug_assert!(
            s < self.n_slots(),
            "slot {s} out of bounds (n_slots={})",
            self.n_slots(),
        );
        let start = self.adj_offsets[s] as usize;
        let end = self.adj_offsets[s + 1] as usize;
        self.adj_list[start..end]
            .iter()
            .zip(&self.adj_edge_idx[start..end])
            .map(move |(&j, &e)| (j, e, (s as u32) < j))
    }

    /// Returns the list of edges as `(a, b)` pairs with `a < b`,
    /// and sorted lexicographically.
    pub fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neighbor_edges_sorted(g: &ContactGraph, s: usize) -> Vec<(u32, u32, bool)> {
        let mut v: Vec<_> = g.neighbor_edges(s).collect();
        v.sort_unstable_by_key(|&(j, _, _)| j);
        v
    }

    fn triangle() -> ContactGraph {
        ContactGraph::build(3, [(0, 1), (0, 2), (1, 2)])
    }

    #[test]
    fn build_zero_slots_zero_edges() {
        let g = ContactGraph::build(0, []);
        assert_eq!(g.n_slots(), 0);
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn n_slots_matches_argument() {
        let g = ContactGraph::build(5, []);
        assert_eq!(g.n_slots(), 5);
    }

    #[test]
    fn n_edges_zero_when_no_edges() {
        let g = ContactGraph::build(4, []);
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn self_loop_is_silently_dropped() {
        let g = ContactGraph::build(3, [(1, 1), (0, 1)]);
        assert_eq!(g.n_edges(), 1);
    }

    #[test]
    fn reversed_edge_is_canonicalized() {
        let g = ContactGraph::build(4, [(3u32, 1u32)]);
        assert_eq!(g.edges(), &[(1, 3)]);
    }

    #[test]
    fn duplicate_edge_is_deduplicated() {
        let g = ContactGraph::build(3, [(0, 1), (0, 1)]);
        assert_eq!(g.n_edges(), 1);
    }

    #[test]
    fn reversed_duplicate_is_deduplicated() {
        let g = ContactGraph::build(3, [(0, 2), (2, 0)]);
        assert_eq!(g.n_edges(), 1);
    }

    #[test]
    fn all_filters_combined() {
        let g = ContactGraph::build(4, [(0, 0), (2, 1), (1, 2), (0, 3)]);
        assert_eq!(g.edges(), &[(0, 3), (1, 2)]);
    }

    #[test]
    fn edges_are_sorted_lexicographically() {
        let g = ContactGraph::build(4, [(2u32, 3u32), (0, 1), (1, 3), (0, 2)]);
        assert_eq!(g.edges(), &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn edges_content_is_exact() {
        let g = ContactGraph::build(3, [(1u32, 0u32), (2, 1)]);
        assert_eq!(g.edges(), &[(0, 1), (1, 2)]);
    }

    #[test]
    fn neighbors_are_symmetric() {
        let g = ContactGraph::build(4, [(0u32, 3u32), (1, 2)]);
        assert!(g.neighbor_edges(0).any(|(j, _, _)| j == 3));
        assert!(g.neighbor_edges(3).any(|(j, _, _)| j == 0));
        assert!(g.neighbor_edges(1).any(|(j, _, _)| j == 2));
        assert!(g.neighbor_edges(2).any(|(j, _, _)| j == 1));
    }

    #[test]
    fn isolated_slot_has_empty_neighbors() {
        let g = ContactGraph::build(5, [(0, 1)]);
        assert!(g.neighbor_edges(4).next().is_none());
    }

    #[test]
    fn triangle_has_three_edges() {
        assert_eq!(triangle().n_edges(), 3);
    }

    #[test]
    fn triangle_each_slot_has_two_neighbors() {
        let g = triangle();
        let nbrs = |s| -> Vec<u32> {
            let mut v: Vec<u32> = g.neighbor_edges(s).map(|(j, _, _)| j).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(nbrs(0), vec![1u32, 2]);
        assert_eq!(nbrs(1), vec![0u32, 2]);
        assert_eq!(nbrs(2), vec![0u32, 1]);
    }

    #[test]
    fn neighbor_edges_triangle_each_slot_correct() {
        let g = triangle();
        assert_eq!(
            neighbor_edges_sorted(&g, 0),
            vec![(1, 0, true), (2, 1, true)]
        );
        assert_eq!(
            neighbor_edges_sorted(&g, 1),
            vec![(0, 0, false), (2, 2, true)]
        );
        assert_eq!(
            neighbor_edges_sorted(&g, 2),
            vec![(0, 1, false), (1, 2, false)]
        );
    }

    #[test]
    fn neighbor_edges_isolated_slot_is_empty() {
        let g = ContactGraph::build(5, [(0u32, 1u32)]);
        assert_eq!(g.neighbor_edges(4).count(), 0);
    }

    #[test]
    fn neighbor_edges_edge_idx_consistent_with_edges() {
        let g = ContactGraph::build(5, [(1u32, 4u32), (2, 3)]);
        for s in 0..g.n_slots() {
            for (j, e, is_left) in g.neighbor_edges(s) {
                let (a, b) = g.edges()[e as usize];
                if is_left {
                    assert_eq!((a, b), (s as u32, j));
                } else {
                    assert_eq!((a, b), (j, s as u32));
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn build_panics_on_n_slots_overflow() {
        ContactGraph::build(u32::MAX as usize + 2, []);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn neighbor_edges_panics_out_of_bounds() {
        let g = ContactGraph::build(3, []);
        let _ = g.neighbor_edges(3);
    }
}
