/// Residue–residue contact graph for packable slots.
///
/// An edge `(a, b)` with `a < b` means slots `a` and `b` have potential
/// non-zero pair energy.
pub struct ContactGraph {
    adj: Vec<Vec<u32>>,
    edges: Vec<(u32, u32)>,
}

impl ContactGraph {
    /// Constructs a contact graph from raw pairwise contacts.
    ///
    /// Self-loops are silently dropped. Each pair is canonicalised so the
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

        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n_slots];
        for &(a, b) in &edges {
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        }

        Self { adj, edges }
    }

    /// Number of packable slots.
    pub fn n_slots(&self) -> usize {
        self.adj.len()
    }

    /// Number of contact edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Returns slot indices that contact slot `s`.
    pub fn neighbors(&self, s: usize) -> &[u32] {
        debug_assert!(
            s < self.adj.len(),
            "slot {s} out of bounds (n_slots={})",
            self.adj.len(),
        );
        &self.adj[s]
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

    fn neighbors_sorted(g: &ContactGraph, s: usize) -> Vec<u32> {
        let mut v = g.neighbors(s).to_vec();
        v.sort_unstable();
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
        assert!(g.neighbors(0).contains(&3));
        assert!(g.neighbors(3).contains(&0));
        assert!(g.neighbors(1).contains(&2));
        assert!(g.neighbors(2).contains(&1));
    }

    #[test]
    fn isolated_slot_has_empty_neighbors() {
        let g = ContactGraph::build(5, [(0, 1)]);
        assert_eq!(g.neighbors(4), &[]);
    }

    #[test]
    fn triangle_has_three_edges() {
        assert_eq!(triangle().n_edges(), 3);
    }

    #[test]
    fn triangle_each_slot_has_two_neighbors() {
        let g = triangle();
        assert_eq!(neighbors_sorted(&g, 0), vec![1u32, 2]);
        assert_eq!(neighbors_sorted(&g, 1), vec![0u32, 2]);
        assert_eq!(neighbors_sorted(&g, 2), vec![0u32, 1]);
    }

    #[test]
    #[should_panic]
    fn build_panics_on_n_slots_overflow() {
        ContactGraph::build(u32::MAX as usize + 2, []);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn neighbors_panics_out_of_bounds() {
        let g = ContactGraph::build(3, []);
        let _ = g.neighbors(3);
    }
}
