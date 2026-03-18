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
