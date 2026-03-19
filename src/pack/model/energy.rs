/// Self-energy for every (slot, candidate) pair.
pub struct SelfEnergyTable {
    data: Vec<f32>,
    offsets: Vec<usize>,
}

impl SelfEnergyTable {
    /// Creates a zero-filled table with the given per-slot candidate counts.
    pub fn new(counts: &[u16]) -> Self {
        let n = counts.len();
        let mut offsets = vec![0usize; n + 1];
        for (i, &c) in counts.iter().enumerate() {
            offsets[i + 1] = offsets[i] + c as usize;
        }
        Self {
            data: vec![0.0; offsets[n]],
            offsets,
        }
    }

    /// Number of slots.
    pub fn n_slots(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Number of candidates for slot `s`.
    pub fn n_candidates(&self, s: usize) -> usize {
        debug_assert!(
            s < self.n_slots(),
            "slot {s} out of bounds (n_slots={})",
            self.n_slots(),
        );
        self.offsets[s + 1] - self.offsets[s]
    }

    /// Self-energy of candidate `r` in slot `s`.
    pub fn get(&self, s: usize, r: usize) -> f32 {
        debug_assert!(
            s < self.n_slots(),
            "slot {s} out of bounds (n_slots={})",
            self.n_slots(),
        );
        debug_assert!(
            r < self.n_candidates(s),
            "candidate {r} out of bounds (n_candidates={})",
            self.n_candidates(s),
        );
        self.data[self.offsets[s] + r]
    }

    /// Sets the self-energy of candidate `r` in slot `s`.
    pub fn set(&mut self, s: usize, r: usize, val: f32) {
        debug_assert!(
            s < self.n_slots(),
            "slot {s} out of bounds (n_slots={})",
            self.n_slots(),
        );
        debug_assert!(
            r < self.n_candidates(s),
            "candidate {r} out of bounds (n_candidates={})",
            self.n_candidates(s),
        );
        self.data[self.offsets[s] + r] = val;
    }

    /// Marks candidate `r` in slot `s` as dead (energy -> `INFINITY`).
    pub fn prune(&mut self, s: usize, r: usize) {
        self.set(s, r, f32::INFINITY);
    }

    /// Returns `true` if candidate `r` in slot `s` has been pruned.
    pub fn is_pruned(&self, s: usize, r: usize) -> bool {
        self.get(s, r) == f32::INFINITY
    }

    /// Physically removes pruned candidates and rebuilds offsets.
    ///
    /// Returns the surviving original indices per slot — pass each
    /// inner slice to `Conformations::compact` to keep coordinates in sync.
    pub fn compact(&mut self) -> Vec<Vec<u16>> {
        let n = self.n_slots();
        let mut alive_all = Vec::with_capacity(n);
        let mut new_data = Vec::new();
        let mut new_offsets = vec![0usize; n + 1];

        for s in 0..n {
            let base = self.offsets[s];
            let count = self.offsets[s + 1] - base;
            let alive: Vec<u16> = (0..count)
                .filter(|&r| self.data[base + r] != f32::INFINITY)
                .map(|r| r as u16)
                .collect();
            for &orig in &alive {
                new_data.push(self.data[base + orig as usize]);
            }
            new_offsets[s + 1] = new_offsets[s] + alive.len();
            alive_all.push(alive);
        }

        self.data = new_data;
        self.offsets = new_offsets;
        alive_all
    }
}

/// Pair energy for every (edge, candidate_i, candidate_j) triple.
pub struct PairEnergyTable {
    data: Vec<f32>,
    offsets: Vec<usize>,
    sizes: Vec<(u16, u16)>,
}

impl PairEnergyTable {
    /// Creates a zero-filled table with the given per-edge sub-matrix sizes.
    pub fn new(dims: &[(u16, u16)]) -> Self {
        let n = dims.len();
        let mut offsets = vec![0usize; n + 1];
        for (i, &(ni, nj)) in dims.iter().enumerate() {
            offsets[i + 1] = offsets[i] + ni as usize * nj as usize;
        }
        Self {
            data: vec![0.0; offsets[n]],
            offsets,
            sizes: dims.to_vec(),
        }
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.sizes.len()
    }

    /// Sub-matrix dimensions `(n_i, n_j)` for `edge`.
    pub fn dims(&self, edge: usize) -> (usize, usize) {
        debug_assert!(
            edge < self.n_edges(),
            "edge {edge} out of bounds (n_edges={})",
            self.n_edges(),
        );
        (self.sizes[edge].0 as usize, self.sizes[edge].1 as usize)
    }

    /// Pair energy for candidates `ri`, `rj` on `edge`.
    pub fn get(&self, edge: usize, ri: usize, rj: usize) -> f32 {
        let (ni, nj) = self.dims(edge);
        debug_assert!(ri < ni, "ri {ri} out of bounds (n_i={ni})");
        debug_assert!(rj < nj, "rj {rj} out of bounds (n_j={nj})");
        self.data[self.offsets[edge] + ri * nj + rj]
    }

    /// Sets pair energy for candidates `ri`, `rj` on `edge`.
    pub fn set(&mut self, edge: usize, ri: usize, rj: usize, val: f32) {
        let (ni, nj) = self.dims(edge);
        debug_assert!(ri < ni, "ri {ri} out of bounds (n_i={ni})");
        debug_assert!(rj < nj, "rj {rj} out of bounds (n_j={nj})");
        self.data[self.offsets[edge] + ri * nj + rj] = val;
    }

    /// Raw sub-matrix slice for `edge` (row-major, length = Ri × Rj).
    pub fn matrix(&self, edge: usize) -> &[f32] {
        debug_assert!(
            edge < self.n_edges(),
            "edge {edge} out of bounds (n_edges={})",
            self.n_edges(),
        );
        &self.data[self.offsets[edge]..self.offsets[edge + 1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_new_empty_has_zero_slots() {
        let t = SelfEnergyTable::new(&[]);
        assert_eq!(t.n_slots(), 0);
    }

    #[test]
    fn self_n_slots_matches_counts() {
        let t = SelfEnergyTable::new(&[3, 5, 2]);
        assert_eq!(t.n_slots(), 3);
    }

    #[test]
    fn self_n_candidates_matches_each_count() {
        let t = SelfEnergyTable::new(&[4, 7, 1]);
        assert_eq!(t.n_candidates(0), 4);
        assert_eq!(t.n_candidates(1), 7);
        assert_eq!(t.n_candidates(2), 1);
    }

    #[test]
    fn self_all_entries_zero_after_new() {
        let t = SelfEnergyTable::new(&[3, 2]);
        for s in 0..t.n_slots() {
            for r in 0..t.n_candidates(s) {
                assert_eq!(t.get(s, r), 0.0);
            }
        }
    }

    #[test]
    fn self_set_then_get_round_trips() {
        let mut t = SelfEnergyTable::new(&[3, 2]);
        t.set(0, 2, 1.5);
        t.set(1, 0, -3.0);
        assert_eq!(t.get(0, 2), 1.5);
        assert_eq!(t.get(1, 0), -3.0);
    }

    #[test]
    fn self_slots_are_independent() {
        let mut t = SelfEnergyTable::new(&[2, 2]);
        t.set(0, 0, 99.0);
        assert_eq!(t.get(1, 0), 0.0);
    }

    #[test]
    fn self_is_pruned_false_initially() {
        let t = SelfEnergyTable::new(&[3]);
        assert!(!t.is_pruned(0, 0));
        assert!(!t.is_pruned(0, 2));
    }

    #[test]
    fn self_prune_sets_infinity() {
        let mut t = SelfEnergyTable::new(&[4]);
        t.prune(0, 1);
        assert_eq!(t.get(0, 1), f32::INFINITY);
    }

    #[test]
    fn self_is_pruned_true_after_prune() {
        let mut t = SelfEnergyTable::new(&[3]);
        t.prune(0, 2);
        assert!(t.is_pruned(0, 2));
        assert!(!t.is_pruned(0, 0));
    }

    #[test]
    fn self_compact_multi_slot() {
        let mut t = SelfEnergyTable::new(&[4, 3, 2]);
        t.set(0, 0, 1.0);
        t.set(0, 2, 3.0);
        t.prune(0, 1);
        t.prune(0, 3);
        t.set(1, 1, 6.0);
        t.set(1, 2, 7.0);
        t.prune(1, 0);
        t.set(2, 0, 8.0);
        t.set(2, 1, 9.0);

        let alive = t.compact();

        assert_eq!(alive, vec![vec![0, 2], vec![1, 2], vec![0, 1]]);
        assert_eq!(t.n_candidates(0), 2);
        assert_eq!(t.n_candidates(1), 2);
        assert_eq!(t.n_candidates(2), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 3.0);
        assert_eq!(t.get(1, 0), 6.0);
        assert_eq!(t.get(1, 1), 7.0);
        assert_eq!(t.get(2, 0), 8.0);
        assert_eq!(t.get(2, 1), 9.0);
    }

    #[test]
    fn self_compact_identity_when_nothing_pruned() {
        let mut t = SelfEnergyTable::new(&[3, 2]);
        t.set(0, 0, 1.0);
        t.set(0, 1, 2.0);
        t.set(0, 2, 3.0);
        t.set(1, 0, 4.0);
        t.set(1, 1, 5.0);

        let alive = t.compact();

        assert_eq!(alive, vec![vec![0, 1, 2], vec![0, 1]]);
        assert_eq!(t.n_candidates(0), 3);
        assert_eq!(t.n_candidates(1), 2);
        assert_eq!(t.get(0, 1), 2.0);
        assert_eq!(t.get(1, 0), 4.0);
    }

    #[test]
    fn self_compact_all_pruned_yields_zero_candidates() {
        let mut t = SelfEnergyTable::new(&[2, 1]);
        t.prune(0, 0);
        t.prune(0, 1);
        t.prune(1, 0);

        let alive = t.compact();

        assert_eq!(alive, vec![vec![], vec![]]);
        assert_eq!(t.n_candidates(0), 0);
        assert_eq!(t.n_candidates(1), 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn self_get_panics_slot_out_of_bounds() {
        let t = SelfEnergyTable::new(&[3]);
        let _ = t.get(1, 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn self_get_panics_candidate_out_of_bounds() {
        let t = SelfEnergyTable::new(&[3]);
        let _ = t.get(0, 3);
    }

    #[test]
    fn pair_new_empty_has_zero_edges() {
        let t = PairEnergyTable::new(&[]);
        assert_eq!(t.n_edges(), 0);
    }

    #[test]
    fn pair_n_edges_matches_dims() {
        let t = PairEnergyTable::new(&[(3, 2), (4, 5)]);
        assert_eq!(t.n_edges(), 2);
    }

    #[test]
    fn pair_dims_match_input() {
        let t = PairEnergyTable::new(&[(3, 2), (4, 5)]);
        assert_eq!(t.dims(0), (3, 2));
        assert_eq!(t.dims(1), (4, 5));
    }

    #[test]
    fn pair_all_entries_zero_after_new() {
        let t = PairEnergyTable::new(&[(2, 3)]);
        for ri in 0..2 {
            for rj in 0..3 {
                assert_eq!(t.get(0, ri, rj), 0.0);
            }
        }
    }

    #[test]
    fn pair_set_then_get_round_trips() {
        let mut t = PairEnergyTable::new(&[(3, 4)]);
        t.set(0, 2, 3, -1.5);
        assert_eq!(t.get(0, 2, 3), -1.5);
    }

    #[test]
    fn pair_edges_are_independent() {
        let mut t = PairEnergyTable::new(&[(2, 2), (2, 2)]);
        t.set(0, 1, 0, 7.0);
        assert_eq!(t.get(1, 1, 0), 0.0);
    }

    #[test]
    fn pair_matrix_length_matches_product() {
        let t = PairEnergyTable::new(&[(3, 4), (2, 5)]);
        assert_eq!(t.matrix(0).len(), 12);
        assert_eq!(t.matrix(1).len(), 10);
    }

    #[test]
    fn pair_matrix_reflects_set_calls() {
        let mut t = PairEnergyTable::new(&[(2, 3)]);
        t.set(0, 0, 0, 1.0);
        t.set(0, 0, 2, 3.0);
        t.set(0, 1, 1, 5.0);
        assert_eq!(t.matrix(0), &[1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn pair_get_panics_edge_out_of_bounds() {
        let t = PairEnergyTable::new(&[(2, 3)]);
        let _ = t.get(1, 0, 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn pair_get_panics_ri_out_of_bounds() {
        let t = PairEnergyTable::new(&[(2, 3)]);
        let _ = t.get(0, 2, 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn pair_get_panics_rj_out_of_bounds() {
        let t = PairEnergyTable::new(&[(2, 3)]);
        let _ = t.get(0, 0, 3);
    }
}
