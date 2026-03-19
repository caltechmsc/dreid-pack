use crate::model::types::Vec3;

/// Candidate sidechain conformations for a single packable residue.
pub struct Conformations {
    data: Vec<Vec3>,
    n_candidates: u16,
    n_atoms: u8,
}

impl Conformations {
    /// Creates a conformation set from pre-built coordinate data.
    pub fn new(data: Vec<Vec3>, n_candidates: u16, n_atoms: u8) -> Self {
        debug_assert_eq!(
            data.len(),
            n_candidates as usize * n_atoms as usize,
            "data.len() must equal n_candidates * n_atoms"
        );
        Self {
            data,
            n_candidates,
            n_atoms,
        }
    }

    /// Number of candidate conformations.
    pub fn n_candidates(&self) -> usize {
        self.n_candidates as usize
    }

    /// Number of sidechain atoms per candidate.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms as usize
    }

    /// Atom coordinates for candidate `c`.
    pub fn coords_of(&self, c: usize) -> &[Vec3] {
        debug_assert!(
            c < self.n_candidates as usize,
            "candidate index {c} out of bounds (n_candidates = {})",
            self.n_candidates
        );
        let n = self.n_atoms as usize;
        &self.data[c * n..(c + 1) * n]
    }

    /// Retains only the candidates at the given original indices,
    /// compacting in place. The old backing storage is freed.
    pub fn compact(&mut self, alive: &[u16]) {
        debug_assert!(
            alive.len() <= u16::MAX as usize,
            "alive count {} exceeds u16 capacity",
            alive.len(),
        );
        let n = self.n_atoms as usize;
        let mut buf = Vec::with_capacity(alive.len() * n);
        for &orig in alive {
            debug_assert!(
                (orig as usize) < self.n_candidates as usize,
                "alive index {orig} out of bounds (n_candidates = {})",
                self.n_candidates
            );
            let start = orig as usize * n;
            buf.extend_from_slice(&self.data[start..start + n]);
        }
        self.data = buf;
        self.n_candidates = alive.len() as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn three_by_two() -> Conformations {
        Conformations::new(
            vec![
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(1.0, 1.0, 0.0),
                v(0.0, 2.0, 0.0),
                v(1.0, 2.0, 0.0),
            ],
            3,
            2,
        )
    }

    #[test]
    fn new_stores_counts() {
        let c = three_by_two();
        assert_eq!(c.n_candidates(), 3);
        assert_eq!(c.n_atoms(), 2);
    }

    #[test]
    fn new_with_zero_candidates() {
        let c = Conformations::new(vec![], 0, 5);
        assert_eq!(c.n_candidates(), 0);
        assert_eq!(c.n_atoms(), 5);
    }

    #[test]
    fn coords_of_returns_correct_atoms() {
        let c = three_by_two();
        assert_eq!(c.coords_of(0), [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)]);
        assert_eq!(c.coords_of(1), [v(0.0, 1.0, 0.0), v(1.0, 1.0, 0.0)]);
        assert_eq!(c.coords_of(2), [v(0.0, 2.0, 0.0), v(1.0, 2.0, 0.0)]);
    }

    #[test]
    fn coords_of_single_candidate() {
        let c = Conformations::new(vec![v(5.0, 6.0, 7.0)], 1, 1);
        assert_eq!(c.coords_of(0), [v(5.0, 6.0, 7.0)]);
    }

    #[test]
    fn compact_retains_subset() {
        let mut c = three_by_two();
        c.compact(&[0, 2]);
        assert_eq!(c.n_candidates(), 2);
        assert_eq!(c.coords_of(0), [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)]);
        assert_eq!(c.coords_of(1), [v(0.0, 2.0, 0.0), v(1.0, 2.0, 0.0)]);
    }

    #[test]
    fn compact_respects_alive_order() {
        let mut c = three_by_two();
        c.compact(&[2, 0]);
        assert_eq!(c.coords_of(0), [v(0.0, 2.0, 0.0), v(1.0, 2.0, 0.0)]);
        assert_eq!(c.coords_of(1), [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)]);
    }

    #[test]
    fn compact_to_empty() {
        let mut c = three_by_two();
        c.compact(&[]);
        assert_eq!(c.n_candidates(), 0);
    }

    #[test]
    fn compact_all_alive() {
        let mut c = three_by_two();
        c.compact(&[0, 1, 2]);
        assert_eq!(c.n_candidates(), 3);
        assert_eq!(c.coords_of(0), [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)]);
        assert_eq!(c.coords_of(1), [v(0.0, 1.0, 0.0), v(1.0, 1.0, 0.0)]);
        assert_eq!(c.coords_of(2), [v(0.0, 2.0, 0.0), v(1.0, 2.0, 0.0)]);
    }

    #[test]
    fn compact_updates_n_candidates() {
        let mut c = three_by_two();
        c.compact(&[1]);
        assert_eq!(c.n_candidates(), 1);
        assert_eq!(c.n_atoms(), 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn new_panics_on_length_mismatch() {
        Conformations::new(vec![v(0.0, 0.0, 0.0)], 1, 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn coords_of_panics_out_of_bounds() {
        let c = three_by_two();
        let _ = c.coords_of(3);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn compact_panics_on_invalid_index() {
        let mut c = three_by_two();
        c.compact(&[0, 3]);
    }
}
