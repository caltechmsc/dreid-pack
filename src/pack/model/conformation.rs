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
