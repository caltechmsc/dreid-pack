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
        self.get(s, r).is_infinite()
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
                .filter(|&r| !self.data[base + r].is_infinite())
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
