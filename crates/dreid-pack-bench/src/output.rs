use std::ops::Index;
use std::time::Duration;

use arrayvec::ArrayVec;

use crate::residue::AminoAcid;

#[derive(Debug, Clone)]
pub struct BenchOutput {
    pub table: ResidueTable,
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct ResidueTable {
    slots: [Vec<Residue>; AminoAcid::COUNT],
}

impl ResidueTable {
    pub fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| Vec::new()),
        }
    }

    pub fn push(&mut self, aa: AminoAcid, residue: Residue) {
        self.slots[aa as u8 as usize].push(residue);
    }

    pub fn iter(&self) -> impl Iterator<Item = (AminoAcid, &[Residue])> {
        AminoAcid::ALL
            .iter()
            .copied()
            .zip(self.slots.iter().map(Vec::as_slice))
    }
}

impl Default for ResidueTable {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<AminoAcid> for ResidueTable {
    type Output = [Residue];

    fn index(&self, aa: AminoAcid) -> &Self::Output {
        &self.slots[aa as u8 as usize]
    }
}

#[derive(Debug, Clone)]
pub struct Residue {
    pub chi_diff: ArrayVec<Option<f64>, 4>,
    pub sc_rmsd: Option<f64>,
}
