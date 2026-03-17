use crate::model::{
    residue::ResidueType,
    types::{TypeIdx, Vec3},
};
use arrayvec::ArrayVec;
use std::collections::{HashMap, HashSet};

pub const MAX_SIDECHAIN_ATOMS: usize = 18;

/// The molecular system with mobile residues to pack and fixed atoms.
#[derive(Debug, Clone)]
pub struct System {
    /// Mobile residues to pack.
    pub mobile: Vec<Residue>,
    /// Fixed atoms: backbone, Gly/Ala sidechains, ligands, solvent.
    pub fixed: FixedAtomPool,
    /// Force-field parameters.
    pub ff: ForceFieldParams,
}

/// A packable residue slot.
#[derive(Debug, Clone)]
pub struct Residue {
    /// Amino acid residue type.
    res_type: ResidueType,
    /// NERF anchor atoms [N, Cα, C].
    anchor: [Vec3; 3],
    /// Backbone φ (rad).
    phi: f32,
    /// Backbone ψ (rad).
    psi: f32,
    /// Per-atom sidechain atom coordinates.
    sidechain: ArrayVec<Vec3, MAX_SIDECHAIN_ATOMS>,
    /// Per-atom DREIDING type.
    atom_types: ArrayVec<TypeIdx, MAX_SIDECHAIN_ATOMS>,
    /// Per-atom partial charges (e).
    atom_charges: ArrayVec<f32, MAX_SIDECHAIN_ATOMS>,
    /// H -> local donor index (u8::MAX = not an H, or no donor recorded).
    donor_of_h: ArrayVec<u8, MAX_SIDECHAIN_ATOMS>,
}

impl Residue {
    /// Returns `None` if `res_type` is not packable (Gly, Ala).
    pub fn new(
        res_type: ResidueType,
        anchor: [Vec3; 3],
        phi: f32,
        psi: f32,
        sidechain: &[Vec3],
        atom_types: &[TypeIdx],
        atom_charges: &[f32],
        donor_of_h: &[u8],
    ) -> Option<Self> {
        if !res_type.is_packable() {
            return None;
        }
        Some(Self {
            res_type,
            anchor,
            phi,
            psi,
            sidechain: sidechain.iter().copied().collect(),
            atom_types: atom_types.iter().copied().collect(),
            atom_charges: atom_charges.iter().copied().collect(),
            donor_of_h: donor_of_h.iter().copied().collect(),
        })
    }

    #[inline]
    pub fn res_type(&self) -> ResidueType {
        self.res_type
    }
    #[inline]
    pub fn anchor(&self) -> &[Vec3; 3] {
        &self.anchor
    }
    #[inline]
    pub fn phi(&self) -> f32 {
        self.phi
    }
    #[inline]
    pub fn psi(&self) -> f32 {
        self.psi
    }
    #[inline]
    pub fn sidechain(&self) -> &[Vec3] {
        &self.sidechain
    }
    #[inline]
    pub fn atom_types(&self) -> &[TypeIdx] {
        &self.atom_types
    }
    #[inline]
    pub fn atom_charges(&self) -> &[f32] {
        &self.atom_charges
    }
    #[inline]
    pub fn donor_of_h(&self) -> &[u8] {
        &self.donor_of_h
    }

    /// Write winner sidechain coordinates (internal use only).
    #[inline]
    pub(crate) fn set_sidechain(&mut self, coords: &[Vec3]) {
        self.sidechain.clear();
        self.sidechain
            .try_extend_from_slice(coords)
            .expect("coords exceed MAX_SIDECHAIN_ATOMS");
    }
}

/// Fixed atoms: backbone, Gly/Ala sidechains, ligands, solvent.
#[derive(Debug, Clone)]
pub struct FixedAtomPool {
    /// Per-atom coordinates.
    pub positions: Vec<Vec3>,
    /// Per-atom DREIDING type.
    pub types: Vec<TypeIdx>,
    /// Per-atom partial charges (e).
    pub charges: Vec<f32>,
    /// H -> local donor index (u32::MAX = not an H, or no donor recorded).
    pub donor_for_h: Vec<u32>,
}

/// Force-field parameters for VdW and H-bond energy calculations.
#[derive(Debug, Clone)]
pub struct ForceFieldParams {
    pub vdw: VdwMatrix,
    pub hbond: HBondParams,
}

/// Flat symmetric VdW matrix.
#[derive(Debug, Clone)]
pub struct VdwMatrix {
    /// Number of atom types (matrix dimension).
    n: usize,
    /// Flat symmetric matrix of (D0, R0^2) parameters.
    data: Vec<(f32, f32)>,
}

impl VdwMatrix {
    /// Creates a new `VdwMatrix` with the given dimension and data.
    pub fn new(n: usize, data: Vec<(f32, f32)>) -> Self {
        debug_assert_eq!(data.len(), n * n);
        Self { n, data }
    }

    /// Returns the (D0, R0^2) parameters for the given pair of atom types.
    #[inline(always)]
    pub fn get(&self, i: TypeIdx, j: TypeIdx) -> (f32, f32) {
        self.data[usize::from(i) * self.n + usize::from(j)]
    }
}

/// H-bond parameters.
#[derive(Debug, Clone)]
pub struct HBondParams {
    /// H-bond donor hydrogen atom types.
    h_types: HashSet<TypeIdx>,
    /// H-bond acceptor atom types.
    acc_types: HashSet<TypeIdx>,
    /// Map from (donor D, H, acceptor A) type to (D_hb, R_hb^2) parameters.
    params: HashMap<(TypeIdx, TypeIdx, TypeIdx), (f32, f32)>,
}

impl HBondParams {
    /// Creates a new `HBondParams` with the given types and parameters.
    pub fn new(
        h_types: HashSet<TypeIdx>,
        acc_types: HashSet<TypeIdx>,
        params: HashMap<(TypeIdx, TypeIdx, TypeIdx), (f32, f32)>,
    ) -> Self {
        Self {
            h_types,
            acc_types,
            params,
        }
    }

    /// Returns `true` if `(ta, tb)` could be an H···A pair.
    #[inline]
    pub fn is_hbond_candidate(&self, ta: TypeIdx, tb: TypeIdx) -> bool {
        (self.h_types.contains(&ta) && self.acc_types.contains(&tb))
            || (self.h_types.contains(&tb) && self.acc_types.contains(&ta))
    }

    /// Returns the (D_hb, R_hb^2) parameters for the given D-H···A types, or `None` if not found.
    #[inline]
    pub fn get(
        &self,
        donor_type: TypeIdx,
        h_type: TypeIdx,
        acc_type: TypeIdx,
    ) -> Option<(f32, f32)> {
        self.params.get(&(donor_type, h_type, acc_type)).copied()
    }
}
