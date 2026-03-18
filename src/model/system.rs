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

/// Per-atom sidechain data for constructing a [`Residue`].
pub struct SidechainAtoms<'a> {
    /// Per-atom sidechain atom coordinates.
    pub coords: &'a [Vec3],
    /// Per-atom DREIDING type.
    pub types: &'a [TypeIdx],
    /// Per-atom partial charges (e).
    pub charges: &'a [f32],
    /// H -> local donor index (u8::MAX = not an H, or no donor recorded).
    pub donor_of_h: &'a [u8],
}

impl Residue {
    /// Returns `None` if `res_type` is not packable (Gly, Ala).
    pub fn new(
        res_type: ResidueType,
        anchor: [Vec3; 3],
        phi: f32,
        psi: f32,
        atoms: SidechainAtoms<'_>,
    ) -> Option<Self> {
        if !res_type.is_packable() {
            return None;
        }
        let n = atoms.coords.len();
        debug_assert!(n <= MAX_SIDECHAIN_ATOMS, "too many sidechain atoms: {n}");
        debug_assert_eq!(atoms.types.len(), n, "types/coords length mismatch");
        debug_assert_eq!(atoms.charges.len(), n, "charges/coords length mismatch");
        debug_assert_eq!(
            atoms.donor_of_h.len(),
            n,
            "donor_of_h/coords length mismatch"
        );
        Some(Self {
            res_type,
            anchor,
            phi,
            psi,
            sidechain: atoms.coords.iter().copied().collect(),
            atom_types: atoms.types.iter().copied().collect(),
            atom_charges: atoms.charges.iter().copied().collect(),
            donor_of_h: atoms.donor_of_h.iter().copied().collect(),
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

    /// Overwrites the stored sidechain coordinates (internal use only).
    #[inline]
    pub(crate) fn set_sidechain(&mut self, coords: &[Vec3]) {
        debug_assert!(
            coords.len() <= MAX_SIDECHAIN_ATOMS,
            "coords.len()={} > MAX_SIDECHAIN_ATOMS={}",
            coords.len(),
            MAX_SIDECHAIN_ATOMS
        );
        self.sidechain.clear();
        // SAFETY: coords.len() ≤ MAX_SIDECHAIN_ATOMS = capacity (verified by debug_assert).
        for &c in coords {
            unsafe { self.sidechain.push_unchecked(c) };
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::residue::ResidueType;
    use crate::model::types::{TypeIdx, Vec3};

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn t(n: u8) -> TypeIdx {
        TypeIdx(n)
    }

    fn ser_residue() -> Residue {
        let anchor = [v(0.0, 0.0, 0.0), v(1.5, 0.0, 0.0), v(3.0, 0.0, 0.0)];
        let coords = [v(1.0, 1.0, 0.0); 5];
        let types = [t(1); 5];
        let charges = [0.1f32; 5];
        let donor_of_h = [u8::MAX; 5];
        Residue::new(
            ResidueType::Ser,
            anchor,
            -1.0,
            1.0,
            SidechainAtoms {
                coords: &coords,
                types: &types,
                charges: &charges,
                donor_of_h: &donor_of_h,
            },
        )
        .unwrap()
    }

    fn identity_vdw(n: usize) -> VdwMatrix {
        let mut data = vec![(0.0f32, 0.0f32); n * n];
        for i in 0..n {
            data[i * n + i] = (1.0, 4.0);
        }
        VdwMatrix::new(n, data)
    }

    fn empty_hbond() -> HBondParams {
        HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new())
    }

    #[test]
    fn residue_new_rejects_non_packable() {
        let anchor = [v(0.0, 0.0, 0.0); 3];
        let empty = SidechainAtoms {
            coords: &[],
            types: &[],
            charges: &[],
            donor_of_h: &[],
        };
        assert!(Residue::new(ResidueType::Gly, anchor, 0.0, 0.0, empty).is_none());
        let empty = SidechainAtoms {
            coords: &[],
            types: &[],
            charges: &[],
            donor_of_h: &[],
        };
        assert!(Residue::new(ResidueType::Ala, anchor, 0.0, 0.0, empty).is_none());
    }

    #[test]
    fn residue_new_accepts_packable() {
        let r = ser_residue();
        assert_eq!(r.res_type(), ResidueType::Ser);
    }

    #[test]
    fn residue_accessors_round_trip() {
        let r = ser_residue();
        assert_eq!(r.anchor()[1], v(1.5, 0.0, 0.0));
        assert_eq!(r.phi(), -1.0);
        assert_eq!(r.psi(), 1.0);
        assert_eq!(r.sidechain().len(), 5);
        assert_eq!(r.atom_types().len(), 5);
        assert_eq!(r.atom_charges().len(), 5);
        assert_eq!(r.donor_of_h().len(), 5);
    }

    #[test]
    fn residue_set_sidechain_overwrites() {
        let mut r = ser_residue();
        let new_coords = [v(9.0, 9.0, 9.0); 5];
        r.set_sidechain(&new_coords);
        assert_eq!(r.sidechain().len(), 5);
        assert!(r.sidechain().iter().all(|&c| c == v(9.0, 9.0, 9.0)));
    }

    #[test]
    fn residue_set_sidechain_clears_before_write() {
        let mut r = ser_residue();
        r.set_sidechain(&[v(1.0, 2.0, 3.0); 3]);
        assert_eq!(r.sidechain().len(), 3);
        r.set_sidechain(&[v(4.0, 5.0, 6.0); 5]);
        assert_eq!(r.sidechain().len(), 5);
        assert!(r.sidechain().iter().all(|&c| c == v(4.0, 5.0, 6.0)));
    }

    #[test]
    fn vdw_matrix_diagonal_lookup() {
        let m = identity_vdw(4);
        assert_eq!(m.get(t(0), t(0)), (1.0, 4.0));
        assert_eq!(m.get(t(1), t(1)), (1.0, 4.0));
    }

    #[test]
    fn vdw_matrix_off_diagonal_zero() {
        let m = identity_vdw(4);
        assert_eq!(m.get(t(0), t(1)), (0.0, 0.0));
        assert_eq!(m.get(t(2), t(3)), (0.0, 0.0));
    }

    #[test]
    fn vdw_matrix_symmetric_fill() {
        let n = 3usize;
        let mut data = vec![(0.0f32, 0.0f32); n * n];
        data[0 * n + 1] = (2.0, 8.0);
        data[1 * n + 0] = (2.0, 8.0);
        let m = VdwMatrix::new(n, data);
        assert_eq!(m.get(t(0), t(1)), m.get(t(1), t(0)));
    }

    #[test]
    fn hbond_candidate_both_directions() {
        let mut h_types = HashSet::new();
        let mut acc_types = HashSet::new();
        h_types.insert(t(1));
        acc_types.insert(t(2));
        let p = HBondParams::new(h_types, acc_types, HashMap::new());

        assert!(p.is_hbond_candidate(t(1), t(2)));
        assert!(p.is_hbond_candidate(t(2), t(1)));
        assert!(!p.is_hbond_candidate(t(0), t(3)));
    }

    #[test]
    fn hbond_get_returns_params() {
        let mut h_types = HashSet::new();
        let mut acc_types = HashSet::new();
        let mut params = HashMap::new();
        h_types.insert(t(1));
        acc_types.insert(t(3));
        params.insert((t(0), t(1), t(3)), (5.0f32, 25.0f32));
        let p = HBondParams::new(h_types, acc_types, params);

        assert_eq!(p.get(t(0), t(1), t(3)), Some((5.0, 25.0)));
        assert_eq!(p.get(t(0), t(1), t(0)), None);
    }

    #[test]
    fn hbond_empty_never_candidate() {
        let p = empty_hbond();
        assert!(!p.is_hbond_candidate(t(0), t(1)));
    }

    #[test]
    fn system_mobile_len() {
        let system = System {
            mobile: vec![ser_residue(), ser_residue()],
            fixed: FixedAtomPool {
                positions: vec![],
                types: vec![],
                charges: vec![],
                donor_for_h: vec![],
            },
            ff: ForceFieldParams {
                vdw: identity_vdw(4),
                hbond: empty_hbond(),
            },
        };
        assert_eq!(system.mobile.len(), 2);
        assert_eq!(system.fixed.positions.len(), 0);
    }

    #[test]
    fn system_fixed_pool_fields_consistent() {
        let n = 3;
        let fixed = FixedAtomPool {
            positions: vec![v(0.0, 0.0, 0.0); n],
            types: vec![t(0); n],
            charges: vec![0.0f32; n],
            donor_for_h: vec![u32::MAX; n],
        };
        assert_eq!(fixed.positions.len(), fixed.types.len());
        assert_eq!(fixed.types.len(), fixed.charges.len());
        assert_eq!(fixed.charges.len(), fixed.donor_for_h.len());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn residue_new_panics_when_coords_exceed_max_sidechain_atoms() {
        let anchor = [v(0.0, 0.0, 0.0); 3];
        let n = MAX_SIDECHAIN_ATOMS + 1;
        let coords = vec![v(1.0, 0.0, 0.0); n];
        let types = vec![t(1); n];
        let charges = vec![0.1f32; n];
        let donor_of_h = vec![u8::MAX; n];
        Residue::new(
            ResidueType::Ser,
            anchor,
            0.0,
            0.0,
            SidechainAtoms {
                coords: &coords,
                types: &types,
                charges: &charges,
                donor_of_h: &donor_of_h,
            },
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn residue_new_panics_on_types_length_mismatch() {
        let anchor = [v(0.0, 0.0, 0.0); 3];
        let coords = [v(1.0, 1.0, 0.0); 5];
        let types = [t(1); 3];
        let charges = [0.1f32; 5];
        let donor_of_h = [u8::MAX; 5];
        Residue::new(
            ResidueType::Ser,
            anchor,
            0.0,
            0.0,
            SidechainAtoms {
                coords: &coords,
                types: &types,
                charges: &charges,
                donor_of_h: &donor_of_h,
            },
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn residue_new_panics_on_charges_length_mismatch() {
        let anchor = [v(0.0, 0.0, 0.0); 3];
        let coords = [v(1.0, 1.0, 0.0); 5];
        let types = [t(1); 5];
        let charges = [0.1f32; 4];
        let donor_of_h = [u8::MAX; 5];
        Residue::new(
            ResidueType::Ser,
            anchor,
            0.0,
            0.0,
            SidechainAtoms {
                coords: &coords,
                types: &types,
                charges: &charges,
                donor_of_h: &donor_of_h,
            },
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn residue_new_panics_on_donor_of_h_length_mismatch() {
        let anchor = [v(0.0, 0.0, 0.0); 3];
        let coords = [v(1.0, 1.0, 0.0); 5];
        let types = [t(1); 5];
        let charges = [0.1f32; 5];
        let donor_of_h = [u8::MAX; 2];
        Residue::new(
            ResidueType::Ser,
            anchor,
            0.0,
            0.0,
            SidechainAtoms {
                coords: &coords,
                types: &types,
                charges: &charges,
                donor_of_h: &donor_of_h,
            },
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn vdw_matrix_new_panics_on_wrong_data_length() {
        VdwMatrix::new(3, vec![(1.0f32, 1.0f32); 8]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn set_sidechain_panics_on_overflow() {
        let mut r = ser_residue();
        let too_many = vec![v(1.0, 0.0, 0.0); MAX_SIDECHAIN_ATOMS + 1];
        r.set_sidechain(&too_many);
    }
}
