use crate::{
    model::{
        system::{ForceFieldParams, HBondParams, Residue, VdwMatrix},
        types::{TypeIdx, Vec3},
    },
    pack::{
        constant::{
            COULOMB_CONST, COULOMB_CUTOFF, HBOND_CUTOFF, HBOND_CUTOFF_SQ, VDW_CUTOFF, VDW_CUTOFF_SQ,
        },
        energy::{BuckKernel, LjKernel, VdwKernel, cos_dha, coulomb_energy, hbond_energy},
        model::{
            conformation::Conformations,
            energy::SelfEnergyTable,
            fixed::{FixedAtoms, NO_DONOR},
        },
    },
};
use rayon::prelude::*;

/// Computes self-energies (SC <-> Fixed), threshold-prunes dead candidates,
/// and compacts both [`SelfEnergyTable`] and [`Conformations`] in sync.
pub fn prune(
    slots: &[Residue],
    conformations: &mut [Conformations],
    fixed: &FixedAtoms<'_>,
    ff: &ForceFieldParams,
    electrostatics: Option<f32>,
    threshold: f32,
) -> SelfEnergyTable {
    let n = slots.len();
    debug_assert_eq!(n, conformations.len());

    let c_d = electrostatics.map(|d| COULOMB_CONST / d);

    let results = match (&ff.vdw, c_d) {
        (VdwMatrix::LennardJones(m), None) => survivors::<_, false>(
            &LjKernel(m),
            slots,
            conformations,
            fixed,
            &ff.hbond,
            0.0,
            threshold,
        ),
        (VdwMatrix::LennardJones(m), Some(c)) => survivors::<_, true>(
            &LjKernel(m),
            slots,
            conformations,
            fixed,
            &ff.hbond,
            c,
            threshold,
        ),
        (VdwMatrix::Buckingham(m), None) => survivors::<_, false>(
            &BuckKernel(m),
            slots,
            conformations,
            fixed,
            &ff.hbond,
            0.0,
            threshold,
        ),
        (VdwMatrix::Buckingham(m), Some(c)) => survivors::<_, true>(
            &BuckKernel(m),
            slots,
            conformations,
            fixed,
            &ff.hbond,
            c,
            threshold,
        ),
    };

    let counts: Vec<u16> = results
        .iter()
        .map(|(alive, _)| alive.len() as u16)
        .collect();
    let mut table = SelfEnergyTable::new(&counts);
    for (s, (alive, energies)) in results.iter().enumerate() {
        for (r, &e) in energies.iter().enumerate() {
            table.set(s, r, e);
        }
        conformations[s].compact(alive);
    }
    debug_assert!((0..n).all(|s| table.n_candidates(s) == conformations[s].n_candidates()));
    table
}

/// Per-slot atom metadata shared across all candidates of the same residue.
struct SlotAtoms<'a> {
    n_a: usize,
    types: &'a [TypeIdx],
    charges: &'a [f32],
    donors: &'a [u8],
}

/// Parallel over slots × candidates: compute energies, threshold-prune,
/// return per-slot `(surviving_indices, surviving_energies)`.
fn survivors<V: VdwKernel + Sync, const COUL: bool>(
    vdw: &V,
    slots: &[Residue],
    conformations: &[Conformations],
    fixed: &FixedAtoms<'_>,
    hbond: &HBondParams,
    c_d: f32,
    threshold: f32,
) -> Vec<(Vec<u16>, Vec<f32>)> {
    slots
        .par_iter()
        .zip(conformations.par_iter())
        .map(|(slot, confs)| {
            let atoms = SlotAtoms {
                n_a: confs.n_atoms(),
                types: slot.atom_types(),
                charges: slot.atom_charges(),
                donors: slot.donor_of_h(),
            };

            let energies: Vec<f32> = (0..confs.n_candidates())
                .into_par_iter()
                .map(|r| self_energy::<V, COUL>(confs.coords_of(r), &atoms, fixed, vdw, hbond, c_d))
                .collect();

            let e_min = energies.iter().copied().fold(f32::INFINITY, f32::min);
            let cutoff = e_min + threshold;

            energies
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, e)| e <= cutoff)
                .map(|(i, e)| (i as u16, e))
                .unzip()
        })
        .collect()
}

/// Non-bonded self-energy of one candidate against all fixed atoms.
fn self_energy<V: VdwKernel, const COUL: bool>(
    coords: &[Vec3],
    atoms: &SlotAtoms<'_>,
    fixed: &FixedAtoms<'_>,
    vdw: &V,
    hbond: &HBondParams,
    c_d: f32,
) -> f32 {
    let query_r = if COUL { COULOMB_CUTOFF } else { VDW_CUTOFF };
    let mut e = 0.0_f32;

    for (pos_a, (&ta, &qa)) in coords
        .iter()
        .copied()
        .zip(atoms.types.iter().zip(atoms.charges))
    {
        for (pos_b, b) in fixed.neighbors(pos_a, query_r) {
            let b = b as usize;
            let tb = fixed.types[b];
            let r_sq = pos_a.dist_sq(pos_b);

            if r_sq <= VDW_CUTOFF_SQ {
                e += vdw.energy(ta, tb, r_sq);
            }

            // Fixed-donor H-bond: b is polar H (fixed side), a is acceptor.
            // Grid found H within query_r of A; D is looked up via table (not grid).
            // Safety: cos_dha > 0 ⟹ |DA|² = |DH|² + |HA|² + 2|DH||HA|cosθ > |HA|²
            // ⟹ |HA| < |DA| ≤ HBOND_CUTOFF ≤ query_r  — H always found.
            // cos_dha ≤ 0 ⟹ effective_cos = max(0,cos) = 0 ⟹ energy = 0 exactly.
            if fixed.donor_for_h[b] != NO_DONOR {
                let d = fixed.donor_for_h[b] as usize;
                let pos_d = fixed.positions[d];
                let r_sq_da = pos_d.dist_sq(pos_a);
                if r_sq_da <= HBOND_CUTOFF_SQ {
                    let cos = cos_dha(pos_d, pos_b, pos_a);
                    e += hbond_energy(r_sq_da, cos, fixed.types[d], tb, ta, hbond);
                }
            }

            e += coulomb_energy::<COUL>(c_d, qa, fixed.charges[b], r_sq);
        }
    }

    for h in 0..atoms.n_a {
        let d_local = atoms.donors[h];
        if d_local == u8::MAX {
            continue;
        }
        let d = d_local as usize;
        let pos_d = coords[d];
        let pos_h = coords[h];
        let td = atoms.types[d];
        let th = atoms.types[h];

        for (pos_acc, f) in fixed.neighbors(pos_d, HBOND_CUTOFF) {
            let r_sq_da = pos_d.dist_sq(pos_acc);
            let cos = cos_dha(pos_d, pos_h, pos_acc);
            e += hbond_energy(r_sq_da, cos, td, th, fixed.types[f as usize], hbond);
        }
    }

    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        residue::ResidueType,
        system::{FixedAtomPool, LjMatrix, LjPair, SidechainAtoms},
    };
    use approx::assert_abs_diff_eq;
    use std::collections::{HashMap, HashSet};

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn t(n: u8) -> TypeIdx {
        TypeIdx(n)
    }

    fn lj_n(n: usize, d0: f32, r0: f32) -> LjMatrix {
        let p = LjPair { d0, r0_sq: r0 * r0 };
        LjMatrix::new(n, vec![p; n * n])
    }

    fn no_hbond() -> HBondParams {
        HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new())
    }

    fn hbond_dha(d_hb: f32, r_hb: f32) -> HBondParams {
        let mut h_types = HashSet::new();
        h_types.insert(t(1));
        let mut acc_types = HashSet::new();
        acc_types.insert(t(2));
        let mut params = HashMap::new();
        params.insert((t(0), t(1), t(2)), (d_hb, r_hb * r_hb));
        HBondParams::new(h_types, acc_types, params)
    }

    fn empty_pool() -> FixedAtomPool {
        FixedAtomPool {
            positions: vec![],
            types: vec![],
            charges: vec![],
            donor_for_h: vec![],
        }
    }

    fn single_atom_pool(pos: Vec3, typ: TypeIdx, charge: f32) -> FixedAtomPool {
        FixedAtomPool {
            positions: vec![pos],
            types: vec![typ],
            charges: vec![charge],
            donor_for_h: vec![NO_DONOR],
        }
    }

    fn donor_h_pool(d_pos: Vec3, h_pos: Vec3, td: TypeIdx, th: TypeIdx) -> FixedAtomPool {
        FixedAtomPool {
            positions: vec![d_pos, h_pos],
            types: vec![td, th],
            charges: vec![0.0, 0.0],
            donor_for_h: vec![NO_DONOR, 0],
        }
    }

    fn make_slot(types: &[TypeIdx], charges: &[f32], donors: &[u8]) -> Residue {
        let n = types.len();
        let coords = vec![Vec3::zero(); n];
        Residue::new(
            ResidueType::Ser,
            [Vec3::zero(); 3],
            0.0,
            0.0,
            std::f32::consts::PI,
            SidechainAtoms {
                coords: &coords,
                types,
                charges,
                donor_of_h: donors,
            },
        )
        .unwrap()
    }

    fn confs_from(n_atoms: u8, candidates: &[&[Vec3]]) -> Conformations {
        let data: Vec<Vec3> = candidates.iter().flat_map(|c| c.iter().copied()).collect();
        Conformations::new(data, candidates.len() as u16, n_atoms)
    }

    #[test]
    fn no_fixed_atoms_returns_zero_energy() {
        let pool = empty_pool();
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[1.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, 2.0, 3.0)),
            &no_hbond(),
            0.0,
        );
        assert_eq!(e, 0.0);
    }

    #[test]
    fn vdw_at_equilibrium_equals_minus_d0() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, d0, r0)),
            &no_hbond(),
            0.0,
        );
        assert_abs_diff_eq!(e, -d0, epsilon = 1e-5);
    }

    #[test]
    fn vdw_repulsive_at_close_range() {
        let r_close = 1.5_f32;
        let pool = single_atom_pool(v(r_close, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, 1.0, 3.0)),
            &no_hbond(),
            0.0,
        );
        assert!(e > 0.0, "expected repulsion at r={r_close}, got {e}");
    }

    #[test]
    fn vdw_beyond_cutoff_contributes_nothing() {
        let pool = single_atom_pool(v(VDW_CUTOFF + 0.01, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF + 1.0);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, 2.0, 3.0)),
            &no_hbond(),
            0.0,
        );
        assert_eq!(e, 0.0);
    }

    #[test]
    fn vdw_guard_excludes_atom_in_coulomb_only_range() {
        let r = (VDW_CUTOFF + COULOMB_CUTOFF) / 2.0;
        let pool = single_atom_pool(v(r, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, COULOMB_CUTOFF + 1.0);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let c_d = COULOMB_CONST / 4.0;
        let e = self_energy::<_, true>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, 5.0, 3.0)),
            &no_hbond(),
            c_d,
        );
        assert_eq!(e, 0.0, "VdW guard must silence atom at r={r}, got {e}");
    }

    #[test]
    fn coulomb_opposite_charges_attractive() {
        let pool = single_atom_pool(v(2.0, 0.0, 0.0), t(0), -1.0);
        let fixed = FixedAtoms::build(&pool, COULOMB_CUTOFF);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[1.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, true>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, 0.0, 5.0)),
            &no_hbond(),
            COULOMB_CONST / 4.0,
        );
        assert!(e < 0.0, "opposite charges must attract, got {e}");
    }

    #[test]
    fn coulomb_same_sign_charges_repulsive() {
        let pool = single_atom_pool(v(2.0, 0.0, 0.0), t(0), 1.0);
        let fixed = FixedAtoms::build(&pool, COULOMB_CUTOFF);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[1.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, true>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, 0.0, 5.0)),
            &no_hbond(),
            COULOMB_CONST / 4.0,
        );
        assert!(e > 0.0, "same-sign charges must repel, got {e}");
    }

    #[test]
    fn coulomb_disabled_contributes_nothing() {
        let pool = single_atom_pool(v(2.0, 0.0, 0.0), t(0), 100.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[100.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, 0.0, 5.0)),
            &no_hbond(),
            0.0,
        );
        assert_eq!(e, 0.0);
    }

    #[test]
    fn coulomb_exact_energy_matches_formula() {
        let (r, qi, qj, diel) = (3.0_f32, 0.5_f32, -0.5_f32, 4.0_f32);
        let pool = single_atom_pool(v(r, 0.0, 0.0), t(0), qj);
        let fixed = FixedAtoms::build(&pool, COULOMB_CUTOFF);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(0)],
            charges: &[qi],
            donors: &[u8::MAX],
        };
        let c_d = COULOMB_CONST / diel;
        let expected = c_d * qi * qj / (r * r);
        let e = self_energy::<_, true>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj_n(1, 0.0, 5.0)),
            &no_hbond(),
            c_d,
        );
        assert_abs_diff_eq!(e, expected, epsilon = 1e-5);
    }

    #[test]
    fn fixed_donor_linear_geometry_is_attractive() {
        let pool = donor_h_pool(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), t(0), t(1));
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let coords = [v(3.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(&coords, &atoms, &fixed, &LjKernel(&lj), &hbond, 0.0);
        assert!(
            e < 0.0,
            "linear fixed-donor H-bond must be attractive, got {e}"
        );
    }

    #[test]
    fn fixed_donor_hbond_at_equilibrium_equals_minus_d_hb() {
        let (d_hb, r_hb) = (3.0_f32, 2.0_f32);
        let pool = donor_h_pool(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), t(0), t(1));
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(d_hb, r_hb);
        let coords = [v(r_hb, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(&coords, &atoms, &fixed, &LjKernel(&lj), &hbond, 0.0);
        assert_abs_diff_eq!(e, -d_hb, epsilon = 1e-5);
    }

    #[test]
    fn fixed_donor_obtuse_geometry_contributes_zero() {
        let pool = donor_h_pool(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), t(0), t(1));
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let coords = [v(0.5, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(&coords, &atoms, &fixed, &LjKernel(&lj), &hbond, 0.0);
        assert_eq!(
            e, 0.0,
            "obtuse geometry must yield zero H-bond energy, got {e}"
        );
    }

    #[test]
    fn fixed_donor_da_beyond_hbond_cutoff_contributes_zero() {
        let d_a = HBOND_CUTOFF + 0.1;
        let pool = donor_h_pool(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), t(0), t(1));
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF + 2.0);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let coords = [v(d_a, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(&coords, &atoms, &fixed, &LjKernel(&lj), &hbond, 0.0);
        assert_eq!(
            e, 0.0,
            "D-A beyond HBOND_CUTOFF must contribute zero, got {e}"
        );
    }

    #[test]
    fn sc_donor_linear_geometry_is_attractive() {
        let pool = single_atom_pool(v(3.0, 0.0, 0.0), t(2), 0.0);
        let fixed = FixedAtoms::build(&pool, HBOND_CUTOFF + 1.0);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let coords = [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 2,
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let e = self_energy::<_, false>(&coords, &atoms, &fixed, &LjKernel(&lj), &hbond, 0.0);
        assert!(
            e < 0.0,
            "SC-donor linear geometry must be attractive, got {e}"
        );
    }

    #[test]
    fn sc_donor_hbond_at_equilibrium_equals_minus_d_hb() {
        let (d_hb, r_hb) = (3.0_f32, 2.0_f32);
        let pool = single_atom_pool(v(r_hb, 0.0, 0.0), t(2), 0.0);
        let fixed = FixedAtoms::build(&pool, HBOND_CUTOFF + 1.0);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(d_hb, r_hb);
        let coords = [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 2,
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let e = self_energy::<_, false>(&coords, &atoms, &fixed, &LjKernel(&lj), &hbond, 0.0);
        assert_abs_diff_eq!(e, -d_hb, epsilon = 1e-5);
    }

    #[test]
    fn sc_donor_no_donor_flag_skips_hbond() {
        let pool = single_atom_pool(v(2.0, 0.0, 0.0), t(2), 0.0);
        let fixed = FixedAtoms::build(&pool, HBOND_CUTOFF + 1.0);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let coords = [v(0.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &[t(1)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = self_energy::<_, false>(&coords, &atoms, &fixed, &LjKernel(&lj), &hbond, 0.0);
        assert_eq!(
            e, 0.0,
            "no donor flag: SC-donor loop must be skipped, got {e}"
        );
    }

    #[test]
    fn sc_donor_acceptor_beyond_hbond_cutoff_contributes_zero() {
        let pool = single_atom_pool(v(HBOND_CUTOFF + 0.5, 0.0, 0.0), t(2), 0.0);
        let fixed = FixedAtoms::build(&pool, HBOND_CUTOFF + 2.0);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let coords = [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
        let atoms = SlotAtoms {
            n_a: 2,
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let e = self_energy::<_, false>(&coords, &atoms, &fixed, &LjKernel(&lj), &hbond, 0.0);
        assert_eq!(
            e, 0.0,
            "acceptor beyond HBOND_CUTOFF must yield zero, got {e}"
        );
    }

    #[test]
    fn prune_large_threshold_retains_all_candidates() {
        let pool = single_atom_pool(v(100.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, 0.0, 3.0)),
            hbond: no_hbond(),
        };
        let types = [t(0)];
        let slot = make_slot(&types, &[0.0], &[u8::MAX]);
        let c: Vec<Vec3> = (0..3).map(|i| v(i as f32, 0.0, 0.0)).collect();
        let mut confs = vec![confs_from(1, &[&c[0..1], &c[1..2], &c[2..3]])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert_eq!(table.n_candidates(0), 3);
        assert_eq!(confs[0].n_candidates(), 3);
    }

    #[test]
    fn prune_zero_threshold_keeps_only_minimum_energy_candidate() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let types = [t(0)];
        let slot = make_slot(&types, &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far = [v(50.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&near, &far])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.0);
        assert_eq!(
            table.n_candidates(0),
            1,
            "only minimum-energy candidate must survive"
        );
        assert_eq!(confs[0].n_candidates(), 1);
        assert!(
            table.get(0, 0) < 0.0,
            "surviving energy must be the low-energy one"
        );
    }

    #[test]
    fn prune_threshold_window_admits_candidates_within_range() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let types = [t(0)];
        let near = [v(0.0, 0.0, 0.0)];
        let far = [v(50.0, 0.0, 0.0)];

        let mut confs_a = vec![confs_from(1, &[&near, &far])];
        let table_a = prune(
            &[make_slot(&types, &[0.0], &[u8::MAX])],
            &mut confs_a,
            &fixed,
            &ff,
            None,
            1.0,
        );
        assert_eq!(
            table_a.n_candidates(0),
            1,
            "threshold=1: only minimum must survive"
        );

        let mut confs_b = vec![confs_from(1, &[&near, &far])];
        let table_b = prune(
            &[make_slot(&types, &[0.0], &[u8::MAX])],
            &mut confs_b,
            &fixed,
            &ff,
            None,
            3.0,
        );
        assert_eq!(
            table_b.n_candidates(0),
            2,
            "threshold=3: both candidates must survive"
        );
    }

    #[test]
    fn prune_table_and_conformations_count_synchronized() {
        let pool = single_atom_pool(v(3.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, 2.0, 3.0)),
            hbond: no_hbond(),
        };
        let types = [t(0)];
        let slot = make_slot(&types, &[0.0], &[u8::MAX]);
        let c0 = [v(0.0, 0.0, 0.0)];
        let c1 = [v(1.0, 0.0, 0.0)];
        let c2 = [v(50.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&c0, &c1, &c2])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.0);
        assert_eq!(table.n_candidates(0), confs[0].n_candidates());
    }

    #[test]
    fn prune_multi_slot_counts_are_independent() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let types = [t(0)];
        let slot = make_slot(&types, &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far_a = [v(50.0, 0.0, 0.0)];
        let far_b = [v(100.0, 0.0, 0.0)];
        let far_c = [v(200.0, 0.0, 0.0)];
        let mut confs = vec![
            confs_from(1, &[&near, &far_a]),
            confs_from(1, &[&far_b, &far_c]),
        ];
        let table = prune(&[slot.clone(), slot], &mut confs, &fixed, &ff, None, 0.0);
        assert_eq!(
            table.n_candidates(0),
            1,
            "slot 0: only minimum must survive"
        );
        assert_eq!(
            table.n_candidates(1),
            2,
            "slot 1: both zero-energy must survive"
        );
    }

    #[test]
    fn prune_surviving_energies_stored_in_table() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let types = [t(0)];
        let slot = make_slot(&types, &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far = [v(50.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&near, &far])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert_abs_diff_eq!(table.get(0, 0), -d0, epsilon = 1e-5);
        assert_abs_diff_eq!(table.get(0, 1), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn prune_with_coulomb_enabled_affects_energy() {
        let (r, qi, qj, diel) = (3.0_f32, 1.0_f32, -1.0_f32, 4.0_f32);
        let pool = single_atom_pool(v(r, 0.0, 0.0), t(0), qj);
        let fixed = FixedAtoms::build(&pool, COULOMB_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, 0.0, 5.0)),
            hbond: no_hbond(),
        };
        let types = [t(0)];
        let slot = make_slot(&types, &[qi], &[u8::MAX]);
        let coords = [v(0.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&coords])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, Some(diel), f32::INFINITY);
        let expected = (COULOMB_CONST / diel) * qi * qj / (r * r);
        assert_abs_diff_eq!(table.get(0, 0), expected, epsilon = 1e-4);
    }
}
