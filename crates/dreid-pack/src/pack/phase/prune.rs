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
            energy::{PRUNED, RotamerBias, SelfEnergyTable},
            fixed::{FixedAtoms, NO_DONOR},
        },
    },
};
use rayon::prelude::*;

/// Computes self-energies (SC <-> Fixed + rotamer bias), threshold-prunes
/// dead candidates, and compacts both [`SelfEnergyTable`] and
/// [`Conformations`] in sync.
pub fn prune(
    slots: &[Residue],
    conformations: &mut [Conformations],
    fixed: &FixedAtoms<'_>,
    ff: &ForceFieldParams,
    electrostatics: Option<f32>,
    bias: &RotamerBias,
    threshold: f32,
) -> SelfEnergyTable {
    let n = slots.len();
    debug_assert_eq!(n, conformations.len());
    debug_assert_eq!(n, bias.n_slots());

    let c_d = electrostatics.map(|d| COULOMB_CONST / d);

    let mut energies = match (&ff.vdw, c_d) {
        (VdwMatrix::LennardJones(m), None) => {
            frame_energies::<_, false>(&LjKernel(m), slots, conformations, fixed, &ff.hbond, 0.0)
        }
        (VdwMatrix::LennardJones(m), Some(c)) => {
            frame_energies::<_, true>(&LjKernel(m), slots, conformations, fixed, &ff.hbond, c)
        }
        (VdwMatrix::Buckingham(m), None) => {
            frame_energies::<_, false>(&BuckKernel(m), slots, conformations, fixed, &ff.hbond, 0.0)
        }
        (VdwMatrix::Buckingham(m), Some(c)) => {
            frame_energies::<_, true>(&BuckKernel(m), slots, conformations, fixed, &ff.hbond, c)
        }
    };

    let mut counts = Vec::with_capacity(n);
    let mut all_survivors: Vec<(Vec<u16>, Vec<f32>)> = Vec::with_capacity(n);
    for (s, slot_e) in energies.iter_mut().enumerate() {
        for (e, &b) in slot_e.iter_mut().zip(bias.slot(s)) {
            *e += b;
        }
        let e_min = slot_e.iter().copied().fold(PRUNED, f32::min);
        let cutoff = e_min + threshold;
        let (alive, kept): (Vec<u16>, Vec<f32>) = slot_e
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, e)| e <= cutoff)
            .map(|(i, e)| (i as u16, e))
            .unzip();
        counts.push(alive.len() as u16);
        all_survivors.push((alive, kept));
    }

    let mut table = SelfEnergyTable::new(&counts);
    for (s, (alive, kept)) in all_survivors.iter().enumerate() {
        for (r, &e) in kept.iter().enumerate() {
            table.set(s, r, e);
        }
        conformations[s].compact(alive);
    }
    debug_assert!((0..n).all(|s| table.n_candidates(s) == conformations[s].n_candidates()));
    table
}

/// Per-slot atom metadata shared across all candidates of the same residue.
struct SlotAtoms<'a> {
    types: &'a [TypeIdx],
    charges: &'a [f32],
    donors: &'a [u8],
}

/// Parallel over slots × candidates: compute frame energies (SC <-> fixed).
fn frame_energies<V: VdwKernel + Sync, const COUL: bool>(
    vdw: &V,
    slots: &[Residue],
    conformations: &[Conformations],
    fixed: &FixedAtoms<'_>,
    hbond: &HBondParams,
    c_d: f32,
) -> Vec<Vec<f32>> {
    slots
        .par_iter()
        .zip(conformations.par_iter())
        .map(|(slot, confs)| {
            let atoms = SlotAtoms {
                types: slot.atom_types(),
                charges: slot.atom_charges(),
                donors: slot.donor_of_h(),
            };

            (0..confs.n_candidates())
                .map(|r| self_energy::<V, COUL>(confs.coords_of(r), &atoms, fixed, vdw, hbond, c_d))
                .collect()
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

    for (pos_h, (&d_local, &th)) in coords
        .iter()
        .copied()
        .zip(atoms.donors.iter().zip(atoms.types))
    {
        if d_local == u8::MAX {
            continue;
        }
        let d = d_local as usize;
        let pos_d = coords[d];
        let td = atoms.types[d];

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
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let c: Vec<Vec3> = (0..3).map(|i| v(i as f32, 0.0, 0.0)).collect();
        let mut confs = vec![confs_from(1, &[&c[0..1], &c[1..2], &c[2..3]])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![0.0; 3]]),
            PRUNED,
        );
        assert_eq!(table.n_candidates(0), 3);
    }

    #[test]
    fn prune_zero_threshold_prunes_all_but_minimum() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far = [v(50.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&near, &far])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![0.0; 2]]),
            0.0,
        );
        assert_eq!(table.n_candidates(0), 1);
    }

    #[test]
    fn prune_candidate_exceeding_threshold_is_eliminated() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far = [v(50.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&near, &far])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![0.0; 2]]),
            1.0,
        );
        assert_eq!(table.n_candidates(0), 1);
    }

    #[test]
    fn prune_candidate_within_threshold_survives() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far = [v(50.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&near, &far])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![0.0; 2]]),
            3.0,
        );
        assert_eq!(table.n_candidates(0), 2);
    }

    #[test]
    fn prune_table_and_conformations_remain_synchronized() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far = [v(50.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&near, &far])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![0.0; 2]]),
            0.0,
        );
        assert_eq!(table.n_candidates(0), confs[0].n_candidates());
    }

    #[test]
    fn prune_equilibrium_energy_stored_in_table() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&near])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![0.0]]),
            PRUNED,
        );
        assert_abs_diff_eq!(table.get(0, 0), -d0, epsilon = 1e-5);
    }

    #[test]
    fn prune_coulomb_energy_matches_formula() {
        let (r, qi, qj, diel) = (3.0_f32, 1.0_f32, -1.0_f32, 4.0_f32);
        let expected = (COULOMB_CONST / diel) * qi * qj / (r * r);
        let pool = single_atom_pool(v(r, 0.0, 0.0), t(0), qj);
        let fixed = FixedAtoms::build(&pool, COULOMB_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, 0.0, 5.0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[qi], &[u8::MAX]);
        let coords = [v(0.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&coords])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            Some(diel),
            &RotamerBias::new(vec![vec![0.0]]),
            PRUNED,
        );
        assert_abs_diff_eq!(table.get(0, 0), expected, epsilon = 1e-4);
    }

    #[test]
    fn prune_positive_bias_adds_to_candidate_energy() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let b = 1.5_f32;
        let mut confs = vec![confs_from(1, &[&near])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![b]]),
            PRUNED,
        );
        assert_abs_diff_eq!(table.get(0, 0), -d0 + b, epsilon = 1e-5);
    }

    #[test]
    fn prune_large_bias_can_eliminate_favorable_candidate() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far = [v(50.0, 0.0, 0.0)];
        let mut confs = vec![confs_from(1, &[&near, &far])];
        let table = prune(
            &[slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![d0 + 1.0, 0.0]]),
            0.0,
        );
        assert_eq!(table.n_candidates(0), 1);
    }

    #[test]
    fn prune_slots_are_pruned_independently() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let pool = single_atom_pool(v(r0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let near = [v(0.0, 0.0, 0.0)];
        let far_a = [v(50.0, 0.0, 0.0)];
        let far_b = [v(100.0, 0.0, 0.0)];
        let far_c = [v(200.0, 0.0, 0.0)];
        let mut confs = vec![
            confs_from(1, &[&near, &far_a]),
            confs_from(1, &[&far_b, &far_c]),
        ];
        let table = prune(
            &[slot.clone(), slot],
            &mut confs,
            &fixed,
            &ff,
            None,
            &RotamerBias::new(vec![vec![0.0; 2], vec![0.0; 2]]),
            0.0,
        );
        assert_eq!(table.n_candidates(0), 1);
        assert_eq!(table.n_candidates(1), 2);
    }
}
