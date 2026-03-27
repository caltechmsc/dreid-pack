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

    for a in 0..atoms.n_a {
        let pos_a = coords[a];
        let ta = atoms.types[a];

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

            e += coulomb_energy::<COUL>(c_d, atoms.charges[a], fixed.charges[b], r_sq);
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
    use std::collections::{HashMap, HashSet};

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn t(n: u8) -> TypeIdx {
        TypeIdx(n)
    }

    fn lj1(d0: f32, r0: f32) -> LjMatrix {
        LjMatrix::new(1, vec![LjPair { d0, r0_sq: r0 * r0 }])
    }

    fn ff_lj_no_hbond(d0: f32, r0: f32) -> ForceFieldParams {
        ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj1(d0, r0)),
            hbond: HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new()),
        }
    }

    fn lj3(d0: f32, r0: f32) -> LjMatrix {
        let p = LjPair { d0, r0_sq: r0 * r0 };
        LjMatrix::new(3, vec![p; 9])
    }

    fn ff_lj_with_hbond(d0: f32, r0: f32, d_hb: f32, r_hb: f32) -> ForceFieldParams {
        let mut h_types = HashSet::new();
        let mut acc_types = HashSet::new();
        let mut params = HashMap::new();
        h_types.insert(t(1));
        acc_types.insert(t(2));
        params.insert((t(0), t(1), t(2)), (d_hb, r_hb * r_hb));
        ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj3(d0, r0)),
            hbond: HBondParams::new(h_types, acc_types, params),
        }
    }

    fn empty_pool() -> FixedAtomPool {
        FixedAtomPool {
            positions: vec![],
            types: vec![],
            charges: vec![],
            donor_for_h: vec![],
        }
    }

    fn single_atom_pool(pos: Vec3, ty: TypeIdx, q: f32) -> FixedAtomPool {
        FixedAtomPool {
            positions: vec![pos],
            types: vec![ty],
            charges: vec![q],
            donor_for_h: vec![NO_DONOR],
        }
    }

    fn donor_h_pool(pos_d: Vec3, pos_h: Vec3, ty_d: TypeIdx, ty_h: TypeIdx) -> FixedAtomPool {
        FixedAtomPool {
            positions: vec![pos_d, pos_h],
            types: vec![ty_d, ty_h],
            charges: vec![0.0, 0.0],
            donor_for_h: vec![NO_DONOR, 0],
        }
    }

    fn make_slot(n: usize, pos: Vec3) -> Residue {
        let coords = vec![pos; n];
        let types = vec![t(0); n];
        let charges = vec![0.0f32; n];
        let donor_of_h = vec![u8::MAX; n];
        let anchor = [v(1.458, 0.0, 0.0), Vec3::zero(), v(-0.524, 0.0, 1.454)];
        Residue::new(
            ResidueType::Val,
            anchor,
            -1.047,
            -0.698,
            std::f32::consts::PI,
            SidechainAtoms {
                coords: &coords,
                types: &types,
                charges: &charges,
                donor_of_h: &donor_of_h,
            },
        )
        .unwrap()
    }

    fn confs_1atom(positions: &[Vec3]) -> Conformations {
        let n_cands = positions.len() as u16;
        Conformations::new(positions.to_vec(), n_cands, 1)
    }

    fn slot_atoms_plain<'a>(types: &'a [TypeIdx], charges: &'a [f32]) -> SlotAtoms<'a> {
        static DONORS: &[u8] = &[u8::MAX];
        SlotAtoms {
            n_a: types.len(),
            types,
            charges,
            donors: DONORS,
        }
    }

    #[test]
    fn empty_slots_returns_empty_table() {
        let pool = empty_pool();
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let table = prune(&[], &mut [], &fixed, &ff, None, 0.0);
        assert_eq!(table.n_slots(), 0);
    }

    #[test]
    fn table_n_slots_matches_slot_count() {
        let pool = empty_pool();
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot = make_slot(1, v(100.0, 0.0, 0.0));
        let mut confs = vec![confs_1atom(&[v(100.0, 0.0, 0.0), v(101.0, 0.0, 0.0)])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert_eq!(table.n_slots(), 1);
    }

    #[test]
    fn table_and_conformations_stay_in_sync_after_prune() {
        let pool = empty_pool();
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot = make_slot(1, v(100.0, 0.0, 0.0));
        let positions = [v(100.0, 0.0, 0.0), v(101.0, 0.0, 0.0), v(102.0, 0.0, 0.0)];
        let mut confs = vec![confs_1atom(&positions)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert_eq!(table.n_candidates(0), confs[0].n_candidates());
    }

    #[test]
    fn no_fixed_neighbors_all_candidates_survive() {
        let pool = empty_pool();
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot = make_slot(1, v(100.0, 0.0, 0.0));
        let positions = [v(100.0, 0.0, 0.0), v(101.0, 0.0, 0.0), v(102.0, 0.0, 0.0)];
        let mut confs = vec![confs_1atom(&positions)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.0);
        assert_eq!(table.n_candidates(0), 3);
    }

    #[test]
    fn no_fixed_neighbors_energy_is_zero() {
        let pool = empty_pool();
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let types = [t(0)];
        let charges = [0.0f32];
        let atoms = slot_atoms_plain(&types, &charges);
        let coords = [v(100.0, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(1.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert_eq!(e, 0.0);
    }

    #[test]
    fn threshold_zero_keeps_only_minimum_energy_candidates() {
        let pool = single_atom_pool(v(1.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot = make_slot(1, v(100.0, 0.0, 0.0));
        let mut confs = vec![confs_1atom(&[v(0.0, 0.0, 0.0), v(100.0, 0.0, 0.0)])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.0);
        assert_eq!(table.n_candidates(0), 1);
    }

    #[test]
    fn threshold_infinity_all_candidates_survive() {
        let pool = single_atom_pool(v(1.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot = make_slot(1, v(100.0, 0.0, 0.0));
        let mut confs = vec![confs_1atom(&[v(0.0, 0.0, 0.0), v(100.0, 0.0, 0.0)])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert_eq!(table.n_candidates(0), 2);
    }

    #[test]
    fn threshold_zero_guarantees_at_least_one_survivor() {
        let pool = empty_pool();
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot = make_slot(1, v(100.0, 0.0, 0.0));
        let mut confs = vec![confs_1atom(&[v(100.0, 0.0, 0.0); 3])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.0);
        assert!(table.n_candidates(0) >= 1);
    }

    #[test]
    fn close_fixed_atom_raises_vdw_energy() {
        let pool = single_atom_pool(v(1.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let types = [t(0)];
        let charges = [0.0f32];
        let atoms = slot_atoms_plain(&types, &charges);
        let coords = [v(0.0, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(1.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert!(e > 0.0, "expected repulsive energy, got {e}");
    }

    #[test]
    fn atom_beyond_vdw_cutoff_contributes_zero_vdw() {
        let far = VDW_CUTOFF + 1.0;
        let pool = single_atom_pool(v(far, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let types = [t(0)];
        let charges = [0.0f32];
        let atoms = slot_atoms_plain(&types, &charges);
        let coords = [v(0.0, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(1.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert_eq!(e, 0.0);
    }

    #[test]
    fn atom_at_vdw_cutoff_does_not_contribute_when_with_coulomb_query_radius() {
        let r = VDW_CUTOFF + 0.5;
        let pool = single_atom_pool(v(r, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF + 1.0);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let types = [t(0)];
        let charges = [0.0f32];
        let atoms = slot_atoms_plain(&types, &charges);
        let coords = [v(0.0, 0.0, 0.0)];
        let e = self_energy::<_, true>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(1.0, 3.0)),
            &ff.hbond,
            332.0,
        );
        assert_eq!(
            e, 0.0,
            "zero charge -> zero coulomb; VdW guard excludes the atom"
        );
    }

    #[test]
    fn buckingham_dispatch_produces_finite_energy() {
        use crate::model::system::{BuckMatrix, BuckPair};
        let pair = BuckPair {
            a: 1000.0,
            b: 3.5,
            c: 50.0,
            r_max_sq: 0.5 * 0.5,
            two_e_max: 2.0 * 50.0,
        };
        let buck = BuckMatrix::new(1, vec![pair]);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::Buckingham(buck),
            hbond: HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new()),
        };
        let pool = single_atom_pool(v(3.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let slot = make_slot(1, v(100.0, 0.0, 0.0));
        let mut confs = vec![confs_1atom(&[v(0.0, 0.0, 0.0)])];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert!(table.get(0, 0).is_finite());
    }

    #[test]
    fn coulomb_disabled_ignores_charges() {
        let pool = single_atom_pool(v(2.0, 0.0, 0.0), t(0), 100.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(0.0, 3.0);
        let types = [t(0)];
        let charges = [1.0f32];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX],
        };
        let coords = [v(0.0, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            COULOMB_CONST,
        );
        assert_eq!(e, 0.0, "COUL=false must ignore charges");
    }

    #[test]
    fn coulomb_enabled_adds_nonzero_energy_for_like_charges() {
        let pool = single_atom_pool(v(2.0, 0.0, 0.0), t(0), 1.0);
        let fixed = FixedAtoms::build(&pool, 9.0);
        let ff = ff_lj_no_hbond(0.0, 3.0);
        let types = [t(0)];
        let charges = [1.0f32];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX],
        };
        let coords = [v(0.0, 0.0, 0.0)];
        let c_d = COULOMB_CONST / 4.0;
        let e = self_energy::<_, true>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            c_d,
        );
        assert!(e > 0.0, "like charges: expected repulsion, got {e}");
    }

    #[test]
    fn coulomb_enabled_opposite_charges_give_negative_energy() {
        let pool = single_atom_pool(v(2.0, 0.0, 0.0), t(0), -1.0);
        let fixed = FixedAtoms::build(&pool, 9.0);
        let ff = ff_lj_no_hbond(0.0, 3.0);
        let types = [t(0)];
        let charges = [1.0f32];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX],
        };
        let coords = [v(0.0, 0.0, 0.0)];
        let c_d = COULOMB_CONST / 4.0;
        let e = self_energy::<_, true>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            c_d,
        );
        assert!(e < 0.0, "opposite charges: expected attraction, got {e}");
    }

    #[test]
    fn fixed_donor_hbond_linear_geometry_negative_energy() {
        let pool = donor_h_pool(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), t(0), t(1));
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_with_hbond(0.0, 3.0, 8.0, 3.0);
        let types = [t(2)];
        let charges = [0.0f32];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX],
        };
        let coords = [v(3.0, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert!(
            e < 0.0,
            "linear fixed-donor H-bond must be attractive, got {e}"
        );
    }

    #[test]
    fn fixed_donor_hbond_obtuse_geometry_zero_energy() {
        let pool = donor_h_pool(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), t(0), t(1));
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_with_hbond(0.0, 3.0, 8.0, 3.0);
        let types = [t(2)];
        let charges = [0.0f32];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX],
        };
        let coords = [v(0.5, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert_eq!(e, 0.0, "obtuse angle: H-bond energy must be zero, got {e}");
    }

    #[test]
    fn fixed_donor_hbond_da_beyond_cutoff_contributes_zero() {
        let d_a = HBOND_CUTOFF + 0.1;
        let pool = donor_h_pool(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), t(0), t(1));
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF + 1.0);
        let ff = ff_lj_with_hbond(0.0, 3.0, 8.0, 3.0);
        let types = [t(2)];
        let charges = [0.0f32];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX],
        };
        let coords = [v(d_a, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert_eq!(
            e, 0.0,
            "D-A beyond cutoff: guard must suppress H-bond, got {e}"
        );
    }

    #[test]
    fn sc_donor_hbond_linear_geometry_negative_energy() {
        let pool = single_atom_pool(v(3.0, 0.0, 0.0), t(2), 0.0);
        let fixed = FixedAtoms::build(&pool, HBOND_CUTOFF + 1.0);
        let ff = ff_lj_with_hbond(0.0, 3.0, 8.0, 3.0);
        let types = [t(0), t(1)];
        let charges = [0.0f32, 0.0];
        let atoms = SlotAtoms {
            n_a: 2,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX, 0],
        };
        let coords = [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert!(
            e < 0.0,
            "linear SC-donor H-bond must be attractive, got {e}"
        );
    }

    #[test]
    fn sc_donor_hbond_no_donor_flag_skips_h_atoms() {
        let pool = single_atom_pool(v(3.0, 0.0, 0.0), t(2), 0.0);
        let fixed = FixedAtoms::build(&pool, HBOND_CUTOFF + 1.0);
        let ff = ff_lj_with_hbond(0.0, 3.0, 8.0, 3.0);
        let types = [t(1)];
        let charges = [0.0f32];
        let atoms = SlotAtoms {
            n_a: 1,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX],
        };
        let coords = [v(1.0, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert_eq!(
            e, 0.0,
            "no donor flag: SC-donor loop must be skipped, got {e}"
        );
    }

    #[test]
    fn sc_donor_hbond_fixed_acceptor_beyond_hbond_cutoff_zero() {
        let far = HBOND_CUTOFF + 1.0;
        let pool = single_atom_pool(v(far, 0.0, 0.0), t(2), 0.0);
        let fixed = FixedAtoms::build(&pool, HBOND_CUTOFF + 2.0);
        let ff = ff_lj_with_hbond(0.0, 3.0, 8.0, 3.0);
        let types = [t(0), t(1)];
        let charges = [0.0f32, 0.0];
        let atoms = SlotAtoms {
            n_a: 2,
            types: &types,
            charges: &charges,
            donors: &[u8::MAX, 0],
        };
        let coords = [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
        let e = self_energy::<_, false>(
            &coords,
            &atoms,
            &fixed,
            &LjKernel(&lj1(0.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert_eq!(
            e, 0.0,
            "acceptor beyond HBOND_CUTOFF: must contribute zero, got {e}"
        );
    }

    #[test]
    fn multiple_slots_energies_are_independent() {
        let pool = single_atom_pool(v(1.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot0 = make_slot(1, v(0.0, 0.0, 0.0));
        let slot1 = make_slot(1, v(100.0, 0.0, 0.0));
        let mut confs = vec![
            confs_1atom(&[v(0.0, 0.0, 0.0)]),
            confs_1atom(&[v(100.0, 0.0, 0.0)]),
        ];
        let table = prune(
            &[slot0, slot1],
            &mut confs,
            &fixed,
            &ff,
            None,
            f32::INFINITY,
        );
        assert!(table.get(0, 0) > 0.0, "slot 0: repulsive energy expected");
        assert_eq!(table.get(1, 0), 0.0, "slot 1: zero energy expected");
    }

    #[test]
    fn pruning_one_slot_does_not_affect_other() {
        let pool = single_atom_pool(v(1.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot0 = make_slot(1, v(0.0, 0.0, 0.0));
        let slot1 = make_slot(1, v(100.0, 0.0, 0.0));
        let mut confs = vec![
            confs_1atom(&[v(0.0, 0.0, 0.0), v(100.0, 0.0, 0.0)]),
            confs_1atom(&[v(200.0, 0.0, 0.0)]),
        ];
        let table = prune(&[slot0, slot1], &mut confs, &fixed, &ff, None, 0.0);
        assert_eq!(
            table.n_candidates(0),
            1,
            "repulsive candidate must be pruned"
        );
        assert_eq!(table.n_candidates(1), 1, "isolated slot must be unaffected");
    }

    #[test]
    fn all_surviving_energies_are_finite() {
        let pool = single_atom_pool(v(3.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let slot = make_slot(1, v(100.0, 0.0, 0.0));
        let positions: Vec<Vec3> = (0..5).map(|i| v(i as f32 * 10.0, 0.0, 0.0)).collect();
        let mut confs = vec![confs_1atom(&positions)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        for r in 0..table.n_candidates(0) {
            assert!(table.get(0, r).is_finite(), "energy[{r}] is not finite");
        }
    }

    #[test]
    fn closer_distance_raises_lj_energy_monotonically() {
        let pool = single_atom_pool(v(0.0, 0.0, 0.0), t(0), 0.0);
        let fixed = FixedAtoms::build(&pool, VDW_CUTOFF);
        let ff = ff_lj_no_hbond(1.0, 3.0);
        let types = [t(0)];
        let charges = [0.0f32];
        let atoms = slot_atoms_plain(&types, &charges);
        let e1 = self_energy::<_, false>(
            &[v(1.0, 0.0, 0.0)],
            &atoms,
            &fixed,
            &LjKernel(&lj1(1.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        let e4 = self_energy::<_, false>(
            &[v(4.0, 0.0, 0.0)],
            &atoms,
            &fixed,
            &LjKernel(&lj1(1.0, 3.0)),
            &ff.hbond,
            0.0,
        );
        assert!(
            e1 > e4,
            "close distance must be more repulsive: e1={e1} e4={e4}"
        );
    }
}
