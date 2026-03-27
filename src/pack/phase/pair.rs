use crate::{
    model::{
        system::{ForceFieldParams, HBondParams, Residue, VdwMatrix},
        types::{TypeIdx, Vec3},
    },
    pack::{
        constant::{COULOMB_CONST, COULOMB_CUTOFF_SQ, HBOND_CUTOFF_SQ, VDW_CUTOFF_SQ},
        energy::{BuckKernel, LjKernel, VdwKernel, cos_dha, coulomb_energy, hbond_energy},
        model::{conformation::Conformations, energy::PairEnergyTable, graph::ContactGraph},
    },
};
use rayon::prelude::*;

/// Computes pair energies (SC <-> SC) for every edge in the contact graph,
/// returning a [`PairEnergyTable`] of every pairwise candidate combination.
pub fn pair(
    slots: &[Residue],
    conformations: &[Conformations],
    graph: &ContactGraph,
    ff: &ForceFieldParams,
    electrostatics: Option<f32>,
) -> PairEnergyTable {
    debug_assert_eq!(slots.len(), conformations.len());
    debug_assert_eq!(graph.n_slots(), slots.len());

    let c_d = electrostatics.map(|d| COULOMB_CONST / d);

    let dims: Vec<(u16, u16)> = graph
        .edges()
        .iter()
        .map(|&(si, sj)| {
            (
                conformations[si as usize].n_candidates() as u16,
                conformations[sj as usize].n_candidates() as u16,
            )
        })
        .collect();

    let results = match (&ff.vdw, c_d) {
        (VdwMatrix::LennardJones(m), None) => {
            compute::<_, false>(&LjKernel(m), slots, conformations, graph, &ff.hbond, 0.0)
        }
        (VdwMatrix::LennardJones(m), Some(c)) => {
            compute::<_, true>(&LjKernel(m), slots, conformations, graph, &ff.hbond, c)
        }
        (VdwMatrix::Buckingham(m), None) => {
            compute::<_, false>(&BuckKernel(m), slots, conformations, graph, &ff.hbond, 0.0)
        }
        (VdwMatrix::Buckingham(m), Some(c)) => {
            compute::<_, true>(&BuckKernel(m), slots, conformations, graph, &ff.hbond, c)
        }
    };

    let mut table = PairEnergyTable::new(&dims);
    for (e, energies) in results.into_iter().enumerate() {
        let nj = dims[e].1 as usize;
        for (idx, val) in energies.into_iter().enumerate() {
            table.set(e, idx / nj, idx % nj, val);
        }
    }
    debug_assert!(graph.edges().iter().enumerate().all(|(e, &(si, sj))| {
        let (ni, nj) = table.dims(e);
        ni == conformations[si as usize].n_candidates()
            && nj == conformations[sj as usize].n_candidates()
    }));
    table
}

/// Per-slot atom metadata shared across all candidates of the same residue.
struct SlotAtoms<'a> {
    types: &'a [TypeIdx],
    charges: &'a [f32],
    donors: &'a [u8],
}

/// Parallel over edges × rotamer pairs: compute per-edge energy matrices.
fn compute<V: VdwKernel + Sync, const COUL: bool>(
    vdw: &V,
    slots: &[Residue],
    conformations: &[Conformations],
    graph: &ContactGraph,
    hbond: &HBondParams,
    c_d: f32,
) -> Vec<Vec<f32>> {
    graph
        .edges()
        .par_iter()
        .map(|&(si, sj)| {
            let (si, sj) = (si as usize, sj as usize);
            let atoms_i = SlotAtoms {
                types: slots[si].atom_types(),
                charges: slots[si].atom_charges(),
                donors: slots[si].donor_of_h(),
            };
            let atoms_j = SlotAtoms {
                types: slots[sj].atom_types(),
                charges: slots[sj].atom_charges(),
                donors: slots[sj].donor_of_h(),
            };
            let confs_i = &conformations[si];
            let confs_j = &conformations[sj];
            let nj = confs_j.n_candidates();

            (0..confs_i.n_candidates() * nj)
                .into_par_iter()
                .map(|idx| {
                    pair_energy::<V, COUL>(
                        confs_i.coords_of(idx / nj),
                        &atoms_i,
                        confs_j.coords_of(idx % nj),
                        &atoms_j,
                        vdw,
                        hbond,
                        c_d,
                    )
                })
                .collect()
        })
        .collect()
}

/// Non-bonded pair energy between two candidate conformations.
fn pair_energy<V: VdwKernel, const COUL: bool>(
    coords_i: &[Vec3],
    atoms_i: &SlotAtoms<'_>,
    coords_j: &[Vec3],
    atoms_j: &SlotAtoms<'_>,
    vdw: &V,
    hbond: &HBondParams,
    c_d: f32,
) -> f32 {
    let mut e = 0.0_f32;

    for (pos_i, (&ti, &qi)) in coords_i
        .iter()
        .copied()
        .zip(atoms_i.types.iter().zip(atoms_i.charges))
    {
        for (pos_j, (&tj, &qj)) in coords_j
            .iter()
            .copied()
            .zip(atoms_j.types.iter().zip(atoms_j.charges))
        {
            let r_sq = pos_i.dist_sq(pos_j);
            if r_sq <= VDW_CUTOFF_SQ {
                e += vdw.energy(ti, tj, r_sq);
                e += coulomb_energy::<COUL>(c_d, qi, qj, r_sq);
            } else if COUL && r_sq <= COULOMB_CUTOFF_SQ {
                e += coulomb_energy::<COUL>(c_d, qi, qj, r_sq);
            }
        }
    }

    for (pos_h, (&d_local, &th)) in coords_i
        .iter()
        .copied()
        .zip(atoms_i.donors.iter().zip(atoms_i.types))
    {
        if d_local == u8::MAX {
            continue;
        }
        let d = d_local as usize;
        let pos_d = coords_i[d];
        let td = atoms_i.types[d];

        for (pos_a, &tj) in coords_j.iter().copied().zip(atoms_j.types) {
            let r_sq_da = pos_d.dist_sq(pos_a);
            if r_sq_da <= HBOND_CUTOFF_SQ {
                e += hbond_energy(r_sq_da, cos_dha(pos_d, pos_h, pos_a), td, th, tj, hbond);
            }
        }
    }

    for (pos_h, (&d_local, &th)) in coords_j
        .iter()
        .copied()
        .zip(atoms_j.donors.iter().zip(atoms_j.types))
    {
        if d_local == u8::MAX {
            continue;
        }
        let d = d_local as usize;
        let pos_d = coords_j[d];
        let td = atoms_j.types[d];

        for (pos_a, &ti) in coords_i.iter().copied().zip(atoms_i.types) {
            let r_sq_da = pos_d.dist_sq(pos_a);
            if r_sq_da <= HBOND_CUTOFF_SQ {
                e += hbond_energy(r_sq_da, cos_dha(pos_d, pos_h, pos_a), td, th, ti, hbond);
            }
        }
    }

    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{
            residue::ResidueType,
            system::{ForceFieldParams, LjMatrix, LjPair, SidechainAtoms, VdwMatrix},
        },
        pack::{
            constant::{COULOMB_CUTOFF, HBOND_CUTOFF, VDW_CUTOFF},
            model::{conformation::Conformations, graph::ContactGraph},
        },
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
    fn vdw_at_equilibrium_equals_minus_d0() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let lj = lj_n(1, d0, r0);
        let atoms = SlotAtoms {
            types: &[t(0)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0)],
            &atoms,
            &[v(r0, 0.0, 0.0)],
            &atoms,
            &LjKernel(&lj),
            &no_hbond(),
            0.0,
        );
        assert_abs_diff_eq!(e, -d0, epsilon = 1e-5);
    }

    #[test]
    fn vdw_repulsive_at_close_range() {
        let r_close = 1.5_f32;
        let lj = lj_n(1, 1.0, 3.0);
        let atoms = SlotAtoms {
            types: &[t(0)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0)],
            &atoms,
            &[v(r_close, 0.0, 0.0)],
            &atoms,
            &LjKernel(&lj),
            &no_hbond(),
            0.0,
        );
        assert!(e > 0.0, "expected VdW repulsion at r={r_close}, got {e}");
    }

    #[test]
    fn vdw_beyond_cutoff_contributes_nothing() {
        let lj = lj_n(1, 2.0, 3.0);
        let atoms = SlotAtoms {
            types: &[t(0)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0)],
            &atoms,
            &[v(VDW_CUTOFF + 0.1, 0.0, 0.0)],
            &atoms,
            &LjKernel(&lj),
            &no_hbond(),
            0.0,
        );
        assert_eq!(e, 0.0);
    }

    #[test]
    fn vdw_absent_in_coulomb_only_range() {
        let (r, qi, qj, diel) = (
            (VDW_CUTOFF + COULOMB_CUTOFF) / 2.0,
            1.0_f32,
            -1.0_f32,
            4.0_f32,
        );
        let lj = lj_n(1, 5.0, 3.0);
        let atoms_p = SlotAtoms {
            types: &[t(0)],
            charges: &[qi],
            donors: &[u8::MAX],
        };
        let atoms_n = SlotAtoms {
            types: &[t(0)],
            charges: &[qj],
            donors: &[u8::MAX],
        };
        let c_d = COULOMB_CONST / diel;
        let expected = c_d * qi * qj / (r * r);
        let e = pair_energy::<_, true>(
            &[v(0.0, 0.0, 0.0)],
            &atoms_p,
            &[v(r, 0.0, 0.0)],
            &atoms_n,
            &LjKernel(&lj),
            &no_hbond(),
            c_d,
        );
        assert_abs_diff_eq!(e, expected, epsilon = 1e-5);
    }

    #[test]
    fn coulomb_opposite_charges_attractive() {
        let r = 2.0_f32;
        let atoms_p = SlotAtoms {
            types: &[t(0)],
            charges: &[1.0],
            donors: &[u8::MAX],
        };
        let atoms_n = SlotAtoms {
            types: &[t(0)],
            charges: &[-1.0],
            donors: &[u8::MAX],
        };
        let lj = lj_n(1, 0.0, 5.0);
        let e = pair_energy::<_, true>(
            &[v(0.0, 0.0, 0.0)],
            &atoms_p,
            &[v(r, 0.0, 0.0)],
            &atoms_n,
            &LjKernel(&lj),
            &no_hbond(),
            COULOMB_CONST / 4.0,
        );
        assert!(e < 0.0, "opposite charges must attract, got {e}");
    }

    #[test]
    fn coulomb_same_sign_charges_repulsive() {
        let r = 2.0_f32;
        let atoms = SlotAtoms {
            types: &[t(0)],
            charges: &[1.0],
            donors: &[u8::MAX],
        };
        let lj = lj_n(1, 0.0, 5.0);
        let e = pair_energy::<_, true>(
            &[v(0.0, 0.0, 0.0)],
            &atoms,
            &[v(r, 0.0, 0.0)],
            &atoms,
            &LjKernel(&lj),
            &no_hbond(),
            COULOMB_CONST / 4.0,
        );
        assert!(e > 0.0, "same-sign charges must repel, got {e}");
    }

    #[test]
    fn coulomb_extended_range_branch_is_active() {
        let r = VDW_CUTOFF + 0.5;
        let atoms_p = SlotAtoms {
            types: &[t(0)],
            charges: &[1.0],
            donors: &[u8::MAX],
        };
        let atoms_n = SlotAtoms {
            types: &[t(0)],
            charges: &[-1.0],
            donors: &[u8::MAX],
        };
        let lj = lj_n(1, 0.0, 5.0);
        let e = pair_energy::<_, true>(
            &[v(0.0, 0.0, 0.0)],
            &atoms_p,
            &[v(r, 0.0, 0.0)],
            &atoms_n,
            &LjKernel(&lj),
            &no_hbond(),
            COULOMB_CONST / 4.0,
        );
        assert!(
            e < 0.0,
            "Coulomb extended-range branch must fire at r={r}, got {e}"
        );
    }

    #[test]
    fn coulomb_beyond_cutoff_contributes_nothing() {
        let atoms_p = SlotAtoms {
            types: &[t(0)],
            charges: &[1.0],
            donors: &[u8::MAX],
        };
        let atoms_n = SlotAtoms {
            types: &[t(0)],
            charges: &[-1.0],
            donors: &[u8::MAX],
        };
        let lj = lj_n(1, 0.0, 5.0);
        let e = pair_energy::<_, true>(
            &[v(0.0, 0.0, 0.0)],
            &atoms_p,
            &[v(COULOMB_CUTOFF + 0.1, 0.0, 0.0)],
            &atoms_n,
            &LjKernel(&lj),
            &no_hbond(),
            COULOMB_CONST / 4.0,
        );
        assert_eq!(e, 0.0);
    }

    #[test]
    fn coulomb_disabled_contributes_nothing() {
        let atoms = SlotAtoms {
            types: &[t(0)],
            charges: &[100.0],
            donors: &[u8::MAX],
        };
        let lj = lj_n(1, 0.0, 5.0);
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0)],
            &atoms,
            &[v(2.0, 0.0, 0.0)],
            &atoms,
            &LjKernel(&lj),
            &no_hbond(),
            0.0,
        );
        assert_eq!(e, 0.0);
    }

    #[test]
    fn coulomb_exact_energy_matches_formula() {
        let (r, qi, qj, diel) = (3.0_f32, 0.5_f32, -0.5_f32, 4.0_f32);
        let atoms_i = SlotAtoms {
            types: &[t(0)],
            charges: &[qi],
            donors: &[u8::MAX],
        };
        let atoms_j = SlotAtoms {
            types: &[t(0)],
            charges: &[qj],
            donors: &[u8::MAX],
        };
        let lj = lj_n(1, 0.0, 5.0);
        let c_d = COULOMB_CONST / diel;
        let expected = c_d * qi * qj / (r * r);
        let e = pair_energy::<_, true>(
            &[v(0.0, 0.0, 0.0)],
            &atoms_i,
            &[v(r, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &no_hbond(),
            c_d,
        );
        assert_abs_diff_eq!(e, expected, epsilon = 1e-5);
    }

    #[test]
    fn hbond_ij_linear_geometry_is_attractive() {
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let atoms_i = SlotAtoms {
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let atoms_j = SlotAtoms {
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            &atoms_i,
            &[v(3.0, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &hbond,
            0.0,
        );
        assert!(e < 0.0, "linear i->j H-bond must be attractive, got {e}");
    }

    #[test]
    fn hbond_ij_at_equilibrium_equals_minus_d_hb() {
        let (d_hb, r_hb) = (3.0_f32, 2.0_f32);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(d_hb, r_hb);
        let atoms_i = SlotAtoms {
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let atoms_j = SlotAtoms {
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            &atoms_i,
            &[v(r_hb, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &hbond,
            0.0,
        );
        assert_abs_diff_eq!(e, -d_hb, epsilon = 1e-5);
    }

    #[test]
    fn hbond_ij_obtuse_geometry_contributes_zero() {
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let atoms_i = SlotAtoms {
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let atoms_j = SlotAtoms {
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            &atoms_i,
            &[v(-0.5, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &hbond,
            0.0,
        );
        assert_eq!(e, 0.0, "obtuse D-H-A geometry must yield zero, got {e}");
    }

    #[test]
    fn hbond_ij_da_beyond_cutoff_contributes_zero() {
        let d_a = HBOND_CUTOFF + 0.1;
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let atoms_i = SlotAtoms {
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let atoms_j = SlotAtoms {
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            &atoms_i,
            &[v(d_a, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &hbond,
            0.0,
        );
        assert_eq!(
            e, 0.0,
            "D-A beyond HBOND_CUTOFF must contribute zero, got {e}"
        );
    }

    #[test]
    fn hbond_ij_no_donor_flag_skips_loop() {
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let atoms_i = SlotAtoms {
            types: &[t(1)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let atoms_j = SlotAtoms {
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let e = pair_energy::<_, false>(
            &[v(1.0, 0.0, 0.0)],
            &atoms_i,
            &[v(3.0, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &hbond,
            0.0,
        );
        assert_eq!(e, 0.0, "no-donor flag must skip i->j H-bond loop, got {e}");
    }

    #[test]
    fn hbond_ji_linear_geometry_is_attractive() {
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(4.0, 3.0);
        let atoms_i = SlotAtoms {
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let atoms_j = SlotAtoms {
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let e = pair_energy::<_, false>(
            &[v(3.0, 0.0, 0.0)],
            &atoms_i,
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &hbond,
            0.0,
        );
        assert!(e < 0.0, "linear j->i H-bond must be attractive, got {e}");
    }

    #[test]
    fn hbond_ji_at_equilibrium_equals_minus_d_hb() {
        let (d_hb, r_hb) = (3.0_f32, 2.0_f32);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(d_hb, r_hb);
        let atoms_i = SlotAtoms {
            types: &[t(2)],
            charges: &[0.0],
            donors: &[u8::MAX],
        };
        let atoms_j = SlotAtoms {
            types: &[t(0), t(1)],
            charges: &[0.0, 0.0],
            donors: &[u8::MAX, 0],
        };
        let e = pair_energy::<_, false>(
            &[v(r_hb, 0.0, 0.0)],
            &atoms_i,
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &hbond,
            0.0,
        );
        assert_abs_diff_eq!(e, -d_hb, epsilon = 1e-5);
    }

    #[test]
    fn hbond_both_directions_sum_to_two_bonds() {
        let (d_hb, r_hb) = (3.0_f32, 2.0_f32);
        let lj = lj_n(3, 0.0, 5.0);
        let hbond = hbond_dha(d_hb, r_hb);
        let big = 50.0_f32;
        let atoms_i = SlotAtoms {
            types: &[t(0), t(1), t(2)],
            charges: &[0.0, 0.0, 0.0],
            donors: &[u8::MAX, 0, u8::MAX],
        };
        let atoms_j = SlotAtoms {
            types: &[t(2), t(0), t(1)],
            charges: &[0.0, 0.0, 0.0],
            donors: &[u8::MAX, u8::MAX, 1],
        };
        let e = pair_energy::<_, false>(
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(big + r_hb, 0.0, 0.0)],
            &atoms_i,
            &[v(r_hb, 0.0, 0.0), v(big, 0.0, 0.0), v(big + 1.0, 0.0, 0.0)],
            &atoms_j,
            &LjKernel(&lj),
            &hbond,
            0.0,
        );
        assert_abs_diff_eq!(e, -2.0 * d_hb, epsilon = 1e-5);
    }

    #[test]
    fn pair_no_edges_returns_empty_table() {
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, 1.0, 3.0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let confs = vec![confs_from(1, &[&[v(0.0, 0.0, 0.0)]])];
        let graph = ContactGraph::build(1, []);
        let table = pair(&[slot], &confs, &graph, &ff, None);
        assert_eq!(table.n_edges(), 0);
    }

    #[test]
    fn pair_table_dimensions_match_conformations() {
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, 0.0, 5.0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let far = v(100.0, 0.0, 0.0);
        let confs = vec![
            confs_from(1, &[&[far], &[far], &[far]]),
            confs_from(1, &[&[far], &[far]]),
        ];
        let graph = ContactGraph::build(2, [(0, 1)]);
        let table = pair(&[slot.clone(), slot], &confs, &graph, &ff, None);
        assert_eq!(table.n_edges(), 1);
        assert_eq!(table.dims(0), (3, 2));
    }

    #[test]
    fn pair_vdw_at_equilibrium_stored_in_table() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let confs = vec![
            confs_from(1, &[&[v(0.0, 0.0, 0.0)]]),
            confs_from(1, &[&[v(r0, 0.0, 0.0)]]),
        ];
        let graph = ContactGraph::build(2, [(0, 1)]);
        let table = pair(&[slot.clone(), slot], &confs, &graph, &ff, None);
        assert_abs_diff_eq!(table.get(0, 0, 0), -d0, epsilon = 1e-5);
    }

    #[test]
    fn pair_four_candidate_combinations_all_correct() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let confs = vec![
            confs_from(1, &[&[v(0.0, 0.0, 0.0)], &[v(100.0, 0.0, 0.0)]]),
            confs_from(1, &[&[v(r0, 0.0, 0.0)], &[v(200.0, 0.0, 0.0)]]),
        ];
        let graph = ContactGraph::build(2, [(0, 1)]);
        let table = pair(&[slot.clone(), slot], &confs, &graph, &ff, None);
        assert_abs_diff_eq!(table.get(0, 0, 0), -d0, epsilon = 1e-5);
        assert_abs_diff_eq!(table.get(0, 0, 1), 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(table.get(0, 1, 0), 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(table.get(0, 1, 1), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn pair_electrostatics_enabled_lowers_opposite_charge_energy() {
        let (r, qi, qj, diel) = (3.0_f32, 1.0_f32, -1.0_f32, 4.0_f32);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, 0.0, 5.0)),
            hbond: no_hbond(),
        };
        let slot_i = make_slot(&[t(0)], &[qi], &[u8::MAX]);
        let slot_j = make_slot(&[t(0)], &[qj], &[u8::MAX]);
        let confs = vec![
            confs_from(1, &[&[v(0.0, 0.0, 0.0)]]),
            confs_from(1, &[&[v(r, 0.0, 0.0)]]),
        ];
        let graph = ContactGraph::build(2, [(0, 1)]);
        let e_no = pair(&[slot_i.clone(), slot_j.clone()], &confs, &graph, &ff, None).get(0, 0, 0);
        let e_yes = pair(&[slot_i, slot_j], &confs, &graph, &ff, Some(diel)).get(0, 0, 0);
        assert!(
            e_yes < e_no,
            "electrostatics must lower energy for opposite charges: no={e_no}, yes={e_yes}",
        );
    }

    #[test]
    fn pair_multiple_edges_computed_independently() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_n(1, d0, r0)),
            hbond: no_hbond(),
        };
        let slot = make_slot(&[t(0)], &[0.0], &[u8::MAX]);
        let confs = vec![
            confs_from(1, &[&[v(0.0, 0.0, 0.0)]]),
            confs_from(1, &[&[v(r0, 0.0, 0.0)]]),
            confs_from(1, &[&[v(100.0, 0.0, 0.0)]]),
        ];
        let graph = ContactGraph::build(3, [(0, 1), (0, 2), (1, 2)]);
        let table = pair(
            &[slot.clone(), slot.clone(), slot],
            &confs,
            &graph,
            &ff,
            None,
        );
        assert_abs_diff_eq!(table.get(0, 0, 0), -d0, epsilon = 1e-5);
        assert_abs_diff_eq!(table.get(1, 0, 0), 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(table.get(2, 0, 0), 0.0, epsilon = 1e-5);
    }
}
