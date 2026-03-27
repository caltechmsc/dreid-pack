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
