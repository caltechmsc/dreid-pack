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
