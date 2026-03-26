use crate::pack::constant::{
    COULOMB_CONST, COULOMB_CUTOFF, COULOMB_CUTOFF_SQ, HBOND_CUTOFF_SQ, HBOND_N, VDW_CUTOFF,
    VDW_CUTOFF_SQ,
};
use crate::{
    model::{
        system::{ForceFieldParams, Residue, VdwMatrix},
        types::{TypeIdx, Vec3},
    },
    pack::model::{
        conformation::Conformations,
        energy::SelfEnergyTable,
        fixed::{FixedAtoms, NO_DONOR},
    },
};
use dreid_kernel::{
    HybridKernel, PairKernel,
    potentials::nonbonded::{Buckingham, HydrogenBond, LennardJones},
};
use rayon::prelude::*;

/// Computes [`SelfEnergyTable`], applies threshold pruning, and compacts in sync.
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

    let coulomb_scale = electrostatics.map(|d| COULOMB_CONST / d);

    let slot_energies: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|s| slot_self_energies(&slots[s], &conformations[s], fixed, ff, coulomb_scale))
        .collect();

    let counts: Vec<u16> = slot_energies.iter().map(|e| e.len() as u16).collect();
    let mut table = SelfEnergyTable::new(&counts);

    for (s, energies) in slot_energies.iter().enumerate() {
        let e_min = energies.iter().copied().fold(f32::INFINITY, f32::min);
        let cutoff = e_min + threshold;
        for (r, &e) in energies.iter().enumerate() {
            if e > cutoff {
                table.prune(s, r);
            } else {
                table.set(s, r, e);
            }
        }
    }

    let alive_all = table.compact();
    for (s, alive) in alive_all.iter().enumerate() {
        conformations[s].compact(alive);
    }

    table
}

/// Computes self-energies for every candidate in a single slot.
fn slot_self_energies(
    slot: &Residue,
    confs: &Conformations,
    fixed: &FixedAtoms<'_>,
    ff: &ForceFieldParams,
    coulomb_scale: Option<f32>,
) -> Vec<f32> {
    let n_r = confs.n_candidates();
    let n_a = confs.n_atoms();
    let types = slot.atom_types();
    let charges = slot.atom_charges();
    let donors = slot.donor_of_h();

    match coulomb_scale {
        None => (0..n_r)
            .map(|r| candidate_energy_no_coul(confs.coords_of(r), n_a, types, donors, fixed, ff))
            .collect(),
        Some(c_d) => (0..n_r)
            .map(|r| {
                candidate_energy_coul(
                    confs.coords_of(r),
                    n_a,
                    types,
                    charges,
                    donors,
                    fixed,
                    ff,
                    c_d,
                )
            })
            .collect(),
    }
}

/// Computes the non-bonded interaction energy of a candidate with all fixed atoms, excluding Coulomb interactions.
fn candidate_energy_no_coul(
    coords: &[Vec3],
    n_a: usize,
    types: &[TypeIdx],
    donors: &[u8],
    fixed: &FixedAtoms<'_>,
    ff: &ForceFieldParams,
) -> f32 {
    let mut energy = 0.0_f32;
    for a in 0..n_a {
        let pos_a = coords[a];
        let type_a = types[a];
        for (_, b_idx) in fixed.neighbors(pos_a, VDW_CUTOFF) {
            let b = b_idx as usize;
            let pos_b = fixed.positions[b];
            let type_b = fixed.types[b];
            let r_sq = pos_a.dist_sq(pos_b);
            if r_sq <= VDW_CUTOFF_SQ {
                energy += vdw_pair(&ff.vdw, type_a, type_b, r_sq);
            }
            if donors[a] != u8::MAX {
                energy += hbond_sc_donor(coords, types, donors[a], a, pos_b, type_b, &ff.hbond);
            }
            if fixed.donor_for_h[b] != NO_DONOR {
                energy += hbond_fixed_donor(fixed, b, pos_a, type_a, &ff.hbond);
            }
        }
    }
    energy
}

/// Computes the non-bonded interaction energy of a candidate with all fixed atoms, including Coulomb interactions.
fn candidate_energy_coul(
    coords: &[Vec3],
    n_a: usize,
    types: &[TypeIdx],
    charges: &[f32],
    donors: &[u8],
    fixed: &FixedAtoms<'_>,
    ff: &ForceFieldParams,
    c_d: f32,
) -> f32 {
    let mut energy = 0.0_f32;
    for a in 0..n_a {
        let pos_a = coords[a];
        let type_a = types[a];
        for (_, b_idx) in fixed.neighbors(pos_a, COULOMB_CUTOFF) {
            let b = b_idx as usize;
            let pos_b = fixed.positions[b];
            let type_b = fixed.types[b];
            let r_sq = pos_a.dist_sq(pos_b);
            if r_sq <= VDW_CUTOFF_SQ {
                energy += vdw_pair(&ff.vdw, type_a, type_b, r_sq);
            }
            if donors[a] != u8::MAX {
                energy += hbond_sc_donor(coords, types, donors[a], a, pos_b, type_b, &ff.hbond);
            }
            if fixed.donor_for_h[b] != NO_DONOR {
                energy += hbond_fixed_donor(fixed, b, pos_a, type_a, &ff.hbond);
            }
            if r_sq <= COULOMB_CUTOFF_SQ {
                energy += c_d * charges[a] * fixed.charges[b] / r_sq;
            }
        }
    }
    energy
}

/// VdW pair energy (LJ 12-6 or Buckingham, dispatched once per call).
fn vdw_pair(vdw: &VdwMatrix, ta: TypeIdx, tb: TypeIdx, r_sq: f32) -> f32 {
    match vdw {
        VdwMatrix::LennardJones(m) => {
            let p = m.get(ta, tb);
            LennardJones::energy(r_sq, (p.d0, p.r0_sq))
        }
        VdwMatrix::Buckingham(m) => {
            let p = m.get(ta, tb);
            Buckingham::energy(r_sq, (p.a, p.b, p.c, p.r_max_sq, p.two_e_max))
        }
    }
}

/// H-bond energy when sidechain atom `h_local` is a donor hydrogen.
fn hbond_sc_donor(
    coords: &[Vec3],
    types: &[TypeIdx],
    d_local: u8,
    h_local: usize,
    acc_pos: Vec3,
    acc_type: TypeIdx,
    hbond: &crate::model::system::HBondParams,
) -> f32 {
    let d = d_local as usize;
    let d_pos = coords[d];
    let h_pos = coords[h_local];
    let r_sq_da = d_pos.dist_sq(acc_pos);
    if r_sq_da > HBOND_CUTOFF_SQ {
        return 0.0;
    }
    if let Some((d_hb, r_hb_sq)) = hbond.get(types[d], types[h_local], acc_type) {
        let cos_theta = cos_dha(d_pos, h_pos, acc_pos);
        HydrogenBond::<HBOND_N>::energy(r_sq_da, cos_theta, (d_hb, r_hb_sq))
    } else {
        0.0
    }
}

/// H-bond energy when a fixed atom `b` is a donor hydrogen.
fn hbond_fixed_donor(
    fixed: &FixedAtoms<'_>,
    b: usize,
    acc_pos: Vec3,
    acc_type: TypeIdx,
    hbond: &crate::model::system::HBondParams,
) -> f32 {
    let d_idx = fixed.donor_for_h[b] as usize;
    let d_pos = fixed.positions[d_idx];
    let h_pos = fixed.positions[b];
    let r_sq_da = d_pos.dist_sq(acc_pos);
    if r_sq_da > HBOND_CUTOFF_SQ {
        return 0.0;
    }
    if let Some((d_hb, r_hb_sq)) = hbond.get(fixed.types[d_idx], fixed.types[b], acc_type) {
        let cos_theta = cos_dha(d_pos, h_pos, acc_pos);
        HydrogenBond::<HBOND_N>::energy(r_sq_da, cos_theta, (d_hb, r_hb_sq))
    } else {
        0.0
    }
}

/// Cosine of the D-H···A angle at the hydrogen.
fn cos_dha(d: Vec3, h: Vec3, a: Vec3) -> f32 {
    let dh = h - d;
    let ha = a - h;
    let denom_sq = dh.len_sq() * ha.len_sq();
    if denom_sq < 1e-16 {
        return 0.0;
    }
    dh.dot(ha) / denom_sq.sqrt()
}
