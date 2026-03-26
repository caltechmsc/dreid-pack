use crate::pack::constant::{
    COULOMB_CONST, COULOMB_CUTOFF, HBOND_CUTOFF_SQ, HBOND_N, VDW_CUTOFF, VDW_CUTOFF_SQ,
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

/// Per-slot atom properties shared across all candidates of the same residue.
struct SlotAtoms<'a> {
    n_a: usize,
    types: &'a [TypeIdx],
    charges: &'a [f32],
    donors: &'a [u8],
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
    let atoms = SlotAtoms {
        n_a: confs.n_atoms(),
        types: slot.atom_types(),
        charges: slot.atom_charges(),
        donors: slot.donor_of_h(),
    };

    match coulomb_scale {
        None => (0..n_r)
            .map(|r| candidate_energy_no_coul(confs.coords_of(r), &atoms, fixed, ff))
            .collect(),
        Some(c_d) => (0..n_r)
            .map(|r| candidate_energy_coul(confs.coords_of(r), &atoms, fixed, ff, c_d))
            .collect(),
    }
}

/// Computes the non-bonded interaction energy of a candidate with all fixed atoms, excluding Coulomb interactions.
fn candidate_energy_no_coul(
    coords: &[Vec3],
    atoms: &SlotAtoms<'_>,
    fixed: &FixedAtoms<'_>,
    ff: &ForceFieldParams,
) -> f32 {
    let mut energy = 0.0_f32;
    for a in 0..atoms.n_a {
        let pos_a = coords[a];
        let type_a = atoms.types[a];
        for (_, b_idx) in fixed.neighbors(pos_a, VDW_CUTOFF) {
            let b = b_idx as usize;
            let pos_b = fixed.positions[b];
            let type_b = fixed.types[b];
            let r_sq = pos_a.dist_sq(pos_b);
            energy += vdw_pair(&ff.vdw, type_a, type_b, r_sq);
            if atoms.donors[a] != u8::MAX {
                energy += hbond_sc_donor(
                    coords,
                    atoms.types,
                    atoms.donors[a],
                    a,
                    pos_b,
                    type_b,
                    &ff.hbond,
                );
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
    atoms: &SlotAtoms<'_>,
    fixed: &FixedAtoms<'_>,
    ff: &ForceFieldParams,
    c_d: f32,
) -> f32 {
    let mut energy = 0.0_f32;
    for a in 0..atoms.n_a {
        let pos_a = coords[a];
        let type_a = atoms.types[a];
        for (_, b_idx) in fixed.neighbors(pos_a, COULOMB_CUTOFF) {
            let b = b_idx as usize;
            let pos_b = fixed.positions[b];
            let type_b = fixed.types[b];
            let r_sq = pos_a.dist_sq(pos_b);
            if r_sq <= VDW_CUTOFF_SQ {
                energy += vdw_pair(&ff.vdw, type_a, type_b, r_sq);
            }
            if atoms.donors[a] != u8::MAX {
                energy += hbond_sc_donor(
                    coords,
                    atoms.types,
                    atoms.donors[a],
                    a,
                    pos_b,
                    type_b,
                    &ff.hbond,
                );
            }
            if fixed.donor_for_h[b] != NO_DONOR {
                energy += hbond_fixed_donor(fixed, b, pos_a, type_a, &ff.hbond);
            }
            energy += c_d * atoms.charges[a] * fixed.charges[b] / r_sq;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        residue::ResidueType,
        system::{FixedAtomPool, ForceFieldParams, HBondParams, LjMatrix, LjPair, SidechainAtoms},
        types::TypeIdx,
    };
    use crate::pack::constant::max_interaction_cutoff;
    use std::collections::{HashMap, HashSet};

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn t(n: u8) -> TypeIdx {
        TypeIdx(n)
    }

    fn lj_uniform(n: usize, d0: f32, r0_sq: f32) -> LjMatrix {
        let pair = LjPair { d0, r0_sq };
        LjMatrix::new(n, vec![pair; n * n])
    }

    fn empty_hbond() -> HBondParams {
        HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new())
    }

    fn ff_lj(n: usize, d0: f32, r0_sq: f32) -> ForceFieldParams {
        ForceFieldParams {
            vdw: VdwMatrix::LennardJones(lj_uniform(n, d0, r0_sq)),
            hbond: empty_hbond(),
        }
    }

    fn make_slot(rt: ResidueType, sidechain: &[Vec3]) -> Residue {
        let n = sidechain.len();
        let anchor = [v(1.458, 0.0, 0.0), v(0.0, 0.0, 0.0), v(-0.524, 0.0, 1.454)];
        let types = vec![t(0); n];
        let charges = vec![0.0_f32; n];
        let donors = vec![u8::MAX; n];
        Residue::new(
            rt,
            anchor,
            -1.0,
            1.0,
            std::f32::consts::PI,
            SidechainAtoms {
                coords: sidechain,
                types: &types,
                charges: &charges,
                donor_of_h: &donors,
            },
        )
        .unwrap()
    }

    fn make_pool(positions: Vec<Vec3>) -> FixedAtomPool {
        let n = positions.len();
        FixedAtomPool {
            positions,
            types: vec![t(0); n],
            charges: vec![0.0; n],
            donor_for_h: vec![NO_DONOR; n],
        }
    }

    fn make_conformations(coords: Vec<Vec3>, n_candidates: u16, n_atoms: u8) -> Conformations {
        Conformations::new(coords, n_candidates, n_atoms)
    }

    #[test]
    fn prune_empty_slots() {
        let pool = make_pool(vec![]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1, 0.0, 1.0);
        let table = prune(&[], &mut [], &fixed, &ff, None, f32::INFINITY);
        assert_eq!(table.n_slots(), 0);
    }

    #[test]
    fn prune_no_fixed_atoms_all_zero() {
        let pool = make_pool(vec![]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1, 0.1, 4.0);

        let sc = [v(2.0, 0.0, 0.0); 5];
        let slot = make_slot(ResidueType::Ser, &sc);

        let mut confs = vec![make_conformations(vec![v(2.0, 0.0, 0.0); 15], 3, 5)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert_eq!(table.n_slots(), 1);
        assert_eq!(table.n_candidates(0), 3);
        for r in 0..3 {
            assert_eq!(table.get(0, r), 0.0);
        }
    }

    #[test]
    fn prune_threshold_removes_high_energy() {
        let pool = make_pool(vec![v(1.0, 0.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1, 0.1, 9.0);

        let sc = [v(0.0, 0.0, 0.0); 5];
        let slot = make_slot(ResidueType::Ser, &sc);

        let mut data = Vec::new();
        for _ in 0..5 {
            data.push(v(0.5, 0.0, 0.0));
        }
        for _ in 0..5 {
            data.push(v(20.0, 0.0, 0.0));
        }
        for _ in 0..5 {
            data.push(v(20.0, 10.0, 0.0));
        }

        let mut confs = vec![make_conformations(data, 3, 5)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.01);

        assert_eq!(table.n_candidates(0), 2);
        assert_eq!(confs[0].n_candidates(), 2);
    }

    #[test]
    fn prune_guarantees_at_least_one_survivor() {
        let pool = make_pool(vec![v(0.0, 0.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1, 10.0, 9.0);

        let sc = [v(1.5, 0.0, 0.0); 5];
        let slot = make_slot(ResidueType::Ser, &sc);

        let data: Vec<Vec3> = (0..10)
            .map(|i| v(1.5 + 0.01 * i as f32, 0.0, 0.0))
            .collect();
        let mut confs = vec![make_conformations(data, 2, 5)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.0);

        assert!(table.n_candidates(0) >= 1);
        assert!(confs[0].n_candidates() >= 1);
    }

    #[test]
    fn prune_conformations_stay_in_sync() {
        let pool = make_pool(vec![v(1.0, 0.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1, 0.1, 9.0);

        let sc = [v(0.0, 0.0, 0.0); 5];
        let slot = make_slot(ResidueType::Ser, &sc);

        let mut data = Vec::new();
        data.extend(std::iter::repeat(v(0.5, 0.0, 0.0)).take(5));
        data.extend(std::iter::repeat(v(50.0, 0.0, 0.0)).take(5));
        data.extend(std::iter::repeat(v(0.3, 0.0, 0.0)).take(5));
        data.extend(std::iter::repeat(v(50.0, 5.0, 0.0)).take(5));

        let mut confs = vec![make_conformations(data, 4, 5)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.01);

        assert_eq!(table.n_candidates(0), confs[0].n_candidates());
    }

    #[test]
    fn prune_coulomb_adds_energy() {
        let mut pool = make_pool(vec![v(3.0, 0.0, 0.0)]);
        pool.charges[0] = 1.0;
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(Some(1.0)));
        let ff = ff_lj(1, 0.0, 1.0);

        let sc = [v(0.0, 0.0, 0.0); 5];
        let slot = {
            let n = sc.len();
            let anchor = [v(1.458, 0.0, 0.0), v(0.0, 0.0, 0.0), v(-0.524, 0.0, 1.454)];
            let types = vec![t(0); n];
            let mut charges = vec![0.0_f32; n];
            charges[0] = 0.5;
            let donors = vec![u8::MAX; n];
            Residue::new(
                ResidueType::Ser,
                anchor,
                -1.0,
                1.0,
                std::f32::consts::PI,
                SidechainAtoms {
                    coords: &sc,
                    types: &types,
                    charges: &charges,
                    donor_of_h: &donors,
                },
            )
            .unwrap()
        };

        let data = vec![
            v(4.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
        ];
        let mut confs = vec![make_conformations(data, 2, 5)];

        let data_no = vec![
            v(4.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
        ];
        let mut confs_no = vec![make_conformations(data_no, 2, 5)];
        let table_no = prune(
            &[slot.clone()],
            &mut confs_no,
            &fixed,
            &ff,
            None,
            f32::INFINITY,
        );
        let table_yes = prune(&[slot], &mut confs, &fixed, &ff, Some(1.0), f32::INFINITY);

        assert_ne!(table_yes.get(0, 0), table_no.get(0, 0));
    }
}
