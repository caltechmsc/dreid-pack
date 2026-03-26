use crate::pack::constant::{
    COULOMB_CONST, COULOMB_CUTOFF, COULOMB_CUTOFF_SQ, HBOND_CUTOFF, HBOND_CUTOFF_SQ, HBOND_N,
    VDW_CUTOFF, VDW_CUTOFF_SQ,
};
use crate::{
    model::{
        system::{ForceFieldParams, HBondParams, Residue, VdwMatrix},
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
        for (_, b_idx) in fixed.neighbors(pos_a, VDW_CUTOFF.max(HBOND_CUTOFF)) {
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
        for (_, b_idx) in fixed.neighbors(pos_a, COULOMB_CUTOFF.max(VDW_CUTOFF).max(HBOND_CUTOFF)) {
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
            if r_sq <= COULOMB_CUTOFF_SQ {
                energy += c_d * atoms.charges[a] * fixed.charges[b] / r_sq;
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
    hbond: &HBondParams,
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
    hbond: &HBondParams,
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
        system::{
            BuckMatrix, BuckPair, FixedAtomPool, ForceFieldParams, HBondParams, LjMatrix, LjPair,
            SidechainAtoms,
        },
    };
    use crate::pack::constant::max_interaction_cutoff;
    use std::collections::{HashMap, HashSet};

    const ANCHOR: [Vec3; 3] = [
        Vec3::new(1.458, 0.0, 0.0),
        Vec3::zero(),
        Vec3::new(-0.524, 0.0, 1.454),
    ];

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn t(n: u8) -> TypeIdx {
        TypeIdx(n)
    }

    fn make_slot(n: usize) -> Residue {
        let coords = vec![Vec3::zero(); n];
        let types = vec![t(0); n];
        let charges = vec![0.0_f32; n];
        let donors = vec![u8::MAX; n];
        Residue::new(
            ResidueType::Ser,
            ANCHOR,
            -1.0,
            1.0,
            std::f32::consts::PI,
            SidechainAtoms {
                coords: &coords,
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

    fn make_confs(pos: Vec3, n_r: u16, n_a: u8) -> Conformations {
        Conformations::new(vec![pos; n_r as usize * n_a as usize], n_r, n_a)
    }

    fn ff_lj(d0: f32, r0_sq: f32) -> ForceFieldParams {
        let pair = LjPair { d0, r0_sq };
        ForceFieldParams {
            vdw: VdwMatrix::LennardJones(LjMatrix::new(1, vec![pair])),
            hbond: HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new()),
        }
    }

    fn ff_buck(p: BuckPair) -> ForceFieldParams {
        ForceFieldParams {
            vdw: VdwMatrix::Buckingham(BuckMatrix::new(1, vec![p])),
            hbond: HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new()),
        }
    }

    #[test]
    fn prune_empty_slots_returns_empty_table() {
        let pool = make_pool(vec![]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(0.0, 1.0);
        let table = prune(&[], &mut [], &fixed, &ff, None, f32::INFINITY);
        assert_eq!(table.n_slots(), 0);
    }

    #[test]
    fn prune_no_fixed_atoms_gives_zero_energy() {
        let pool = make_pool(vec![]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1.0, 4.0);
        let slot = make_slot(1);
        let mut confs = vec![make_confs(Vec3::zero(), 3, 1)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert_eq!(table.n_candidates(0), 3);
        for r in 0..3 {
            assert_eq!(table.get(0, r), 0.0);
        }
    }

    #[test]
    fn prune_lj_energy_equals_negative_well_depth_at_equilibrium() {
        let d0 = 2.0_f32;
        let r0 = 2.0_f32;
        let pool = make_pool(vec![v(r0, 0.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(d0, r0 * r0);
        let slot = make_slot(1);
        let mut confs = vec![make_confs(Vec3::zero(), 1, 1)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert!(
            (table.get(0, 0) - (-d0)).abs() < 1e-5,
            "expected -D0 = {}, got {}",
            -d0,
            table.get(0, 0)
        );
    }

    #[test]
    fn prune_buckingham_path_dispatches_to_different_energy() {
        let buck = BuckPair {
            a: 200.0,
            b: 2.0,
            c: 0.0,
            r_max_sq: 0.0,
            two_e_max: 0.0,
        };
        let pool = make_pool(vec![v(2.0, 0.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let slot = make_slot(1);
        let mut confs_b = vec![make_confs(Vec3::zero(), 1, 1)];
        let mut confs_l = vec![make_confs(Vec3::zero(), 1, 1)];
        let tb = prune(
            &[slot.clone()],
            &mut confs_b,
            &fixed,
            &ff_buck(buck),
            None,
            f32::INFINITY,
        );
        let tl = prune(
            &[slot],
            &mut confs_l,
            &fixed,
            &ff_lj(1.0, 4.0),
            None,
            f32::INFINITY,
        );
        assert!(tb.get(0, 0).is_finite(), "Buckingham energy must be finite");
        assert_ne!(
            tb.get(0, 0),
            tl.get(0, 0),
            "Buckingham and LJ must dispatch independently"
        );
    }

    #[test]
    fn prune_threshold_removes_high_energy_candidates() {
        let d0 = 1.0_f32;
        let r0 = 2.0_f32;
        let pool = make_pool(vec![v(r0, 0.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(d0, r0 * r0);
        let slot = make_slot(1);
        let mut confs = vec![Conformations::new(
            vec![Vec3::zero(), v(50.0, 0.0, 0.0)],
            2,
            1,
        )];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.5 * d0);
        assert_eq!(table.n_candidates(0), 1);
        assert_eq!(confs[0].n_candidates(), 1);
    }

    #[test]
    fn prune_guarantees_at_least_one_survivor() {
        let pool = make_pool(vec![v(2.0, 0.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1.0, 4.0);
        let slot = make_slot(1);
        let coords: Vec<Vec3> = (0..5).map(|i| v(0.1 * i as f32, 0.0, 0.0)).collect();
        let mut confs = vec![Conformations::new(coords, 5, 1)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.0);
        assert!(table.n_candidates(0) >= 1);
        assert!(confs[0].n_candidates() >= 1);
    }

    #[test]
    fn prune_conformations_stay_in_sync_after_pruning() {
        let pool = make_pool(vec![v(2.0, 0.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1.0, 4.0);
        let slot = make_slot(1);
        let coords = vec![
            v(0.1, 0.0, 0.0),
            v(0.2, 0.0, 0.0),
            v(50.0, 0.0, 0.0),
            v(60.0, 0.0, 0.0),
        ];
        let mut confs = vec![Conformations::new(coords, 4, 1)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, 0.01);
        assert_eq!(table.n_candidates(0), confs[0].n_candidates());
    }

    #[test]
    fn prune_coulomb_increases_energy_for_like_charges() {
        let mut pool = make_pool(vec![v(3.0, 0.0, 0.0)]);
        pool.charges[0] = 1.0;
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(Some(1.0)));
        let ff = ff_lj(0.0, 1.0);
        let sc_charges = [0.5_f32];
        let slot = Residue::new(
            ResidueType::Ser,
            ANCHOR,
            -1.0,
            1.0,
            std::f32::consts::PI,
            SidechainAtoms {
                coords: &[Vec3::zero()],
                types: &[t(0)],
                charges: &sc_charges,
                donor_of_h: &[u8::MAX],
            },
        )
        .unwrap();
        let mut confs_no = vec![make_confs(Vec3::zero(), 1, 1)];
        let mut confs_yes = vec![make_confs(Vec3::zero(), 1, 1)];
        let t_no = prune(
            &[slot.clone()],
            &mut confs_no,
            &fixed,
            &ff,
            None,
            f32::INFINITY,
        );
        let t_yes = prune(
            &[slot],
            &mut confs_yes,
            &fixed,
            &ff,
            Some(1.0),
            f32::INFINITY,
        );
        assert!(
            t_yes.get(0, 0) > t_no.get(0, 0),
            "Coulomb must increase energy for like charges"
        );
    }

    #[test]
    fn prune_hbond_sc_donor_contributes_negative_energy() {
        let (td, th, ta) = (t(0), t(1), t(2));
        let r_hb = 3.0_f32;
        let mut params = HashMap::new();
        params.insert((td, th, ta), (1.0_f32, r_hb * r_hb));
        let mut h_types = HashSet::new();
        h_types.insert(th);
        let mut acc_types = HashSet::new();
        acc_types.insert(ta);
        let pool = FixedAtomPool {
            positions: vec![v(r_hb, 0.0, 0.0)],
            types: vec![ta],
            charges: vec![0.0],
            donor_for_h: vec![NO_DONOR],
        };
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let zero_lj = LjPair {
            d0: 0.0,
            r0_sq: 1.0,
        };
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(LjMatrix::new(3, vec![zero_lj; 9])),
            hbond: HBondParams::new(h_types, acc_types, params),
        };
        let slot = Residue::new(
            ResidueType::Ser,
            ANCHOR,
            -1.0,
            1.0,
            std::f32::consts::PI,
            SidechainAtoms {
                coords: &[Vec3::zero(), v(1.0, 0.0, 0.0)],
                types: &[td, th],
                charges: &[0.0_f32; 2],
                donor_of_h: &[u8::MAX, 0u8],
            },
        )
        .unwrap();
        let mut confs = vec![Conformations::new(
            vec![Vec3::zero(), v(1.0, 0.0, 0.0)],
            1,
            2,
        )];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert!(
            (table.get(0, 0) - (-1.0)).abs() < 1e-5,
            "expected E = -D_hb = -1.0, got {}",
            table.get(0, 0)
        );
    }

    #[test]
    fn prune_hbond_fixed_donor_contributes_negative_energy() {
        let (td, th, ta) = (t(0), t(1), t(2));
        let r_hb = 3.0_f32;
        let mut params = HashMap::new();
        params.insert((td, th, ta), (1.0_f32, r_hb * r_hb));
        let mut h_types = HashSet::new();
        h_types.insert(th);
        let mut acc_types = HashSet::new();
        acc_types.insert(ta);
        let pool = FixedAtomPool {
            positions: vec![Vec3::zero(), v(1.0, 0.0, 0.0)],
            types: vec![td, th],
            charges: vec![0.0; 2],
            donor_for_h: vec![NO_DONOR, 0u32],
        };
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let zero_lj = LjPair {
            d0: 0.0,
            r0_sq: 1.0,
        };
        let ff = ForceFieldParams {
            vdw: VdwMatrix::LennardJones(LjMatrix::new(3, vec![zero_lj; 9])),
            hbond: HBondParams::new(h_types, acc_types, params),
        };
        let slot = Residue::new(
            ResidueType::Ser,
            ANCHOR,
            -1.0,
            1.0,
            std::f32::consts::PI,
            SidechainAtoms {
                coords: &[v(r_hb, 0.0, 0.0)],
                types: &[ta],
                charges: &[0.0_f32],
                donor_of_h: &[u8::MAX],
            },
        )
        .unwrap();
        let mut confs = vec![make_confs(v(r_hb, 0.0, 0.0), 1, 1)];
        let table = prune(&[slot], &mut confs, &fixed, &ff, None, f32::INFINITY);
        assert!(
            (table.get(0, 0) - (-1.0)).abs() < 1e-5,
            "expected E = -D_hb = -1.0, got {}",
            table.get(0, 0)
        );
    }

    #[test]
    fn prune_parallel_matches_sequential_per_slot() {
        let pool = make_pool(vec![v(2.0, 0.0, 0.0), v(0.0, 2.0, 0.0)]);
        let fixed = FixedAtoms::build(&pool, max_interaction_cutoff(None));
        let ff = ff_lj(1.0, 4.0);
        let positions = [Vec3::zero(), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0)];
        let slots: Vec<_> = positions.iter().map(|_| make_slot(1)).collect();
        let mut confs: Vec<_> = positions.iter().map(|&p| make_confs(p, 2, 1)).collect();
        let table = prune(&slots, &mut confs, &fixed, &ff, None, f32::INFINITY);
        for s in 0..3 {
            let mut c = vec![make_confs(positions[s], 2, 1)];
            let single = prune(&slots[s..s + 1], &mut c, &fixed, &ff, None, f32::INFINITY);
            for r in 0..single.n_candidates(0) {
                assert_eq!(
                    table.get(s, r),
                    single.get(0, r),
                    "slot {s} candidate {r}: parallel={} sequential={}",
                    table.get(s, r),
                    single.get(0, r)
                );
            }
        }
    }
}
