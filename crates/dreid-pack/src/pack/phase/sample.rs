use crate::{
    model::{residue::ResidueType, system::Residue, types::Vec3},
    pack::{
        constant::{ROTAMER_BIAS_CAP, rotamer_weight},
        model::{conformation::Conformations, energy::RotamerBias},
    },
};
use arrayvec::ArrayVec;
use rayon::prelude::*;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_3};

/// Generates candidate [`Conformations`] and a [`RotamerBias`] table for
/// every packable slot.
pub fn sample(
    slots: &[Residue],
    prob_cutoff: f32,
    sample_polar_h: bool,
    include_input: bool,
) -> (Vec<Conformations>, RotamerBias) {
    let (confs, bias_vecs): (Vec<_>, Vec<_>) = slots
        .par_iter()
        .map(|slot| sample_one(slot, prob_cutoff, sample_polar_h, include_input))
        .unzip();
    (confs, RotamerBias::new(bias_vecs))
}

/// Converts a [`Vec3`] to the `rotamer` crate's [`Vec3`](rotamer::Vec3).
fn to_rvec(v: Vec3) -> rotamer::Vec3 {
    rotamer::Vec3::new(v.x, v.y, v.z)
}

/// Reinterprets a [`rotamer::Vec3`] slice as a [`Vec3`] slice.
fn cast_coords(src: &[rotamer::Vec3]) -> &[Vec3] {
    const _: () = assert!(
        size_of::<rotamer::Vec3>() == size_of::<Vec3>()
            && align_of::<rotamer::Vec3>() == align_of::<Vec3>()
    );
    // SAFETY: both types are `#[repr(C)]` with layout `{ x: f32, y: f32, z: f32 }`.
    unsafe { std::slice::from_raw_parts(src.as_ptr().cast(), src.len()) }
}

/// Returns polar-hydrogen torsion angles to sample.
fn polar_h_angles(rt: ResidueType, sample: bool) -> ArrayVec<f32, 6> {
    let period = rt.polar_h_period();
    if !sample || period == 0.0 {
        let mut v = ArrayVec::new();
        v.push(0.0);
        return v;
    }
    let n = (period / FRAC_PI_3).round() as usize;
    (0..n).map(|i| i as f32 * period / n as f32).collect()
}

/// Converts a raw rotamer probability into an additive self-energy bias.
fn bias_energy(prob: f32, max_prob: f32) -> f32 {
    (-(prob / max_prob).ln()).clamp(0.0, ROTAMER_BIAS_CAP)
}

/// Expands Dunbrack rotamers into coordinate data.
macro_rules! expand {
    ($slot:expr, $cutoff:expr, $data:expr, $erot:expr, $count:expr, $D:ty, $R:ty) => {{
        use rotamer::SidechainCoords as _;

        let [an, aca, ac] = *$slot.anchor();
        let (n, ca, c) = (to_rvec(an), to_rvec(aca), to_rvec(ac));
        let mut rots = ArrayVec::<_, { <$D as dunbrack::Residue>::N_ROTAMERS }>::new();
        rots.extend(<$D as dunbrack::Residue>::rotamers(
            $slot.phi().to_degrees(),
            $slot.psi().to_degrees(),
        ));
        let (best, max_prob) = rots
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.prob.total_cmp(&b.prob))
            .map_or((0, 1.0), |(i, r)| (i, r.prob));
        $data.reserve(
            rots.len() * <<$R as rotamer::Sidechain>::Coords as rotamer::SidechainCoords>::N,
        );
        for (i, rot) in rots.iter().enumerate() {
            if rot.prob < $cutoff && i != best {
                continue;
            }
            let chi = core::array::from_fn(|j| rot.chi_mean[j].to_radians());
            let coords = <$R>::build(n, ca, c, chi);
            $data.extend_from_slice(cast_coords(coords.as_slice()));
            $erot.push(bias_energy(rot.prob, max_prob));
            $count += 1;
        }
    }};

    ($slot:expr, $cutoff:expr, $data:expr, $erot:expr, $count:expr, $D:ty, $R:ty, $ph:expr) => {{
        use rotamer::SidechainCoords as _;

        let [an, aca, ac] = *$slot.anchor();
        let (n, ca, c) = (to_rvec(an), to_rvec(aca), to_rvec(ac));
        let mut rots = ArrayVec::<_, { <$D as dunbrack::Residue>::N_ROTAMERS }>::new();
        rots.extend(<$D as dunbrack::Residue>::rotamers(
            $slot.phi().to_degrees(),
            $slot.psi().to_degrees(),
        ));
        let (best, max_prob) = rots
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.prob.total_cmp(&b.prob))
            .map_or((0, 1.0), |(i, r)| (i, r.prob));
        $data.reserve(
            rots.len()
                * $ph.len()
                * <<$R as rotamer::Sidechain>::Coords as rotamer::SidechainCoords>::N,
        );
        for (i, rot) in rots.iter().enumerate() {
            if rot.prob < $cutoff && i != best {
                continue;
            }
            let chi = core::array::from_fn(|j| rot.chi_mean[j].to_radians());
            let e = bias_energy(rot.prob, max_prob);
            for &tau in $ph.as_slice() {
                let coords = <$R>::build(n, ca, c, chi, [tau]);
                $data.extend_from_slice(cast_coords(coords.as_slice()));
                $erot.push(e);
                $count += 1;
            }
        }
    }};
}

/// Samples candidate conformations for a single slot.
fn sample_one(
    slot: &Residue,
    prob_cutoff: f32,
    sample_polar_h: bool,
    include_input: bool,
) -> (Conformations, Vec<f32>) {
    debug_assert!(slot.res_type().is_packable());

    let rt = slot.res_type();
    let n_atoms = rt.n_atoms();
    let mut data: Vec<Vec3> = Vec::new();
    let mut erot: Vec<f32> = Vec::new();
    let mut n_candidates: u16 = 0;
    let ph = polar_h_angles(rt, sample_polar_h);

    use ResidueType as T;
    match rt {
        T::Val => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Val,
            rotamer::Val
        ),
        T::Cym => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Cyh,
            rotamer::Cym
        ),
        T::Cys => {
            expand!(
                slot,
                prob_cutoff,
                data,
                erot,
                n_candidates,
                dunbrack::Cyh,
                rotamer::Cys,
                ph
            )
        }
        T::Ser => {
            expand!(
                slot,
                prob_cutoff,
                data,
                erot,
                n_candidates,
                dunbrack::Ser,
                rotamer::Ser,
                ph
            )
        }
        T::Thr => {
            expand!(
                slot,
                prob_cutoff,
                data,
                erot,
                n_candidates,
                dunbrack::Thr,
                rotamer::Thr,
                ph
            )
        }
        T::Pro => {
            if slot.omega().abs() < FRAC_PI_2 {
                expand!(
                    slot,
                    prob_cutoff,
                    data,
                    erot,
                    n_candidates,
                    dunbrack::Cpr,
                    rotamer::Pro
                )
            } else {
                expand!(
                    slot,
                    prob_cutoff,
                    data,
                    erot,
                    n_candidates,
                    dunbrack::Tpr,
                    rotamer::Pro
                )
            }
        }
        T::Asp => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Asp,
            rotamer::Asp
        ),
        T::Asn => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Asn,
            rotamer::Asn
        ),
        T::Ile => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Ile,
            rotamer::Ile
        ),
        T::Leu => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Leu,
            rotamer::Leu
        ),
        T::Phe => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Phe,
            rotamer::Phe
        ),
        T::Tym => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Tyr,
            rotamer::Tym
        ),
        T::Hid => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::His,
            rotamer::Hid
        ),
        T::Hie => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::His,
            rotamer::Hie
        ),
        T::Hip => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::His,
            rotamer::Hip
        ),
        T::Trp => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Trp,
            rotamer::Trp
        ),
        T::Ash => {
            expand!(
                slot,
                prob_cutoff,
                data,
                erot,
                n_candidates,
                dunbrack::Asp,
                rotamer::Ash,
                ph
            )
        }
        T::Tyr => {
            expand!(
                slot,
                prob_cutoff,
                data,
                erot,
                n_candidates,
                dunbrack::Tyr,
                rotamer::Tyr,
                ph
            )
        }
        T::Met => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Met,
            rotamer::Met
        ),
        T::Glu => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Glu,
            rotamer::Glu
        ),
        T::Gln => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Gln,
            rotamer::Gln
        ),
        T::Glh => {
            expand!(
                slot,
                prob_cutoff,
                data,
                erot,
                n_candidates,
                dunbrack::Glu,
                rotamer::Glh,
                ph
            )
        }
        T::Arg => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Arg,
            rotamer::Arg
        ),
        T::Arn => expand!(
            slot,
            prob_cutoff,
            data,
            erot,
            n_candidates,
            dunbrack::Arg,
            rotamer::Arn
        ),
        T::Lyn => {
            expand!(
                slot,
                prob_cutoff,
                data,
                erot,
                n_candidates,
                dunbrack::Lys,
                rotamer::Lyn,
                ph
            )
        }
        T::Lys => {
            expand!(
                slot,
                prob_cutoff,
                data,
                erot,
                n_candidates,
                dunbrack::Lys,
                rotamer::Lys,
                ph
            )
        }
        T::Gly | T::Ala | T::Cyx => unreachable!(),
    }

    if include_input {
        debug_assert_eq!(slot.sidechain().len(), n_atoms as usize);
        data.extend_from_slice(slot.sidechain());
        erot.push(0.0);
        n_candidates += 1;
    }

    data.shrink_to_fit();

    let w = rotamer_weight(rt);
    for e in &mut erot {
        *e *= w;
    }

    (Conformations::new(data, n_candidates, n_atoms), erot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{system::SidechainAtoms, types::TypeIdx};

    const ANCHOR: [Vec3; 3] = [
        Vec3::new(1.458, 0.0, 0.0),
        Vec3::zero(),
        Vec3::new(-0.524, 0.0, 1.454),
    ];
    const HELIX_PHI: f32 = -1.047;
    const HELIX_PSI: f32 = -0.698;

    fn make_slot(rt: ResidueType) -> Residue {
        make_slot_omega(rt, std::f32::consts::PI)
    }

    fn make_slot_omega(rt: ResidueType, omega: f32) -> Residue {
        let n = rt.n_atoms() as usize;
        let coords = vec![Vec3::splat(1.0); n];
        let types = vec![TypeIdx(0); n];
        let charges = vec![0.0f32; n];
        let donor_of_h = vec![u8::MAX; n];
        Residue::new(
            rt,
            ANCHOR,
            HELIX_PHI,
            HELIX_PSI,
            omega,
            SidechainAtoms {
                coords: &coords,
                types: &types,
                charges: &charges,
                donor_of_h: &donor_of_h,
            },
        )
        .unwrap()
    }

    #[test]
    fn bias_energy_at_max_prob_is_zero() {
        assert_eq!(bias_energy(1.0, 1.0), 0.0);
    }

    #[test]
    fn bias_energy_below_max_is_positive() {
        assert!(bias_energy(0.5, 1.0) > 0.0);
    }

    #[test]
    fn bias_energy_near_zero_prob_is_capped() {
        assert_eq!(bias_energy(1e-10, 1.0), ROTAMER_BIAS_CAP);
    }

    #[test]
    fn bias_energy_prob_above_max_clamps_to_zero() {
        assert_eq!(bias_energy(1.001, 1.0), 0.0);
    }

    #[test]
    fn polar_h_angles_disabled_yields_single_zero() {
        let ph = polar_h_angles(ResidueType::Ser, false);
        assert_eq!(ph.len(), 1);
        assert_eq!(ph[0], 0.0);
    }

    #[test]
    fn polar_h_angles_nonpolar_residue_yields_single_zero() {
        let ph = polar_h_angles(ResidueType::Val, true);
        assert_eq!(ph.len(), 1);
        assert_eq!(ph[0], 0.0);
    }

    #[test]
    fn polar_h_angles_full_period_yields_six_angles() {
        let ph = polar_h_angles(ResidueType::Ser, true);
        assert_eq!(ph.len(), 6);
    }

    #[test]
    fn polar_h_angles_half_period_yields_three_angles() {
        let ph = polar_h_angles(ResidueType::Tyr, true);
        assert_eq!(ph.len(), 3);
    }

    #[test]
    fn polar_h_angles_third_period_yields_two_angles() {
        let ph = polar_h_angles(ResidueType::Lys, true);
        assert_eq!(ph.len(), 2);
    }

    #[test]
    fn val_yields_three_candidates() {
        let slot = make_slot(ResidueType::Val);
        let (c, _) = sample_one(&slot, 0.0, false, false);
        assert_eq!(c.n_candidates(), 3);
    }

    #[test]
    fn val_candidate_atom_width_matches_residue_type() {
        let slot = make_slot(ResidueType::Val);
        let (c, _) = sample_one(&slot, 0.0, false, false);
        assert_eq!(c.coords_of(0).len(), ResidueType::Val.n_atoms() as usize);
    }

    #[test]
    fn prob_cutoff_reduces_candidates() {
        let slot = make_slot(ResidueType::Val);
        let (c, _) = sample_one(&slot, 0.1, false, false);
        assert!(c.n_candidates() >= 1 && c.n_candidates() <= 3);
    }

    #[test]
    fn prob_cutoff_always_keeps_one_survivor() {
        let slot = make_slot(ResidueType::Val);
        let (c, _) = sample_one(&slot, 1.0, false, false);
        assert_eq!(c.n_candidates(), 1);
    }

    #[test]
    fn ser_polar_h_expands_by_period() {
        let slot = make_slot(ResidueType::Ser);
        let (c, _) = sample_one(&slot, 0.0, true, false);
        assert_eq!(c.n_candidates(), 3 * 6);
    }

    #[test]
    fn ser_polar_h_disabled_yields_base_count() {
        let slot = make_slot(ResidueType::Ser);
        let (c, _) = sample_one(&slot, 0.0, false, false);
        assert_eq!(c.n_candidates(), 3);
    }

    #[test]
    fn tyr_polar_h_period_is_half() {
        let slot = make_slot(ResidueType::Tyr);
        let (c, _) = sample_one(&slot, 0.0, true, false);
        assert_eq!(c.n_candidates(), 18 * 3);
    }

    #[test]
    fn lys_polar_h_period_is_third() {
        let slot = make_slot(ResidueType::Lys);
        let (c, _) = sample_one(&slot, 0.0, true, false);
        assert_eq!(c.n_candidates(), 73 * 2);
    }

    #[test]
    fn include_input_appends_one_candidate() {
        let slot = make_slot(ResidueType::Val);
        let (without, _) = sample_one(&slot, 0.0, false, false);
        let (with_input, _) = sample_one(&slot, 0.0, false, true);
        assert_eq!(with_input.n_candidates(), without.n_candidates() + 1);
    }

    #[test]
    fn include_input_preserves_sidechain_coords() {
        let slot = make_slot(ResidueType::Val);
        let (c, _) = sample_one(&slot, 0.0, false, true);
        let last = c.coords_of(c.n_candidates() - 1);
        assert_eq!(last, slot.sidechain());
    }

    #[test]
    fn pro_trans_uses_trans_library() {
        let slot = make_slot(ResidueType::Pro);
        let (c, _) = sample_one(&slot, 0.0, false, false);
        assert_eq!(c.n_candidates(), 2);
    }

    #[test]
    fn pro_cis_uses_cis_library() {
        let slot = make_slot_omega(ResidueType::Pro, 0.0);
        let (c, _) = sample_one(&slot, 0.0, false, false);
        assert_eq!(c.n_candidates(), 2);
    }

    #[test]
    fn all_candidate_coords_finite() {
        let slot = make_slot(ResidueType::Arg);
        let (c, _) = sample_one(&slot, 0.0, false, false);
        for i in 0..c.n_candidates() {
            for v in c.coords_of(i) {
                assert!(
                    v.x.is_finite() && v.y.is_finite() && v.z.is_finite(),
                    "non-finite coordinate in candidate {i}: {v:?}"
                );
            }
        }
    }

    #[test]
    fn bias_vec_length_matches_candidate_count() {
        let slot = make_slot(ResidueType::Val);
        let (c, erot) = sample_one(&slot, 0.0, false, false);
        assert_eq!(erot.len(), c.n_candidates());
    }

    #[test]
    fn bias_vec_entries_all_nonnegative() {
        let slot = make_slot(ResidueType::Val);
        let (_, erot) = sample_one(&slot, 0.0, false, false);
        assert!(erot.iter().all(|&e| e >= 0.0));
    }

    #[test]
    fn best_rotamer_has_zero_bias() {
        let slot = make_slot(ResidueType::Val);
        let (_, erot) = sample_one(&slot, 1.0, false, false);
        assert_eq!(erot.len(), 1);
        assert_eq!(erot[0], 0.0);
    }

    #[test]
    fn sample_empty_slots_yields_empty_output() {
        let (confs, bias) = sample(&[], 0.0, false, false);
        assert!(confs.is_empty());
        assert_eq!(bias.n_slots(), 0);
    }

    #[test]
    fn sample_slot_count_matches_input_length() {
        let slots = [make_slot(ResidueType::Val), make_slot(ResidueType::Ser)];
        let (confs, bias) = sample(&slots, 0.0, false, false);
        assert_eq!(confs.len(), slots.len());
        assert_eq!(bias.n_slots(), slots.len());
    }

    #[test]
    fn sample_parallel_matches_sequential() {
        let slots = [
            make_slot(ResidueType::Val),
            make_slot(ResidueType::Ser),
            make_slot(ResidueType::Pro),
        ];
        let (par, _) = sample(&slots, 0.0, true, false);
        let seq: Vec<_> = slots
            .iter()
            .map(|s| sample_one(s, 0.0, true, false).0)
            .collect();
        assert_eq!(par.len(), seq.len());
        for (p, s) in par.iter().zip(seq.iter()) {
            assert_eq!(p.n_candidates(), s.n_candidates());
            assert_eq!(p.coords_of(0).len(), s.coords_of(0).len());
        }
    }
}
