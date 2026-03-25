use crate::{
    model::{residue::ResidueType, system::Residue, types::Vec3},
    pack::model::conformation::Conformations,
};
use arrayvec::ArrayVec;
use rayon::prelude::*;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_3};

/// Generates candidate [`Conformations`] for every packable slot.
pub fn sample(
    slots: &[Residue],
    prob_cutoff: f32,
    sample_polar_h: bool,
    include_input: bool,
) -> Vec<Conformations> {
    slots
        .par_iter()
        .map(|slot| sample_one(slot, prob_cutoff, sample_polar_h, include_input))
        .collect()
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

/// Expands Dunbrack rotamers into coordinate data.
macro_rules! expand {
    ($slot:expr, $cutoff:expr, $data:expr, $count:expr, $D:ty, $R:ty) => {{
        use rotamer::SidechainCoords as _;

        let [an, aca, ac] = *$slot.anchor();
        let (n, ca, c) = (to_rvec(an), to_rvec(aca), to_rvec(ac));
        let mut rots = ArrayVec::<_, { <$D as dunbrack::Residue>::N_ROTAMERS }>::new();
        rots.extend(<$D as dunbrack::Residue>::rotamers(
            $slot.phi().to_degrees(),
            $slot.psi().to_degrees(),
        ));
        let best = rots
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.prob.total_cmp(&b.prob))
            .map_or(0, |(i, _)| i);
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
            $count += 1;
        }
    }};

    ($slot:expr, $cutoff:expr, $data:expr, $count:expr, $D:ty, $R:ty, $ph:expr) => {{
        use rotamer::SidechainCoords as _;

        let [an, aca, ac] = *$slot.anchor();
        let (n, ca, c) = (to_rvec(an), to_rvec(aca), to_rvec(ac));
        let mut rots = ArrayVec::<_, { <$D as dunbrack::Residue>::N_ROTAMERS }>::new();
        rots.extend(<$D as dunbrack::Residue>::rotamers(
            $slot.phi().to_degrees(),
            $slot.psi().to_degrees(),
        ));
        let best = rots
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.prob.total_cmp(&b.prob))
            .map_or(0, |(i, _)| i);
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
            for &tau in $ph.as_slice() {
                let coords = <$R>::build(n, ca, c, chi, [tau]);
                $data.extend_from_slice(cast_coords(coords.as_slice()));
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
) -> Conformations {
    debug_assert!(slot.res_type().is_packable());

    let rt = slot.res_type();
    let n_atoms = rt.n_atoms();
    let mut data: Vec<Vec3> = Vec::new();
    let mut n_candidates: u16 = 0;
    let ph = polar_h_angles(rt, sample_polar_h);

    use ResidueType as T;
    match rt {
        T::Val => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Val,
            rotamer::Val
        ),
        T::Cym => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Cyh,
            rotamer::Cym
        ),
        T::Cyx => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Cyd,
            rotamer::Cyx
        ),
        T::Cys => {
            expand!(
                slot,
                prob_cutoff,
                data,
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
                    n_candidates,
                    dunbrack::Cpr,
                    rotamer::Pro
                )
            } else {
                expand!(
                    slot,
                    prob_cutoff,
                    data,
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
            n_candidates,
            dunbrack::Asp,
            rotamer::Asp
        ),
        T::Asn => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Asn,
            rotamer::Asn
        ),
        T::Ile => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Ile,
            rotamer::Ile
        ),
        T::Leu => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Leu,
            rotamer::Leu
        ),
        T::Phe => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Phe,
            rotamer::Phe
        ),
        T::Tym => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Tyr,
            rotamer::Tym
        ),
        T::Hid => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::His,
            rotamer::Hid
        ),
        T::Hie => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::His,
            rotamer::Hie
        ),
        T::Hip => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::His,
            rotamer::Hip
        ),
        T::Trp => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Trp,
            rotamer::Trp
        ),
        T::Ash => {
            expand!(
                slot,
                prob_cutoff,
                data,
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
            n_candidates,
            dunbrack::Met,
            rotamer::Met
        ),
        T::Glu => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Glu,
            rotamer::Glu
        ),
        T::Gln => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Gln,
            rotamer::Gln
        ),
        T::Glh => {
            expand!(
                slot,
                prob_cutoff,
                data,
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
            n_candidates,
            dunbrack::Arg,
            rotamer::Arg
        ),
        T::Arn => expand!(
            slot,
            prob_cutoff,
            data,
            n_candidates,
            dunbrack::Arg,
            rotamer::Arn
        ),
        T::Lyn => {
            expand!(
                slot,
                prob_cutoff,
                data,
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
                n_candidates,
                dunbrack::Lys,
                rotamer::Lys,
                ph
            )
        }
        T::Gly | T::Ala => unreachable!(),
    }

    if include_input {
        debug_assert_eq!(slot.sidechain().len(), n_atoms as usize);
        data.extend_from_slice(slot.sidechain());
        n_candidates += 1;
    }

    Conformations::new(data, n_candidates, n_atoms)
}
