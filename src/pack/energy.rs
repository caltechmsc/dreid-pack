use crate::model::system::{BuckMatrix, HBondParams, LjMatrix};
use crate::model::types::{TypeIdx, Vec3};
use crate::pack::constant::HBOND_N;
use dreid_kernel::{
    HybridKernel, PairKernel,
    potentials::nonbonded::{Buckingham, HydrogenBond, LennardJones},
};

/// Zero-cost VdW pair energy dispatch.
pub trait VdwKernel {
    /// Returns the VdW pair energy at squared distance `r_sq`.
    fn energy(&self, ta: TypeIdx, tb: TypeIdx, r_sq: f32) -> f32;
}

/// [`VdwKernel`] backed by a Lennard-Jones 12-6 parameter matrix.
pub struct LjKernel<'a>(pub &'a LjMatrix);

/// [`VdwKernel`] backed by a Buckingham (EXP-6) parameter matrix.
pub struct BuckKernel<'a>(pub &'a BuckMatrix);

impl VdwKernel for LjKernel<'_> {
    fn energy(&self, ta: TypeIdx, tb: TypeIdx, r_sq: f32) -> f32 {
        let p = self.0.get(ta, tb);
        LennardJones::energy(r_sq, (p.d0, p.r0_sq))
    }
}

impl VdwKernel for BuckKernel<'_> {
    fn energy(&self, ta: TypeIdx, tb: TypeIdx, r_sq: f32) -> f32 {
        let p = self.0.get(ta, tb);
        Buckingham::energy(r_sq, (p.a, p.b, p.c, p.r_max_sq, p.two_e_max))
    }
}

/// Returns the cosine of the D–H···A angle measured at H.
pub fn cos_dha(d: Vec3, h: Vec3, a: Vec3) -> f32 {
    let dh = h - d;
    let ha = a - h;
    let denom_sq = dh.len_sq() * ha.len_sq();
    if denom_sq < 1e-16 {
        return 0.0;
    }
    dh.dot(ha) / denom_sq.sqrt()
}

/// Returns the H-bond energy for the D–H···A triple.
pub fn hbond_energy(
    r_sq_da: f32,
    cos_theta: f32,
    d_type: TypeIdx,
    h_type: TypeIdx,
    a_type: TypeIdx,
    hbond: &HBondParams,
) -> f32 {
    match hbond.get(d_type, h_type, a_type) {
        Some((d_hb, r_hb_sq)) => {
            HydrogenBond::<HBOND_N>::energy(r_sq_da, cos_theta, (d_hb, r_hb_sq))
        }
        None => 0.0,
    }
}

/// Returns the Coulomb energy under distance-dependent dielectric ε(r) = D·r.
pub fn coulomb_energy<const COUL: bool>(c_d: f32, qi: f32, qj: f32, r_sq: f32) -> f32 {
    if COUL { c_d * qi * qj / r_sq } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::system::{BuckPair, LjPair};
    use approx::assert_abs_diff_eq;
    use std::collections::{HashMap, HashSet};

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn t(n: u8) -> TypeIdx {
        TypeIdx(n)
    }

    fn hbond_params(td: TypeIdx, th: TypeIdx, ta: TypeIdx, d_hb: f32, r_hb_sq: f32) -> HBondParams {
        let mut h_types = HashSet::new();
        h_types.insert(th);
        let mut acc_types = HashSet::new();
        acc_types.insert(ta);
        let mut params = HashMap::new();
        params.insert((td, th, ta), (d_hb, r_hb_sq));
        HBondParams::new(h_types, acc_types, params)
    }

    #[test]
    fn cos_dha_collinear_geometry_returns_one() {
        let c = cos_dha(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(3.0, 0.0, 0.0));
        assert_abs_diff_eq!(c, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn cos_dha_right_angle_at_h_returns_zero() {
        let c = cos_dha(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(1.0, 1.0, 0.0));
        assert_abs_diff_eq!(c, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn cos_dha_obtuse_angle_is_negative() {
        let c = cos_dha(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.5, 0.0, 0.0));
        assert!(c < 0.0, "got {c}");
    }

    #[test]
    fn cos_dha_degenerate_d_equals_h_returns_zero() {
        let c = cos_dha(v(1.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(2.0, 0.0, 0.0));
        assert_eq!(c, 0.0);
    }

    #[test]
    fn lj_kernel_energy_matches_raw_kernel() {
        let pair = LjPair {
            d0: 1.0,
            r0_sq: 4.0,
        };
        let m = LjMatrix::new(1, vec![pair]);
        let r_sq = 6.25_f32;
        let expected = LennardJones::energy(r_sq, (pair.d0, pair.r0_sq));
        assert_abs_diff_eq!(
            LjKernel(&m).energy(t(0), t(0), r_sq),
            expected,
            epsilon = 1e-7
        );
    }

    #[test]
    fn lj_kernel_minimum_at_equilibrium_distance() {
        let (d0, r0) = (2.0_f32, 3.0_f32);
        let m = LjMatrix::new(1, vec![LjPair { d0, r0_sq: r0 * r0 }]);
        let e = LjKernel(&m).energy(t(0), t(0), r0 * r0);
        assert_abs_diff_eq!(e, -d0, epsilon = 1e-5);
    }

    #[test]
    fn buck_kernel_energy_matches_raw_kernel() {
        let pair = BuckPair {
            a: 100.0,
            b: 2.5,
            c: 50.0,
            r_max_sq: 1.0,
            two_e_max: 200.0,
        };
        let m = BuckMatrix::new(1, vec![pair]);
        let r_sq = 9.0_f32;
        let expected = Buckingham::energy(
            r_sq,
            (pair.a, pair.b, pair.c, pair.r_max_sq, pair.two_e_max),
        );
        assert_abs_diff_eq!(
            BuckKernel(&m).energy(t(0), t(0), r_sq),
            expected,
            epsilon = 1e-6
        );
    }

    #[test]
    fn lj_and_buck_kernels_give_different_energies_for_same_input() {
        let lj_m = LjMatrix::new(
            1,
            vec![LjPair {
                d0: 1.0,
                r0_sq: 4.0,
            }],
        );
        let buck_m = BuckMatrix::new(
            1,
            vec![BuckPair {
                a: 100.0,
                b: 2.5,
                c: 10.0,
                r_max_sq: 1.0,
                two_e_max: 50.0,
            }],
        );
        let r_sq = 9.0_f32;
        assert_ne!(
            LjKernel(&lj_m).energy(t(0), t(0), r_sq),
            BuckKernel(&buck_m).energy(t(0), t(0), r_sq),
        );
    }

    #[test]
    fn hbond_energy_unknown_type_triple_returns_zero() {
        let empty = HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new());
        let e = hbond_energy(4.0, 1.0, t(0), t(1), t(2), &empty);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn hbond_energy_negative_at_equilibrium_linear_geometry() {
        let r_hb = 3.0_f32;
        let d_hb = 1.0_f32;
        let hbond = hbond_params(t(0), t(1), t(2), d_hb, r_hb * r_hb);
        let e = hbond_energy(r_hb * r_hb, 1.0, t(0), t(1), t(2), &hbond);
        assert_abs_diff_eq!(e, -d_hb, epsilon = 1e-5);
    }

    #[test]
    fn hbond_energy_zero_when_cos_theta_is_zero() {
        let r_hb = 3.0_f32;
        let hbond = hbond_params(t(0), t(1), t(2), 1.0, r_hb * r_hb);
        let e = hbond_energy(r_hb * r_hb, 0.0, t(0), t(1), t(2), &hbond);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn hbond_energy_zero_for_obtuse_geometry_via_cos_dha() {
        let r_hb = 1.0_f32;
        let hbond = hbond_params(t(0), t(1), t(2), 1.0, r_hb * r_hb);
        let d = v(0.0, 0.0, 0.0);
        let h = v(3.0, 0.0, 0.0);
        let a = v(1.0, 0.0, 0.0);
        let r_sq_da = d.dist_sq(a);
        let cos = cos_dha(d, h, a);
        let e = hbond_energy(r_sq_da, cos, t(0), t(1), t(2), &hbond);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn coulomb_energy_like_charges_positive() {
        let e = coulomb_energy::<true>(332.0, 1.0, 1.0, 9.0);
        assert_abs_diff_eq!(e, 332.0 / 9.0, epsilon = 1e-4);
    }

    #[test]
    fn coulomb_energy_opposite_charges_negative() {
        assert!(coulomb_energy::<true>(332.0, 1.0, -1.0, 9.0) < 0.0);
    }

    #[test]
    fn coulomb_energy_zero_charge_returns_zero() {
        assert_eq!(coulomb_energy::<true>(332.0, 0.0, 1.0, 9.0), 0.0);
    }

    #[test]
    fn coulomb_energy_scales_inversely_with_r_sq() {
        let e1 = coulomb_energy::<true>(1.0, 1.0, 1.0, 4.0);
        let e2 = coulomb_energy::<true>(1.0, 1.0, 1.0, 8.0);
        assert_abs_diff_eq!(e1 / e2, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn coulomb_energy_disabled_always_returns_zero() {
        assert_eq!(coulomb_energy::<false>(332.0, 1.0, 1.0, 9.0), 0.0);
        assert_eq!(coulomb_energy::<false>(332.0, -3.0, 5.0, 0.1), 0.0);
    }
}
