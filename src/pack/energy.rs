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
