mod compare;
mod extract;
mod geometry;
mod symmetry;

use arrayvec::ArrayVec;
use bio_forge::Structure;

use crate::residue::AminoAcid;

pub fn diff(
    crystal: &Structure,
    packed: &Structure,
    chain_id: &str,
    res_id: i32,
    ins_code: Option<char>,
    aa: AminoAcid,
) -> ArrayVec<Option<f64>, 4> {
    let chi_crystal = extract::extract(crystal, chain_id, res_id, ins_code, aa);
    let chi_packed = extract::extract(packed, chain_id, res_id, ins_code, aa);
    let sym = symmetry::has_symmetric_last_chi(aa);
    compare::chi_diffs(&chi_crystal, &chi_packed, sym)
}
