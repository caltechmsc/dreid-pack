use std::f64::consts::PI;

use arrayvec::ArrayVec;

use crate::chi::geometry::diff;

pub fn chi_diffs(
    crystal: &[Option<f64>],
    packed: &[Option<f64>],
    symmetric_last: bool,
) -> ArrayVec<Option<f64>, 4> {
    let n = crystal.len().min(packed.len());
    let mut out = ArrayVec::new();

    for i in 0..n {
        let d = match (crystal[i], packed[i]) {
            (Some(c), Some(p)) => {
                let mut d = diff(c, p);
                if symmetric_last && i + 1 == n {
                    d = d.min(PI - d);
                }
                Some(d)
            }
            _ => None,
        };
        out.push(d);
    }

    out
}
