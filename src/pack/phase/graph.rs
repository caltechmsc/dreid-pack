use crate::{
    model::{system::Residue, types::Vec3},
    pack::model::{graph::ContactGraph, spatial::SpatialGrid},
};
use rayon::prelude::*;

/// Builds the [`ContactGraph`] connecting slots whose Cα distance satisfies
/// `dist ≤ reach_i + reach_j + vdw_cutoff`.
pub fn build(slots: &[Residue], vdw_cutoff: f32) -> ContactGraph {
    let n = slots.len();
    if n == 0 {
        return ContactGraph::build(0, std::iter::empty());
    }

    let slots_info: Vec<(Vec3, f32)> = slots
        .par_iter()
        .map(|s| (s.anchor()[1], s.res_type().reach()))
        .collect();

    let max_reach = slots_info
        .par_iter()
        .map(|&(_, r)| r)
        .reduce(|| 0.0_f32, f32::max);
    let cell_size = 2.0 * max_reach + vdw_cutoff;

    let grid = SpatialGrid::build(
        slots_info
            .iter()
            .enumerate()
            .map(|(i, &(p, r))| (p, (r, i as u32))),
        cell_size,
    );

    let slots_info = &slots_info;

    let edges: Vec<(u32, u32)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (ca_i, reach_i) = slots_info[i];
            let query_r = reach_i + max_reach + vdw_cutoff;

            grid.query(ca_i, query_r)
                .filter_map(move |(pos_j, (reach_j, j))| {
                    if j <= i as u32 {
                        return None;
                    }
                    let threshold = reach_i + reach_j + vdw_cutoff;
                    if ca_i.dist_sq(pos_j) <= threshold * threshold {
                        Some((i as u32, j))
                    } else {
                        None
                    }
                })
        })
        .collect();

    ContactGraph::build(n, edges)
}
