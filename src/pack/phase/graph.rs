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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{residue::ResidueType, system::SidechainAtoms, types::TypeIdx};

    fn make_slot(ca: Vec3, res_type: ResidueType) -> Residue {
        let n = res_type.n_atoms() as usize;
        let coords = vec![Vec3::zero(); n];
        let types = vec![TypeIdx(0); n];
        let charges = vec![0.0f32; n];
        let donor_of_h = vec![u8::MAX; n];
        Residue::new(
            res_type,
            [Vec3::zero(), ca, Vec3::zero()],
            0.0,
            0.0,
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
    fn build_empty_slots_returns_no_edges() {
        let g = build(&[], 4.0);
        assert_eq!(g.n_slots(), 0);
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn build_single_slot_returns_no_edges() {
        let slots = [make_slot(Vec3::zero(), ResidueType::Ser)];
        let g = build(&slots, 4.0);
        assert_eq!(g.n_slots(), 1);
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn close_pair_within_threshold_yields_one_edge() {
        let slots = [
            make_slot(Vec3::new(0.0, 0.0, 0.0), ResidueType::Ser),
            make_slot(Vec3::new(5.0, 0.0, 0.0), ResidueType::Ser),
        ];
        let g = build(&slots, 4.0);
        assert_eq!(g.n_edges(), 1);
        assert_eq!(g.edges(), &[(0, 1)]);
    }

    #[test]
    fn pair_beyond_threshold_yields_no_edge() {
        let slots = [
            make_slot(Vec3::new(0.0, 0.0, 0.0), ResidueType::Ser),
            make_slot(Vec3::new(15.0, 0.0, 0.0), ResidueType::Ser),
        ];
        let g = build(&slots, 4.0);
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn mixed_reaches_within_threshold_yield_one_edge() {
        let slots = [
            make_slot(Vec3::new(0.0, 0.0, 0.0), ResidueType::Ser),
            make_slot(Vec3::new(14.0, 0.0, 0.0), ResidueType::Trp),
        ];
        let g = build(&slots, 4.0);
        assert_eq!(g.n_edges(), 1);
    }

    #[test]
    fn chain_connectivity_skips_distant_pair() {
        let slots = [
            make_slot(Vec3::new(0.0, 0.0, 0.0), ResidueType::Ser),
            make_slot(Vec3::new(11.0, 0.0, 0.0), ResidueType::Ser),
            make_slot(Vec3::new(22.0, 0.0, 0.0), ResidueType::Ser),
        ];
        let g = build(&slots, 4.0);
        assert_eq!(g.n_edges(), 2);
        assert_eq!(g.edges(), &[(0, 1), (1, 2)]);
    }
}
