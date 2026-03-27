use crate::pack::model::{
    energy::{PairEnergyTable, SelfEnergyTable},
    graph::ContactGraph,
};
use rayon::prelude::*;

/// Runs Dead-End Elimination on the energy tables, pruning rotamers that
/// provably cannot appear in the Global Minimum Energy Conformation. Returns
/// the total number of candidates pruned.
pub fn dee(self_e: &mut SelfEnergyTable, pair_e: &PairEnergyTable, graph: &ContactGraph) -> usize {
    let n = self_e.n_slots();
    debug_assert_eq!(graph.n_slots(), n);
    debug_assert_eq!(graph.n_edges(), pair_e.n_edges());

    let mut fixed = vec![false; n];

    let mut total_eliminated = 0usize;

    for phase in [
        Phase::Goldstein,
        Phase::Split,
        Phase::Goldstein,
        Phase::Split,
    ] {
        total_eliminated += converge(phase, self_e, pair_e, graph, &fixed);
        absorb(self_e, pair_e, graph, &mut fixed);
    }

    total_eliminated
}

/// DEE Phase: Goldstein or Split.
#[derive(Clone, Copy)]
enum Phase {
    Goldstein,
    Split,
}

/// Runs the given DEE phase in rounds until no more eliminations occur.
fn converge(
    phase: Phase,
    self_e: &mut SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    fixed: &[bool],
) -> usize {
    let n = self_e.n_slots();
    let mut total = 0usize;

    loop {
        let elims: Vec<Vec<usize>> = (0..n)
            .into_par_iter()
            .map(|i| {
                if fixed[i] || !has_choice(self_e, i) {
                    return Vec::new();
                }
                match phase {
                    Phase::Goldstein => goldstein_slot(i, self_e, pair_e, graph, fixed),
                    Phase::Split => split_slot(i, self_e, pair_e, graph, fixed),
                }
            })
            .collect();

        let mut round_eliminated = 0usize;
        for (i, dead) in elims.into_iter().enumerate() {
            round_eliminated += dead.len();
            for s in dead {
                self_e.prune(i, s);
            }
        }

        if round_eliminated == 0 {
            break;
        }
        total += round_eliminated;
    }

    total
}

/// Returns `true` if slot `s` has at least two non-pruned candidates.
fn has_choice(self_e: &SelfEnergyTable, s: usize) -> bool {
    let mut seen = 0u8;
    for r in 0..self_e.n_candidates(s) {
        if !self_e.is_pruned(s, r) {
            seen += 1;
            if seen >= 2 {
                return true;
            }
        }
    }
    false
}

/// Pair-energy lookup for `(ci_at_slot_i, cj_at_neighbor)` on `edge`.
fn pair_val(mat: &[f32], stride: usize, is_left: bool, ci: usize, cj: usize) -> f32 {
    if is_left {
        mat[ci * stride + cj]
    } else {
        mat[cj * stride + ci]
    }
}

/// Applies the Goldstein criterion to every surviving candidate at slot `i`.
fn goldstein_slot(
    i: usize,
    self_e: &SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    fixed: &[bool],
) -> Vec<usize> {
    let nc = self_e.n_candidates(i);
    let mut dead = Vec::new();

    let edges: Vec<(usize, &[f32], usize, bool)> = graph
        .neighbor_edges(i)
        .filter(|&(j, _, _)| !fixed[j as usize])
        .map(|(j, e, is_left)| {
            let e = e as usize;
            let mat = pair_e.matrix(e);
            let stride = pair_e.dims(e).1;
            (j as usize, mat, stride, is_left)
        })
        .collect();

    for s in 0..nc {
        if self_e.is_pruned(i, s) {
            continue;
        }

        let es = self_e.get(i, s);

        let is_dead = (0..nc).any(|r| {
            if r == s || self_e.is_pruned(i, r) {
                return false;
            }

            let mut ex = es - self_e.get(i, r);

            for &(j, mat, stride, is_left) in &edges {
                let nc_j = self_e.n_candidates(j);

                let mut min_diff = f32::INFINITY;
                for t in 0..nc_j {
                    if self_e.is_pruned(j, t) {
                        continue;
                    }
                    let diff =
                        pair_val(mat, stride, is_left, s, t) - pair_val(mat, stride, is_left, r, t);
                    if diff < min_diff {
                        min_diff = diff;
                    }
                }

                ex += min_diff;
            }

            ex > 0.0
        });

        if is_dead {
            dead.push(s);
        }
    }

    dead
}

/// Applies the Split DEE criterion to every surviving candidate at slot `i`.
fn split_slot(
    i: usize,
    self_e: &SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    fixed: &[bool],
) -> Vec<usize> {
    let nc = self_e.n_candidates(i);
    let mut dead = Vec::new();

    let neighbors: Vec<(usize, &[f32], usize, bool)> = graph
        .neighbor_edges(i)
        .filter(|&(j, _, _)| !fixed[j as usize])
        .map(|(j, e, is_left)| {
            let e = e as usize;
            let mat = pair_e.matrix(e);
            let stride = pair_e.dims(e).1;
            (j as usize, mat, stride, is_left)
        })
        .collect();

    let n_nbr = neighbors.len();
    if n_nbr == 0 {
        return dead;
    }

    let max_comp = nc.saturating_sub(1);
    let mut comp_buf = Vec::with_capacity(max_comp);
    let mut diff_self_buf = Vec::with_capacity(max_comp);
    let mut min_pair_buf = vec![0.0f32; max_comp * n_nbr];
    let mut sum_all_buf = Vec::with_capacity(max_comp);

    for s in 0..nc {
        if self_e.is_pruned(i, s) {
            continue;
        }
        let es = self_e.get(i, s);

        comp_buf.clear();
        diff_self_buf.clear();
        for r in 0..nc {
            if r == s || self_e.is_pruned(i, r) {
                continue;
            }
            diff_self_buf.push(es - self_e.get(i, r));
            comp_buf.push(r);
        }
        let n_comp = comp_buf.len();
        if n_comp == 0 {
            continue;
        }

        for (ci, &r) in comp_buf.iter().enumerate() {
            for (ki, &(j, mat, stride, is_left)) in neighbors.iter().enumerate() {
                let nc_j = self_e.n_candidates(j);
                let mut min_v = f32::INFINITY;
                for t in 0..nc_j {
                    if self_e.is_pruned(j, t) {
                        continue;
                    }
                    let diff =
                        pair_val(mat, stride, is_left, s, t) - pair_val(mat, stride, is_left, r, t);
                    if diff < min_v {
                        min_v = diff;
                    }
                }
                min_pair_buf[ci * n_nbr + ki] = min_v;
            }
        }

        sum_all_buf.clear();
        for ci in 0..n_comp {
            let row = &min_pair_buf[ci * n_nbr..(ci + 1) * n_nbr];
            sum_all_buf.push(diff_self_buf[ci] + row.iter().sum::<f32>());
        }

        let mut pruned = false;

        'split_k: for (ki, &(k, mat_k, stride_k, is_left_k)) in neighbors.iter().enumerate() {
            let nc_k = self_e.n_candidates(k);

            let all_vk_eliminated = (0..nc_k).filter(|&vk| !self_e.is_pruned(k, vk)).all(|vk| {
                (0..n_comp).any(|ci| {
                    let ex = sum_all_buf[ci] - min_pair_buf[ci * n_nbr + ki]
                        + pair_val(mat_k, stride_k, is_left_k, s, vk)
                        - pair_val(mat_k, stride_k, is_left_k, comp_buf[ci], vk);
                    ex > 0.0
                })
            });

            if all_vk_eliminated {
                pruned = true;
                break 'split_k;
            }
        }

        if pruned {
            dead.push(s);
        }
    }

    dead
}

/// Fixes every slot that has exactly one surviving candidate and absorbs
/// its pair-energy contribution into the self-energy of its neighbors.
fn absorb(
    self_e: &mut SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    fixed: &mut [bool],
) {
    let n = self_e.n_slots();

    let newly_fixed: Vec<(usize, usize)> = (0..n)
        .filter_map(|s| {
            if fixed[s] {
                return None;
            }
            let mut sole = None;
            for r in 0..self_e.n_candidates(s) {
                if !self_e.is_pruned(s, r) {
                    if sole.is_some() {
                        return None; // more than one survivor
                    }
                    sole = Some(r);
                }
            }
            sole.map(|r| (s, r))
        })
        .collect();

    if newly_fixed.is_empty() {
        return;
    }

    for &(fi, best_rot) in &newly_fixed {
        for (j, edge, is_left) in graph.neighbor_edges(fi) {
            let j = j as usize;
            if fixed[j] {
                continue;
            }
            let edge = edge as usize;
            let mat = pair_e.matrix(edge);
            let stride = pair_e.dims(edge).1;

            let nc_j = self_e.n_candidates(j);
            for rj in 0..nc_j {
                if self_e.is_pruned(j, rj) {
                    continue;
                }
                let pv = pair_val(mat, stride, is_left, best_rot, rj);
                if pv != 0.0 {
                    let v = self_e.get(j, rj);
                    self_e.set(j, rj, v + pv);
                }
            }
        }
    }

    for &(fi, _) in &newly_fixed {
        fixed[fi] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_slot(
        counts: [u16; 2],
        pair_data: &[f32],
    ) -> (SelfEnergyTable, PairEnergyTable, ContactGraph) {
        let self_e = SelfEnergyTable::new(&counts);
        let pair_e = PairEnergyTable::new(&[(counts[0], counts[1])]);
        let graph = ContactGraph::build(2, [(0u32, 1u32)]);
        debug_assert_eq!(pair_data.len(), counts[0] as usize * counts[1] as usize);
        let mut pair_e = pair_e;
        let (ni, nj) = (counts[0] as usize, counts[1] as usize);
        for ri in 0..ni {
            for rj in 0..nj {
                pair_e.set(0, ri, rj, pair_data[ri * nj + rj]);
            }
        }
        (self_e, pair_e, graph)
    }

    fn alive(self_e: &SelfEnergyTable, s: usize) -> Vec<usize> {
        (0..self_e.n_candidates(s))
            .filter(|&r| !self_e.is_pruned(s, r))
            .collect()
    }

    #[test]
    fn goldstein_no_neighbors_eliminates_locally_dominated_candidate() {
        let mut self_e = SelfEnergyTable::new(&[3]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 2.0);
        self_e.set(0, 2, 3.0);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());
        let fixed = vec![false];

        let mut dead = goldstein_slot(0, &self_e, &pair_e, &graph, &fixed);
        dead.sort_unstable();
        assert_eq!(dead, [0, 2]);
    }

    #[test]
    fn goldstein_no_neighbors_tied_candidates_both_survive() {
        let mut self_e = SelfEnergyTable::new(&[3]);
        self_e.set(0, 0, 3.0);
        self_e.set(0, 1, 3.0);
        self_e.set(0, 2, 5.0);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());
        let fixed = vec![false];

        let dead = goldstein_slot(0, &self_e, &pair_e, &graph, &fixed);

        assert_eq!(dead, [2]);
    }

    #[test]
    fn goldstein_equal_self_energy_does_not_eliminate() {
        let mut self_e = SelfEnergyTable::new(&[2]);
        self_e.set(0, 0, 3.0);
        self_e.set(0, 1, 3.0);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());
        let fixed = vec![false];

        let dead = goldstein_slot(0, &self_e, &pair_e, &graph, &fixed);

        assert!(dead.is_empty());
    }

    #[test]
    fn goldstein_with_neighbor_eliminates_dominated_candidate() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 2.0);
        let fixed = vec![false, false];

        let dead = goldstein_slot(0, &self_e, &pair_e, &graph, &fixed);

        assert_eq!(dead, [0]);
    }

    #[test]
    fn goldstein_with_neighbor_saved_by_pair_term() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[-10.0, -10.0, 0.0, 0.0]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 2.0);
        let fixed = vec![false, false];

        let dead = goldstein_slot(0, &self_e, &pair_e, &graph, &fixed);

        assert!(!dead.contains(&0));
    }

    #[test]
    fn goldstein_fixed_neighbor_is_skipped() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[-5.0, -5.0, 0.0, 0.0]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 2.0);

        let dead_unfixed = goldstein_slot(0, &self_e, &pair_e, &graph, &[false, false]);
        assert!(
            !dead_unfixed.contains(&0),
            "pair term should rescue candidate 0 when unfixed"
        );

        let dead_fixed = goldstein_slot(0, &self_e, &pair_e, &graph, &[false, true]);
        assert!(
            dead_fixed.contains(&0),
            "candidate 0 must be eliminated when neighbor is fixed"
        );
    }

    #[test]
    fn goldstein_skips_already_pruned_candidates() {
        let mut self_e = SelfEnergyTable::new(&[3]);
        self_e.set(0, 0, 1.0);
        self_e.set(0, 1, 5.0);
        self_e.set(0, 2, 9.0);
        self_e.prune(0, 2);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());
        let fixed = vec![false];

        let dead = goldstein_slot(0, &self_e, &pair_e, &graph, &fixed);

        assert!(dead.contains(&1));
        assert!(!dead.contains(&2));
    }

    #[test]
    fn goldstein_eliminates_candidate_from_right_side_of_edge() {
        let (self_e, pair_e, graph) = two_slot([2, 2], &[0.0, 2.0, 0.0, 2.0]);
        let fixed = vec![false, false];

        let dead = goldstein_slot(1, &self_e, &pair_e, &graph, &fixed);

        assert_eq!(dead, [1]);
    }

    #[test]
    fn split_eliminates_candidate_when_goldstein_cannot() {
        let (self_e, pair_e, graph) = two_slot([3, 2], &[10.0, -10.0, 11.0, -15.0, 0.0, 5.0]);
        let fixed = vec![false, false];

        let g_dead = goldstein_slot(0, &self_e, &pair_e, &graph, &fixed);
        assert!(
            !g_dead.contains(&0),
            "Goldstein must not eliminate s=0 in this case"
        );

        let s_dead = split_slot(0, &self_e, &pair_e, &graph, &fixed);
        assert!(s_dead.contains(&0), "Split must eliminate s=0");
    }

    #[test]
    fn split_does_not_fire_when_criterion_fails_for_some_vk() {
        let (self_e, pair_e, graph) = two_slot([2, 2], &[5.0, -100.0, 0.0, 0.0]);
        let fixed = vec![false, false];

        let dead = split_slot(0, &self_e, &pair_e, &graph, &fixed);

        assert!(!dead.contains(&0));
    }

    #[test]
    fn split_with_single_candidate_produces_no_dead() {
        let mut self_e = SelfEnergyTable::new(&[2, 2]);
        self_e.set(0, 0, 0.0);
        self_e.prune(0, 1);
        let pair_e = PairEnergyTable::new(&[(2, 2)]);
        let graph = ContactGraph::build(2, [(0u32, 1u32)]);
        let fixed = vec![false, false];

        let dead = split_slot(0, &self_e, &pair_e, &graph, &fixed);

        assert!(dead.is_empty());
    }

    #[test]
    fn split_with_no_neighbors_produces_no_dead() {
        let mut self_e = SelfEnergyTable::new(&[3]);
        self_e.set(0, 0, 1.0);
        self_e.set(0, 1, 5.0);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());
        let fixed = vec![false];

        let dead = split_slot(0, &self_e, &pair_e, &graph, &fixed);

        assert!(dead.is_empty());
    }

    #[test]
    fn absorb_folds_pair_energy_into_unfixed_neighbor() {
        let (mut self_e, pair_e, graph) = two_slot([1, 2], &[3.0, 7.0]);
        let mut fixed = vec![false, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(1, 0), 3.0);
        assert_eq!(self_e.get(1, 1), 7.0);
        assert!(fixed[0]);
        assert!(!fixed[1]);
    }

    #[test]
    fn absorb_does_not_touch_multi_candidate_slot() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let mut fixed = vec![false, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert!(!fixed[0]);
        assert!(!fixed[1]);
        assert_eq!(self_e.get(0, 0), 0.0);
        assert_eq!(self_e.get(1, 0), 0.0);
    }

    #[test]
    fn absorb_skips_already_fixed_neighbor() {
        let (mut self_e, pair_e, graph) = two_slot([1, 1], &[5.0]);
        let mut fixed = vec![false, true];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(1, 0), 0.0);
        assert!(fixed[0]);
    }

    #[test]
    fn absorb_no_op_when_all_already_fixed() {
        let (mut self_e, pair_e, graph) = two_slot([1, 2], &[3.0, 7.0]);
        let mut fixed = vec![true, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(1, 0), 0.0);
        assert_eq!(self_e.get(1, 1), 0.0);
    }

    #[test]
    fn absorb_skips_pruned_candidates_at_neighbor() {
        let (mut self_e, pair_e, graph) = two_slot([1, 2], &[3.0, 7.0]);
        self_e.prune(1, 1);
        let mut fixed = vec![false, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(1, 0), 3.0);
        assert_eq!(self_e.get(1, 1), f32::INFINITY);
    }

    #[test]
    fn absorb_right_side_fixed_slot_absorbs_into_neighbor() {
        let (mut self_e, pair_e, graph) = two_slot([2, 1], &[3.0, 7.0]);
        let mut fixed = vec![false, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(0, 0), 3.0);
        assert_eq!(self_e.get(0, 1), 7.0);
        assert!(fixed[1]);
        assert!(!fixed[0]);
    }

    #[test]
    fn dee_no_op_on_single_candidate_slots() {
        let (mut self_e, pair_e, graph) = two_slot([1, 1], &[1.0]);
        let n = dee(&mut self_e, &pair_e, &graph);
        assert_eq!(n, 0);
    }

    #[test]
    fn dee_returns_total_eliminated_count() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[0.0; 4]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 1.0);
        self_e.set(1, 0, 3.0);
        self_e.set(1, 1, 2.0);

        let n = dee(&mut self_e, &pair_e, &graph);

        assert_eq!(n, 2);
        assert!(self_e.is_pruned(0, 0));
        assert!(!self_e.is_pruned(0, 1));
        assert!(self_e.is_pruned(1, 0));
        assert!(!self_e.is_pruned(1, 1));
    }

    #[test]
    fn dee_does_not_eliminate_equal_energy_candidates() {
        let (mut self_e, pair_e, graph) = two_slot([2, 1], &[0.0, 0.0]);
        self_e.set(0, 0, 4.0);
        self_e.set(0, 1, 4.0);

        dee(&mut self_e, &pair_e, &graph);

        assert!(!self_e.is_pruned(0, 0));
        assert!(!self_e.is_pruned(0, 1));
    }

    #[test]
    fn dee_converges_to_single_survivor_per_slot() {
        let mut self_e = SelfEnergyTable::new(&[5, 5, 5]);
        for s in 0..3 {
            self_e.set(s, 0, 0.0);
            self_e.set(s, 1, 1.0);
            self_e.set(s, 2, 2.0);
            self_e.set(s, 3, 3.0);
            self_e.set(s, 4, 4.0);
        }
        let pair_e = PairEnergyTable::new(&[(5, 5), (5, 5)]);
        let graph = ContactGraph::build(3, [(0u32, 1u32), (1u32, 2u32)]);

        let n = dee(&mut self_e, &pair_e, &graph);

        assert_eq!(n, 12);
        for s in 0..3 {
            assert_eq!(alive(&self_e, s), [0]);
        }
    }
}
