use crate::pack::model::{
    energy::{PairEnergyTable, SelfEnergyTable},
    graph::ContactGraph,
};
use rayon::prelude::*;

/// Minimum chunk length for parallel dispatch.
const MIN_PAR_LEN: usize = 32;

/// Runs Dead-End Elimination on the energy tables, pruning rotamers that
/// provably cannot appear in the Global Minimum Energy Conformation. Returns
/// the total number of candidates pruned.
pub fn dee(self_e: &mut SelfEnergyTable, pair_e: &PairEnergyTable, graph: &ContactGraph) -> usize {
    let n = self_e.n_slots();
    debug_assert_eq!(graph.n_slots(), n);
    debug_assert_eq!(graph.n_edges(), pair_e.n_edges());

    let mut fixed = vec![false; n];
    let mut total = 0;

    for phase in [
        Phase::Goldstein,
        Phase::Split,
        Phase::Goldstein,
        Phase::Split,
    ] {
        total += converge(phase, self_e, pair_e, graph, &fixed);
        absorb(self_e, pair_e, graph, &mut fixed);
    }

    total
}

/// DEE phase selector.
#[derive(Clone, Copy)]
enum Phase {
    Goldstein,
    Split,
}

/// Precomputed edge to a non-fixed neighbor slot, carrying a reference into
/// the pair-energy matrix and the index-layout flag.
struct Edge<'a> {
    slot: usize,
    mat: &'a [f32],
    stride: usize,
    is_left: bool,
}

impl Edge<'_> {
    /// Pair-energy value for candidate `ci` at the source slot and `cj` here.
    fn pair_val(&self, ci: usize, cj: usize) -> f32 {
        if self.is_left {
            self.mat[ci * self.stride + cj]
        } else {
            self.mat[cj * self.stride + ci]
        }
    }

    /// Tightest pair-energy gap between `s` and `r` over living candidates.
    fn min_diff(&self, s: usize, r: usize, alive: &[u16]) -> f32 {
        if self.is_left {
            let off_s = s * self.stride;
            let off_r = r * self.stride;
            let mut m = f32::INFINITY;
            for &t in alive {
                let d = self.mat[off_s + t as usize] - self.mat[off_r + t as usize];
                if d < m {
                    m = d;
                }
            }
            m
        } else {
            let mut m = f32::INFINITY;
            for &t in alive {
                let off = t as usize * self.stride;
                let d = self.mat[off + s] - self.mat[off + r];
                if d < m {
                    m = d;
                }
            }
            m
        }
    }
}

/// Collects alive candidate indices per slot, sorted by ascending self-energy.
fn build_alive(self_e: &SelfEnergyTable) -> Vec<Vec<u16>> {
    (0..self_e.n_slots())
        .map(|s| {
            let mut v: Vec<u16> = (0..self_e.n_candidates(s) as u16)
                .filter(|&r| !self_e.is_pruned(s, r as usize))
                .collect();
            v.sort_unstable_by(|&a, &b| {
                self_e
                    .get(s, a as usize)
                    .total_cmp(&self_e.get(s, b as usize))
            });
            v
        })
        .collect()
}

/// Collects precomputed [`Edge`] info per slot, skipping fixed neighbors.
fn build_edges<'a>(
    n: usize,
    pair_e: &'a PairEnergyTable,
    graph: &ContactGraph,
    fixed: &[bool],
) -> Vec<Vec<Edge<'a>>> {
    (0..n)
        .map(|i| {
            graph
                .neighbor_edges(i)
                .filter(|&(j, _, _)| !fixed[j as usize])
                .map(|(j, e, is_left)| {
                    let e = e as usize;
                    Edge {
                        slot: j as usize,
                        mat: pair_e.matrix(e),
                        stride: pair_e.dims(e).1,
                        is_left,
                    }
                })
                .collect()
        })
        .collect()
}

/// Runs the given DEE phase in rounds until convergence.
fn converge(
    phase: Phase,
    self_e: &mut SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    fixed: &[bool],
) -> usize {
    let n = self_e.n_slots();
    let mut alive = build_alive(self_e);
    let edges = build_edges(n, pair_e, graph, fixed);

    let mut dirty: Vec<bool> = (0..n).map(|i| !fixed[i] && alive[i].len() >= 2).collect();
    let mut total = 0;

    loop {
        let work: Vec<usize> = dirty
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| d.then_some(i))
            .collect();
        if work.is_empty() {
            break;
        }
        for &i in &work {
            dirty[i] = false;
        }

        let eliminate = |&i: &usize| -> Option<(usize, Vec<usize>)> {
            if alive[i].len() < 2 {
                return None;
            }
            let dead = match phase {
                Phase::Goldstein => goldstein(i, self_e, &edges[i], &alive),
                Phase::Split => split(i, self_e, &edges[i], &alive),
            };
            if dead.is_empty() {
                None
            } else {
                Some((i, dead))
            }
        };

        let elims: Vec<(usize, Vec<usize>)> = work
            .par_iter()
            .with_min_len(MIN_PAR_LEN)
            .filter_map(eliminate)
            .collect();
        if elims.is_empty() {
            break;
        }

        for (i, dead) in &elims {
            total += dead.len();
            for &r in dead {
                self_e.prune(*i, r);
            }
            alive[*i].retain(|&r| !dead.contains(&(r as usize)));
            for edge in &edges[*i] {
                if alive[edge.slot].len() >= 2 {
                    dirty[edge.slot] = true;
                }
            }
        }
    }

    total
}

/// Goldstein DEE: candidate `s` at slot `i` is dead if any single witness `r`
/// dominates it across all neighbor interactions.
fn goldstein(
    i: usize,
    self_e: &SelfEnergyTable,
    edges: &[Edge<'_>],
    alive: &[Vec<u16>],
) -> Vec<usize> {
    let alive_i = &alive[i];
    let mut dead = Vec::new();

    for &s in alive_i {
        let s = s as usize;
        let es = self_e.get(i, s);

        let dominated = alive_i.iter().any(|&r| {
            let r = r as usize;
            if r == s {
                return false;
            }
            let mut excess = es - self_e.get(i, r);
            for edge in edges {
                excess += edge.min_diff(s, r, &alive[edge.slot]);
            }
            excess > 0.0
        });

        if dominated {
            dead.push(s);
        }
    }

    dead
}

/// Split DEE: candidate `s` at slot `i` is dead if, for some neighbor `k`,
/// every candidate `vk` at `k` is dominated by some witness `r`.
fn split(i: usize, self_e: &SelfEnergyTable, edges: &[Edge<'_>], alive: &[Vec<u16>]) -> Vec<usize> {
    let alive_i = &alive[i];
    let n_nbr = edges.len();
    if n_nbr == 0 {
        return Vec::new();
    }

    let max_wit = alive_i.len().saturating_sub(1);
    let mut witnesses: Vec<usize> = Vec::with_capacity(max_wit);
    let mut self_diffs = Vec::with_capacity(max_wit);
    let mut pair_diffs = vec![0.0f32; max_wit * n_nbr];
    let mut totals = Vec::with_capacity(max_wit);
    let mut dead = Vec::new();

    for &s in alive_i {
        let s = s as usize;
        let es = self_e.get(i, s);

        witnesses.clear();
        self_diffs.clear();
        for &r in alive_i {
            let r = r as usize;
            if r == s {
                continue;
            }
            self_diffs.push(es - self_e.get(i, r));
            witnesses.push(r);
        }
        let n_wit = witnesses.len();
        if n_wit == 0 {
            continue;
        }

        for (wi, &r) in witnesses.iter().enumerate() {
            for (ki, edge) in edges.iter().enumerate() {
                pair_diffs[wi * n_nbr + ki] = edge.min_diff(s, r, &alive[edge.slot]);
            }
        }

        totals.clear();
        for wi in 0..n_wit {
            let row = &pair_diffs[wi * n_nbr..(wi + 1) * n_nbr];
            totals.push(self_diffs[wi] + row.iter().sum::<f32>());
        }

        let pruned = edges.iter().enumerate().any(|(ki, edge)| {
            alive[edge.slot].iter().all(|&vk| {
                let vk = vk as usize;
                (0..n_wit).any(|wi| {
                    totals[wi] - pair_diffs[wi * n_nbr + ki] + edge.pair_val(s, vk)
                        - edge.pair_val(witnesses[wi], vk)
                        > 0.0
                })
            })
        });

        if pruned {
            dead.push(s);
        }
    }

    dead
}

/// Fixes every slot that has exactly one surviving candidate and folds its
/// pair-energy contribution into the self-energy of each neighbor.
fn absorb(
    self_e: &mut SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    fixed: &mut [bool],
) {
    let newly_fixed: Vec<(usize, usize)> = (0..self_e.n_slots())
        .filter_map(|s| {
            if fixed[s] {
                return None;
            }
            let mut sole = None;
            for r in 0..self_e.n_candidates(s) {
                if !self_e.is_pruned(s, r) {
                    if sole.is_some() {
                        return None;
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

    for &(fi, sole_rot) in &newly_fixed {
        for (j, e, is_left) in graph.neighbor_edges(fi) {
            let j = j as usize;
            if fixed[j] {
                continue;
            }
            let e = e as usize;
            let mat = pair_e.matrix(e);
            let stride = pair_e.dims(e).1;

            for rj in 0..self_e.n_candidates(j) {
                if self_e.is_pruned(j, rj) {
                    continue;
                }
                let pv = if is_left {
                    mat[sole_rot * stride + rj]
                } else {
                    mat[rj * stride + sole_rot]
                };
                if pv != 0.0 {
                    self_e.set(j, rj, self_e.get(j, rj) + pv);
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
    use crate::pack::model::energy::PRUNED;

    fn two_slot(
        counts: [u16; 2],
        pair_data: &[f32],
    ) -> (SelfEnergyTable, PairEnergyTable, ContactGraph) {
        let self_e = SelfEnergyTable::new(&counts);
        let mut pair_e = PairEnergyTable::new(&[(counts[0], counts[1])]);
        debug_assert_eq!(pair_data.len(), counts[0] as usize * counts[1] as usize);
        pair_e.matrices_mut()[0].copy_from_slice(pair_data);
        let graph = ContactGraph::build(2, [(0u32, 1u32)]);
        (self_e, pair_e, graph)
    }

    fn alive_at(self_e: &SelfEnergyTable, s: usize) -> Vec<usize> {
        (0..self_e.n_candidates(s))
            .filter(|&r| !self_e.is_pruned(s, r))
            .collect()
    }

    fn run_goldstein(
        i: usize,
        self_e: &SelfEnergyTable,
        pair_e: &PairEnergyTable,
        graph: &ContactGraph,
        fixed: &[bool],
    ) -> Vec<usize> {
        let alive = build_alive(self_e);
        let edges = build_edges(self_e.n_slots(), pair_e, graph, fixed);
        goldstein(i, self_e, &edges[i], &alive)
    }

    fn run_split(
        i: usize,
        self_e: &SelfEnergyTable,
        pair_e: &PairEnergyTable,
        graph: &ContactGraph,
        fixed: &[bool],
    ) -> Vec<usize> {
        let alive = build_alive(self_e);
        let edges = build_edges(self_e.n_slots(), pair_e, graph, fixed);
        split(i, self_e, &edges[i], &alive)
    }

    #[test]
    fn goldstein_eliminates_dominated_candidates() {
        let mut self_e = SelfEnergyTable::new(&[3]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 2.0);
        self_e.set(0, 2, 3.0);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());

        let mut dead = run_goldstein(0, &self_e, &pair_e, &graph, &[false]);
        dead.sort_unstable();
        assert_eq!(dead, [0, 2]);
    }

    #[test]
    fn goldstein_preserves_tied_lowest() {
        let mut self_e = SelfEnergyTable::new(&[3]);
        self_e.set(0, 0, 3.0);
        self_e.set(0, 1, 3.0);
        self_e.set(0, 2, 5.0);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());

        let dead = run_goldstein(0, &self_e, &pair_e, &graph, &[false]);
        assert_eq!(dead, [2]);
    }

    #[test]
    fn goldstein_exact_tie_survives() {
        let mut self_e = SelfEnergyTable::new(&[2]);
        self_e.set(0, 0, 3.0);
        self_e.set(0, 1, 3.0);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());

        let dead = run_goldstein(0, &self_e, &pair_e, &graph, &[false]);
        assert!(dead.is_empty());
    }

    #[test]
    fn goldstein_with_pair_energy() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 2.0);

        let dead = run_goldstein(0, &self_e, &pair_e, &graph, &[false, false]);
        assert_eq!(dead, [0]);
    }

    #[test]
    fn goldstein_rescued_by_pair_energy() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[-10.0, -10.0, 0.0, 0.0]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 2.0);

        let dead = run_goldstein(0, &self_e, &pair_e, &graph, &[false, false]);
        assert!(!dead.contains(&0));
    }

    #[test]
    fn goldstein_ignores_fixed_neighbor() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[-5.0, -5.0, 0.0, 0.0]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 2.0);

        let dead_unfixed = run_goldstein(0, &self_e, &pair_e, &graph, &[false, false]);
        assert!(
            !dead_unfixed.contains(&0),
            "pair term should rescue candidate 0 when unfixed"
        );

        let dead_fixed = run_goldstein(0, &self_e, &pair_e, &graph, &[false, true]);
        assert!(
            dead_fixed.contains(&0),
            "candidate 0 must be eliminated when neighbor is fixed"
        );
    }

    #[test]
    fn goldstein_skips_pruned() {
        let mut self_e = SelfEnergyTable::new(&[3]);
        self_e.set(0, 0, 1.0);
        self_e.set(0, 1, 5.0);
        self_e.set(0, 2, 9.0);
        self_e.prune(0, 2);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());

        let dead = run_goldstein(0, &self_e, &pair_e, &graph, &[false]);
        assert!(dead.contains(&1));
        assert!(!dead.contains(&2));
    }

    #[test]
    fn goldstein_right_side_of_edge() {
        let (self_e, pair_e, graph) = two_slot([2, 2], &[0.0, 2.0, 0.0, 2.0]);

        let dead = run_goldstein(1, &self_e, &pair_e, &graph, &[false, false]);
        assert_eq!(dead, [1]);
    }

    #[test]
    fn split_succeeds_where_goldstein_fails() {
        let (self_e, pair_e, graph) = two_slot([3, 2], &[10.0, -10.0, 11.0, -15.0, 0.0, 5.0]);
        let fixed = [false, false];

        let g_dead = run_goldstein(0, &self_e, &pair_e, &graph, &fixed);
        assert!(!g_dead.contains(&0), "Goldstein must not eliminate s=0");

        let s_dead = run_split(0, &self_e, &pair_e, &graph, &fixed);
        assert!(s_dead.contains(&0), "Split must eliminate s=0");
    }

    #[test]
    fn split_blocked_by_surviving_vk() {
        let (self_e, pair_e, graph) = two_slot([2, 2], &[5.0, -100.0, 0.0, 0.0]);

        let dead = run_split(0, &self_e, &pair_e, &graph, &[false, false]);
        assert!(!dead.contains(&0));
    }

    #[test]
    fn split_single_candidate_no_op() {
        let mut self_e = SelfEnergyTable::new(&[2, 2]);
        self_e.set(0, 0, 0.0);
        self_e.prune(0, 1);
        let pair_e = PairEnergyTable::new(&[(2, 2)]);
        let graph = ContactGraph::build(2, [(0u32, 1u32)]);

        let dead = run_split(0, &self_e, &pair_e, &graph, &[false, false]);
        assert!(dead.is_empty());
    }

    #[test]
    fn split_no_neighbors_no_op() {
        let mut self_e = SelfEnergyTable::new(&[3]);
        self_e.set(0, 0, 1.0);
        self_e.set(0, 1, 5.0);
        let pair_e = PairEnergyTable::new(&[]);
        let graph = ContactGraph::build(1, std::iter::empty::<(u32, u32)>());

        let dead = run_split(0, &self_e, &pair_e, &graph, &[false]);
        assert!(dead.is_empty());
    }

    #[test]
    fn absorb_folds_into_neighbor() {
        let (mut self_e, pair_e, graph) = two_slot([1, 2], &[3.0, 7.0]);
        let mut fixed = vec![false, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(1, 0), 3.0);
        assert_eq!(self_e.get(1, 1), 7.0);
        assert!(fixed[0]);
        assert!(!fixed[1]);
    }

    #[test]
    fn absorb_no_op_multi_candidate() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let mut fixed = vec![false, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert!(!fixed[0]);
        assert!(!fixed[1]);
        assert_eq!(self_e.get(0, 0), 0.0);
        assert_eq!(self_e.get(1, 0), 0.0);
    }

    #[test]
    fn absorb_skips_fixed_neighbor() {
        let (mut self_e, pair_e, graph) = two_slot([1, 1], &[5.0]);
        let mut fixed = vec![false, true];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(1, 0), 0.0);
        assert!(fixed[0]);
    }

    #[test]
    fn absorb_no_op_all_fixed() {
        let (mut self_e, pair_e, graph) = two_slot([1, 2], &[3.0, 7.0]);
        let mut fixed = vec![true, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(1, 0), 0.0);
        assert_eq!(self_e.get(1, 1), 0.0);
    }

    #[test]
    fn absorb_skips_pruned() {
        let (mut self_e, pair_e, graph) = two_slot([1, 2], &[3.0, 7.0]);
        self_e.prune(1, 1);
        let mut fixed = vec![false, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(1, 0), 3.0);
        assert_eq!(self_e.get(1, 1), PRUNED);
    }

    #[test]
    fn absorb_right_side_of_edge() {
        let (mut self_e, pair_e, graph) = two_slot([2, 1], &[3.0, 7.0]);
        let mut fixed = vec![false, false];

        absorb(&mut self_e, &pair_e, &graph, &mut fixed);

        assert_eq!(self_e.get(0, 0), 3.0);
        assert_eq!(self_e.get(0, 1), 7.0);
        assert!(fixed[1]);
        assert!(!fixed[0]);
    }

    #[test]
    fn dee_single_candidates_no_op() {
        let (mut self_e, pair_e, graph) = two_slot([1, 1], &[1.0]);
        assert_eq!(dee(&mut self_e, &pair_e, &graph), 0);
    }

    #[test]
    fn dee_returns_elimination_count() {
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
    fn dee_preserves_tied_candidates() {
        let (mut self_e, pair_e, graph) = two_slot([2, 1], &[0.0, 0.0]);
        self_e.set(0, 0, 4.0);
        self_e.set(0, 1, 4.0);

        dee(&mut self_e, &pair_e, &graph);

        assert!(!self_e.is_pruned(0, 0));
        assert!(!self_e.is_pruned(0, 1));
    }

    #[test]
    fn dee_full_convergence() {
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
            assert_eq!(alive_at(&self_e, s), [0]);
        }
    }
}
