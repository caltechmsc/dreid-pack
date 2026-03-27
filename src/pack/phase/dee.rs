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

    let mut alive: Vec<Vec<usize>> = (0..n)
        .map(|s| {
            (0..self_e.n_candidates(s))
                .filter(|&r| !self_e.is_pruned(s, r))
                .collect()
        })
        .collect();

    let mut total_eliminated = 0usize;

    for phase in [Phase::Goldstein, Phase::Split, Phase::Goldstein] {
        let eliminated = converge(phase, self_e, pair_e, graph, &mut alive);
        total_eliminated += eliminated;
        absorb(self_e, pair_e, graph, &mut alive);
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
    alive: &mut [Vec<usize>],
) -> usize {
    let n = self_e.n_slots();
    let mut total = 0usize;

    loop {
        let snap: Vec<Vec<usize>> = alive.to_vec();

        let elims: Vec<Vec<usize>> = (0..n)
            .into_par_iter()
            .map(|i| {
                if snap[i].len() <= 1 {
                    return Vec::new();
                }
                match phase {
                    Phase::Goldstein => goldstein_slot(i, self_e, pair_e, graph, &snap),
                    Phase::Split => split_slot(i, self_e, pair_e, graph, &snap),
                }
            })
            .collect();

        let mut round_eliminated = 0usize;
        for (i, dead) in elims.into_iter().enumerate() {
            for s in &dead {
                self_e.prune(i, *s);
            }
            if !dead.is_empty() {
                round_eliminated += dead.len();
                alive[i].retain(|r| !dead.contains(r));
            }
        }

        if round_eliminated == 0 {
            break;
        }
        total += round_eliminated;
    }

    total
}

/// Applies the Goldstein criterion to every surviving candidate at slot `i`.
/// Returns the list of candidate indices that were eliminated.
fn goldstein_slot(
    i: usize,
    self_e: &SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    alive: &[Vec<usize>],
) -> Vec<usize> {
    let alive_i = &alive[i];
    let mut dead = Vec::new();

    for &s in alive_i {
        if dead.contains(&s) {
            continue;
        }

        let es = self_e.get(i, s);

        let is_eliminated = alive_i.iter().any(|&r| {
            if r == s || dead.contains(&r) {
                return false;
            }

            let mut ex = es - self_e.get(i, r);

            for (j, edge, is_left) in graph.neighbor_edges(i) {
                let j = j as usize;
                let edge = edge as usize;

                if alive[j].is_empty() {
                    continue;
                }

                let pair_val = |cand_i: usize, cand_j: usize| {
                    if is_left {
                        pair_e.get(edge, cand_i, cand_j)
                    } else {
                        pair_e.get(edge, cand_j, cand_i)
                    }
                };

                let mut min_diff = f32::INFINITY;
                for &t in &alive[j] {
                    let diff = pair_val(s, t) - pair_val(r, t);
                    if diff < min_diff {
                        min_diff = diff;
                    }
                }

                ex += min_diff;
            }

            ex > 0.0
        });

        if is_eliminated {
            dead.push(s);
        }
    }

    dead
}

/// Applies the Split DEE criterion to every surviving candidate at slot `i`.
/// Returns the list of candidate indices that were eliminated.
fn split_slot(
    i: usize,
    self_e: &SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    alive: &[Vec<usize>],
) -> Vec<usize> {
    let alive_i = &alive[i];
    let mut dead = Vec::new();

    let neighbors: Vec<(usize, usize, bool)> = graph
        .neighbor_edges(i)
        .map(|(j, e, left)| (j as usize, e as usize, left))
        .collect();

    for &s in alive_i {
        if dead.contains(&s) {
            continue;
        }
        let es = self_e.get(i, s);

        let competitors: Vec<usize> = alive_i
            .iter()
            .copied()
            .filter(|&r| r != s && !dead.contains(&r))
            .collect();

        if competitors.is_empty() {
            continue;
        }

        let diff_self: Vec<f32> = competitors.iter().map(|&r| es - self_e.get(i, r)).collect();

        let min_pair: Vec<Vec<f32>> = competitors
            .iter()
            .map(|&r| {
                neighbors
                    .iter()
                    .map(|&(j, edge, is_left)| {
                        let pair_val = |ci: usize, cj: usize| {
                            if is_left {
                                pair_e.get(edge, ci, cj)
                            } else {
                                pair_e.get(edge, cj, ci)
                            }
                        };
                        alive[j]
                            .iter()
                            .map(|&t| pair_val(s, t) - pair_val(r, t))
                            .fold(f32::INFINITY, f32::min)
                    })
                    .collect()
            })
            .collect();

        let sum_all: Vec<f32> = (0..competitors.len())
            .map(|ci| diff_self[ci] + min_pair[ci].iter().sum::<f32>())
            .collect();

        let mut pruned = false;

        'split_k: for (ki, &(k, edge_k, is_left_k)) in neighbors.iter().enumerate() {
            if alive[k].is_empty() {
                continue;
            }

            let pair_val_k = |ci: usize, ck: usize| {
                if is_left_k {
                    pair_e.get(edge_k, ci, ck)
                } else {
                    pair_e.get(edge_k, ck, ci)
                }
            };

            let all_vk_eliminated = alive[k].iter().all(|&vk| {
                competitors.iter().enumerate().any(|(ci, &r)| {
                    let ex = sum_all[ci] - min_pair[ci][ki] + pair_val_k(s, vk) - pair_val_k(r, vk);
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
    alive: &mut [Vec<usize>],
) {
    let n = self_e.n_slots();

    let fixed: Vec<(usize, usize)> = (0..n)
        .filter_map(|s| {
            if alive[s].len() == 1 {
                Some((s, alive[s][0]))
            } else {
                None
            }
        })
        .collect();

    if fixed.is_empty() {
        return;
    }

    for &(fi, best_rot) in &fixed {
        for (j, edge, is_left) in graph.neighbor_edges(fi) {
            let j = j as usize;
            let edge = edge as usize;

            if alive[j].len() <= 1 {
                continue;
            }

            for &rj in &alive[j] {
                let pair_val = if is_left {
                    pair_e.get(edge, best_rot, rj)
                } else {
                    pair_e.get(edge, rj, best_rot)
                };

                if pair_val != 0.0 {
                    let v = self_e.get(j, rj);
                    self_e.set(j, rj, v + pair_val);
                }
            }
        }
    }

    for &(fi, _) in &fixed {
        alive[fi].clear();
    }
}
