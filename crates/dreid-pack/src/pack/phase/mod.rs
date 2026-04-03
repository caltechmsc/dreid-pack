mod dee;
mod dp;
mod graph;
mod pair;
mod prune;
mod sample;

use super::config::PackConfig;
use super::constant::max_interaction_cutoff;
use super::model::fixed::FixedAtoms;
use super::progress::Progress;
use crate::model::system::System;

/// Packs all mobile side chains in [`System`] to their Global Minimum Energy
/// Conformation (GMEC).
///
/// `P` drives phase-level progress callbacks; use `pack::<()>` for zero-cost no-op tracking.
pub fn pack<P: Progress + Default + Sync>(system: &mut System, config: &PackConfig) {
    let progress = P::default();

    if system.mobile.is_empty() {
        return;
    }

    let cutoff = max_interaction_cutoff(config.electrostatics);

    // Phases 1, 2 & 3-pre: conformer sampling, contact-graph construction, and
    // fixed-atom spatial index are all mutually independent — run in parallel.
    let (((mut conformations, bias), contact_graph), fixed) = rayon::join(
        || {
            rayon::join(
                || {
                    progress.sampling_begin();

                    let (conformations, bias) = sample::sample(
                        &system.mobile,
                        config.rotamer_prob_cutoff,
                        config.sample_polar_h,
                        config.include_input_conformation,
                    );

                    let (total, max, min) = conformations.iter().fold(
                        (0usize, 0usize, usize::MAX),
                        |(t, mx, mn), c| {
                            let n = c.n_candidates();
                            (t + n, mx.max(n), mn.min(n))
                        },
                    );
                    progress.sampling_done(total, max, min);

                    (conformations, bias)
                },
                || {
                    progress.graph_begin();

                    let graph = graph::build(&system.mobile, cutoff);

                    let (degree_max, n_isolated) =
                        (0..graph.n_slots()).fold((0usize, 0usize), |(dm, ni), s| {
                            let d = graph.neighbor_edges(s).count();
                            (dm.max(d), ni + (d == 0) as usize)
                        });
                    progress.graph_done(graph.n_edges(), degree_max, n_isolated);

                    graph
                },
            )
        },
        || FixedAtoms::build(&system.fixed, cutoff),
    );

    // Phase 3: self-energies (SC <-> fixed + rotamer preference) and in-place
    // compaction of provably dead conformers.
    progress.prune_begin();

    let mut self_e = prune::prune(
        &system.mobile,
        &mut conformations,
        &fixed,
        &system.ff,
        config.electrostatics,
        &bias,
        config.self_energy_threshold,
    );

    let (conf_after, trivial_prune) =
        conformations
            .iter()
            .fold((0usize, 0usize), |(after, trivial), c| {
                let n = c.n_candidates();
                (after + n, trivial + (n == 1) as usize)
            });
    progress.prune_done(conf_after, trivial_prune);

    // Phase 4: pair energies (SC <-> SC) for every edge in the contact graph.
    progress.pair_begin();

    let pair_e = pair::pair(
        &system.mobile,
        &conformations,
        &contact_graph,
        &system.ff,
        config.electrostatics,
    );

    let n_edges = pair_e.n_edges();
    let total_entries: usize = (0..n_edges)
        .map(|e| {
            let (ni, nj) = pair_e.dims(e);
            ni * nj
        })
        .sum();
    progress.pair_done(n_edges, total_entries);

    // Phase 5: Dead-End Elimination — eliminates rotamers provably absent from
    // the GMEC, pruning self energies in-place.
    progress.dee_begin();

    let eliminated = dee::dee(&mut self_e, &pair_e, &contact_graph);

    let trivial_dee = (0..self_e.n_slots())
        .filter(|&s| {
            (0..self_e.n_candidates(s))
                .filter(|&r| !self_e.is_pruned(s, r))
                .count()
                == 1
        })
        .count();
    progress.dee_done(eliminated, trivial_dee);

    // Phase 6: tree-decomposition DP over the pruned interaction graph —
    // returns one best candidate index per slot.
    progress.dp_begin();

    let best = dp::dp(&mut self_e, &pair_e, &contact_graph);

    progress.dp_done();

    // Write-back: overwrite each mobile residue's side-chain coordinates with
    // its GMEC conformation. Each write is ≤ 18 × Vec3 (≤ 216 bytes).
    for (residue, (confs, &bi)) in system
        .mobile
        .iter_mut()
        .zip(conformations.iter().zip(best.iter()))
    {
        residue.set_sidechain(confs.coords_of(bi));
    }
}
