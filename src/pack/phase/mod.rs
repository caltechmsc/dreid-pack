mod dee;
mod dp;
mod graph;
mod pair;
mod prune;
mod sample;

use super::config::PackConfig;
use super::constant::max_interaction_cutoff;
use super::model::fixed::FixedAtoms;
use crate::model::system::System;

/// Packs all mobile side chains in [`System`] to their Global Minimum Energy
/// Conformation (GMEC).
pub fn pack(system: &mut System, config: &PackConfig) {
    if system.mobile.is_empty() {
        return;
    }

    let cutoff = max_interaction_cutoff(config.electrostatics);

    // Phases 1, 2 & 3-pre: conformer sampling, contact-graph construction, and
    // fixed-atom spatial index are all mutually independent — run in parallel.
    let ((mut conformations, contact_graph), fixed) = rayon::join(
        || {
            rayon::join(
                || {
                    sample::sample(
                        &system.mobile,
                        config.rotamer_prob_cutoff,
                        config.sample_polar_h,
                        config.include_input_conformation,
                    )
                },
                || graph::build(&system.mobile, cutoff),
            )
        },
        || FixedAtoms::build(&system.fixed, cutoff),
    );

    // Phase 3: self-energies (SC <-> fixed) and in-place compaction of provably
    // dead conformers.
    let mut self_e = prune::prune(
        &system.mobile,
        &mut conformations,
        &fixed,
        &system.ff,
        config.electrostatics,
        config.self_energy_threshold,
    );

    // Phase 4: pair energies (SC <-> SC) for every edge in the contact graph.
    let pair_e = pair::pair(
        &system.mobile,
        &conformations,
        &contact_graph,
        &system.ff,
        config.electrostatics,
    );

    // Phase 5: Dead-End Elimination — eliminates rotamers provably absent from
    // the GMEC, pruning self energies in-place.
    dee::dee(&mut self_e, &pair_e, &contact_graph);

    // Phase 6: tree-decomposition DP over the pruned interaction graph —
    // returns one best candidate index per slot.
    let best = dp::dp(&mut self_e, &pair_e, &contact_graph);

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
