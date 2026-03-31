/// Progress callbacks for the packing pipeline.
pub trait Progress {
    fn sampling_begin(&self);
    fn sampling_done(&self, total: usize, max: usize, min: usize);

    fn graph_begin(&self);
    fn graph_done(&self, n_edges: usize, degree_max: usize, n_isolated: usize);

    fn prune_begin(&self);
    fn prune_done(&self, after: usize, trivial: usize);

    fn pair_begin(&self);
    fn pair_done(&self, n_edges: usize, total_entries: usize);

    fn dee_begin(&self);
    fn dee_done(&self, eliminated: usize, trivial: usize);

    fn dp_begin(&self);
    fn dp_done(&self);
}

/// No-op progress — zero cost after monomorphization.
impl Progress for () {
    fn sampling_begin(&self) {}
    fn sampling_done(&self, _: usize, _: usize, _: usize) {}
    fn graph_begin(&self) {}
    fn graph_done(&self, _: usize, _: usize, _: usize) {}
    fn prune_begin(&self) {}
    fn prune_done(&self, _: usize, _: usize) {}
    fn pair_begin(&self) {}
    fn pair_done(&self, _: usize, _: usize) {}
    fn dee_begin(&self) {}
    fn dee_done(&self, _: usize, _: usize) {}
    fn dp_begin(&self) {}
    fn dp_done(&self) {}
}
