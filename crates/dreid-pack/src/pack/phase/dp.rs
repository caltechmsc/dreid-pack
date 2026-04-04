use crate::pack::model::{
    energy::{PRUNED, PairEnergyTable, SelfEnergyTable},
    graph::ContactGraph,
};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Pair-energy significance threshold (kcal/mol). Edges where every surviving
/// candidate pair has `|pair_e| ≤ PAIR_CUT` are dropped from the work graph.
const PAIR_CUT: f32 = 2.0;

/// Maximum treewidth for exact tree-decomposition DP. Components exceeding
/// this trigger edge decomposition to reduce treewidth.
const TREEWIDTH_CUT: usize = 5;

/// Initial threshold for rank-1 edge decomposition.
const EDGE_DECOMP_THRESHOLD_INIT: f32 = 0.5;

/// Finds the Global Minimum Energy Conformation (GMEC) for remaining
/// multi-candidate slots via tree-decomposition dynamic programming.
/// Returns one rotamer index per slot.
pub fn dp(
    self_e: &mut SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
) -> Vec<usize> {
    let n = self_e.n_slots();
    debug_assert_eq!(graph.n_slots(), n);

    let mut best = vec![0usize; n];

    let mut multi: Vec<usize> = Vec::new();
    for (s, best_s) in best.iter_mut().enumerate() {
        let alive = alive_indices(self_e, s);
        debug_assert!(!alive.is_empty(), "slot {s} has zero surviving candidates");
        if alive.len() == 1 {
            *best_s = alive[0];
        } else {
            multi.push(s);
        }
    }

    if multi.is_empty() {
        return best;
    }

    let n_multi = multi.len();
    let mut inv = vec![u32::MAX; n];
    for (li, &gi) in multi.iter().enumerate() {
        inv[gi] = li as u32;
    }

    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n_multi];
    let mut sig_edges: Vec<(u32, u32, u32)> = Vec::new();

    for (edge_idx, &(ga, gb)) in graph.edges().iter().enumerate() {
        let (ga, gb) = (ga as usize, gb as usize);
        let la = inv[ga];
        let lb = inv[gb];
        if la == u32::MAX || lb == u32::MAX {
            continue;
        }

        if is_significant(self_e, pair_e, edge_idx, ga, gb) {
            adj[la as usize].push(lb);
            adj[lb as usize].push(la);
            sig_edges.push((la, lb, edge_idx as u32));
        }
    }

    let mut in_component = vec![false; n_multi];
    for li in 0..n_multi {
        if adj[li].is_empty() {
            best[multi[li]] = best_by_self(self_e, multi[li]);
        } else {
            in_component[li] = true;
        }
    }

    let components = find_components(n_multi, &adj, &in_component);

    if components.is_empty() {
        return best;
    }

    let comp_results: Vec<Vec<(usize, usize)>> = components
        .par_iter()
        .map(|comp| solve_component(comp, &multi, &sig_edges, self_e, pair_e, graph))
        .collect();

    for assignments in comp_results {
        for (slot, rot) in assignments {
            best[slot] = rot;
        }
    }

    best
}

/// Returns the global indices of non-pruned candidates at slot `s`.
fn alive_indices(self_e: &SelfEnergyTable, s: usize) -> Vec<usize> {
    (0..self_e.n_candidates(s))
        .filter(|&r| !self_e.is_pruned(s, r))
        .collect()
}

/// Returns the alive candidate with the lowest self-energy.
fn best_by_self(self_e: &SelfEnergyTable, s: usize) -> usize {
    let mut best_r = 0;
    let mut best_e = PRUNED;
    for r in 0..self_e.n_candidates(s) {
        let e = self_e.get(s, r);
        if e < best_e {
            best_e = e;
            best_r = r;
        }
    }
    best_r
}

/// Returns `true` if any alive pair has `|pair_e| > PAIR_CUT`.
fn is_significant(
    self_e: &SelfEnergyTable,
    pair_e: &PairEnergyTable,
    edge_idx: usize,
    ga: usize,
    gb: usize,
) -> bool {
    let mat = pair_e.matrix(edge_idx);
    let stride = pair_e.dims(edge_idx).1;
    let nc_a = self_e.n_candidates(ga);
    let nc_b = self_e.n_candidates(gb);

    (0..nc_a).any(|ra| {
        if self_e.is_pruned(ga, ra) {
            return false;
        }
        (0..nc_b).any(|rb| {
            if self_e.is_pruned(gb, rb) {
                return false;
            }
            let v = mat[ra * stride + rb];
            !(-PAIR_CUT..=PAIR_CUT).contains(&v)
        })
    })
}

/// Pair-energy lookup for `(ci_at_slot_i, cj_at_neighbor)` on `edge`.
fn pair_val(mat: &[f32], stride: usize, is_left: bool, ci: usize, cj: usize) -> f32 {
    if is_left {
        mat[ci * stride + cj]
    } else {
        mat[cj * stride + ci]
    }
}

/// Finds connected components in the work graph, returning component-local indices.
fn find_components(n: usize, adj: &[Vec<u32>], active: &[bool]) -> Vec<Vec<u32>> {
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if !active[start] || visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start as u32];
        visited[start] = true;
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &v in &adj[u as usize] {
                if !visited[v as usize] {
                    visited[v as usize] = true;
                    stack.push(v);
                }
            }
        }
        components.push(comp);
    }

    components
}

/// Solves a single connected component. Returns `(global_slot, best_rot)` pairs.
fn solve_component(
    comp: &[u32],
    multi: &[usize],
    global_sig_edges: &[(u32, u32, u32)],
    self_e: &SelfEnergyTable,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
) -> Vec<(usize, usize)> {
    let cn = comp.len();
    let mut comp_inv = vec![u32::MAX; multi.len()];
    for (ci, &li) in comp.iter().enumerate() {
        comp_inv[li as usize] = ci as u32;
    }

    let gi = |ci: usize| -> usize { multi[comp[ci] as usize] };

    let mut work_edges: Vec<(u32, u32, u32)> = Vec::new();
    for &(la, lb, eidx) in global_sig_edges {
        let ca = comp_inv[la as usize];
        let cb = comp_inv[lb as usize];
        if ca == u32::MAX || cb == u32::MAX {
            continue;
        }
        let (ca, cb) = if ca < cb { (ca, cb) } else { (cb, ca) };
        work_edges.push((ca, cb, eidx));
    }
    work_edges.sort_unstable();
    work_edges.dedup();

    let mut adj = vec![Vec::<u32>::new(); cn];
    for &(a, b, _) in &work_edges {
        adj[a as usize].push(b);
        adj[b as usize].push(a);
    }

    let mut local_self: Vec<Vec<f32>> = (0..cn)
        .map(|ci| {
            let s = gi(ci);
            (0..self_e.n_candidates(s))
                .map(|r| self_e.get(s, r))
                .collect()
        })
        .collect();

    let mut result = vec![0usize; cn];
    let mut threshold = EDGE_DECOMP_THRESHOLD_INIT;

    loop {
        if let Some((bags, _)) = eliminate(cn, &adj, TREEWIDTH_CUT) {
            let tree = root_tree(&bags);
            tree_dp(
                &tree,
                &local_self,
                pair_e,
                graph,
                gi,
                &work_edges,
                &mut result,
            );
            break;
        }

        edge_decompose(
            &mut adj,
            &mut work_edges,
            &mut local_self,
            gi,
            pair_e,
            graph,
            threshold,
        );
        threshold *= 2.0;

        for ci in 0..cn {
            if adj[ci].is_empty() && result[ci] == 0 {
                result[ci] = min_energy_rot(&local_self[ci]);
            }
        }
    }

    (0..cn).map(|ci| (gi(ci), result[ci])).collect()
}

/// Returns the index of the rotamer with minimum finite energy.
fn min_energy_rot(se: &[f32]) -> usize {
    let mut best_r = 0;
    let mut best_e = PRUNED;
    for (r, &e) in se.iter().enumerate() {
        if e < best_e {
            best_e = e;
            best_r = r;
        }
    }
    best_r
}

/// Decomposes edges whose pair energy is well-approximated by a rank-1
/// factorization `pair(m,n) ≈ a[m] + b[n]`, folding marginals into self-energy.
fn edge_decompose(
    adj: &mut [Vec<u32>],
    work_edges: &mut Vec<(u32, u32, u32)>,
    local_self: &mut [Vec<f32>],
    gi: impl Fn(usize) -> usize,
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    threshold: f32,
) {
    let mut to_remove = Vec::new();

    for (we_idx, &(ca, cb, eidx)) in work_edges.iter().enumerate() {
        let sa = gi(ca as usize);

        let eidx = eidx as usize;
        let mat = pair_e.matrix(eidx);
        let stride = pair_e.dims(eidx).1;
        let is_left = (graph.edges()[eidx].0 as usize) == sa;

        let se_a = &local_self[ca as usize];
        let se_b = &local_self[cb as usize];

        let alive_a: Vec<usize> = se_a
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e != PRUNED)
            .map(|(r, _)| r)
            .collect();
        let alive_b: Vec<usize> = se_b
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e != PRUNED)
            .map(|(r, _)| r)
            .collect();

        let na = alive_a.len();
        let nb = alive_b.len();
        if na == 0 || nb == 0 {
            continue;
        }

        let mut sum_all = 0.0f32;
        for &m in &alive_a {
            for &n in &alive_b {
                sum_all += pair_val(mat, stride, is_left, m, n);
            }
        }
        let a_bar = sum_all / (2.0 * na as f32 * nb as f32);

        let mut ak = vec![PRUNED; se_a.len()];
        for &m in &alive_a {
            let mut s = 0.0f32;
            for &n in &alive_b {
                s += pair_val(mat, stride, is_left, m, n);
            }
            ak[m] = s / nb as f32 - a_bar;
        }

        let mut bk = vec![PRUNED; se_b.len()];
        for &n in &alive_b {
            let mut s = 0.0f32;
            for &m in &alive_a {
                s += pair_val(mat, stride, is_left, m, n);
            }
            bk[n] = s / na as f32 - a_bar;
        }

        let mut maxdev = 0.0f32;
        for &m in &alive_a {
            for &n in &alive_b {
                let dev = (pair_val(mat, stride, is_left, m, n) - ak[m] - bk[n]).abs();
                if dev > maxdev {
                    maxdev = dev;
                }
            }
        }

        if maxdev <= threshold {
            for &m in &alive_a {
                local_self[ca as usize][m] += ak[m];
            }
            for &n in &alive_b {
                local_self[cb as usize][n] += bk[n];
            }
            to_remove.push(we_idx);
        }
    }

    for &idx in to_remove.iter().rev() {
        let (ca, cb, _) = work_edges.remove(idx);
        adj[ca as usize].retain(|&x| x != cb);
        adj[cb as usize].retain(|&x| x != ca);
    }
}

/// Maximum Cardinality Search. Produces a PEO when the graph is chordal.
fn mcs_order(matrix: &[bool], n: usize) -> Vec<usize> {
    let mut numbered = vec![false; n];
    let mut card = vec![0u32; n];
    let mut sigma = vec![0usize; n];

    for i in (1..=n).rev() {
        let v = (0..n)
            .filter(|&v| !numbered[v])
            .max_by_key(|&v| (card[v], Reverse(v)))
            .unwrap();
        sigma[v] = i;
        numbered[v] = true;
        for u in 0..n {
            if !numbered[u] && matrix[v * n + u] {
                card[u] += 1;
            }
        }
    }

    let mut order = vec![0usize; n];
    for v in 0..n {
        order[sigma[v] - 1] = v;
    }
    order
}

/// Returns `true` if `order` is a Perfect Elimination Ordering: for each
/// vertex, its first later neighbor is adjacent to all other later neighbors.
fn is_peo(matrix: &[bool], n: usize, order: &[usize]) -> bool {
    let mut pos = vec![0usize; n];
    for (i, &v) in order.iter().enumerate() {
        pos[v] = i;
    }

    for (i, &v) in order.iter().enumerate() {
        let mut f = usize::MAX;
        let mut f_pos = usize::MAX;
        for u in 0..n {
            if matrix[v * n + u] && pos[u] > i && pos[u] < f_pos {
                f_pos = pos[u];
                f = u;
            }
        }
        if f == usize::MAX {
            continue;
        }
        for u in 0..n {
            if u != f && matrix[v * n + u] && pos[u] > i && !matrix[f * n + u] {
                return false;
            }
        }
    }
    true
}

/// A bag produced by the elimination ordering.
struct Bag {
    elim: u32,
    sep: Vec<u32>,
}

/// Builds elimination bags using the best available heuristic (MCS for chordal, min-fill otherwise).
fn eliminate(n: usize, adj: &[Vec<u32>], max_width: usize) -> Option<(Vec<Bag>, usize)> {
    if n == 0 {
        return Some((Vec::new(), 0));
    }

    let mut matrix = vec![false; n * n];
    for (v, nbrs) in adj.iter().enumerate() {
        for &u in nbrs {
            matrix[v * n + u as usize] = true;
        }
    }

    let mcs = mcs_order(&matrix, n);
    if is_peo(&matrix, n, &mcs) {
        return build_bags(&mut matrix, n, &mcs, max_width);
    }

    build_bags_min_fill(&mut matrix, n, max_width)
}

/// Builds bags following a predetermined elimination ordering.
fn build_bags(
    matrix: &mut [bool],
    n: usize,
    order: &[usize],
    max_width: usize,
) -> Option<(Vec<Bag>, usize)> {
    let mut eliminated = vec![false; n];
    let mut bags = Vec::with_capacity(n);
    let mut width = 0;

    for &v in order {
        let sep = collect_sep(matrix, n, v, &eliminated);
        if sep.len() > max_width {
            return None;
        }
        width = width.max(sep.len());
        apply_fill_in(matrix, n, &sep);
        eliminated[v] = true;
        bags.push(Bag {
            elim: v as u32,
            sep,
        });
    }

    Some((bags, width))
}

/// Builds bags with incremental min-fill vertex selection using a priority queue.
fn build_bags_min_fill(
    matrix: &mut [bool],
    n: usize,
    max_width: usize,
) -> Option<(Vec<Bag>, usize)> {
    let mut eliminated = vec![false; n];
    let mut bags = Vec::with_capacity(n);
    let mut width = 0;

    let mut fill_cost: Vec<u32> = (0..n)
        .map(|v| compute_fill(matrix, n, v, &eliminated))
        .collect();
    let mut epoch = vec![0u32; n];

    let mut heap: BinaryHeap<_> = fill_cost
        .iter()
        .enumerate()
        .map(|(v, &cost)| Reverse((cost, v as u32, 0u32)))
        .collect();

    for _ in 0..n {
        let v = loop {
            let Reverse((cost, v, g)) = heap.pop().unwrap();
            let v = v as usize;
            if eliminated[v] || g != epoch[v] {
                continue;
            }
            if cost != fill_cost[v] {
                epoch[v] = epoch[v].wrapping_add(1);
                heap.push(Reverse((fill_cost[v], v as u32, epoch[v])));
                continue;
            }
            break v;
        };

        let sep = collect_sep(matrix, n, v, &eliminated);
        if sep.len() > max_width {
            return None;
        }
        width = width.max(sep.len());

        let mut affected = Vec::new();
        for i in 0..sep.len() {
            for j in (i + 1)..sep.len() {
                let (a, b) = (sep[i] as usize, sep[j] as usize);
                if !matrix[a * n + b] {
                    matrix[a * n + b] = true;
                    matrix[b * n + a] = true;
                    affected.push(a);
                    affected.push(b);
                }
            }
        }

        eliminated[v] = true;
        bags.push(Bag {
            elim: v as u32,
            sep,
        });

        let mut dirty: Vec<usize> = Vec::new();
        for u in 0..n {
            if !eliminated[u] && matrix[v * n + u] {
                dirty.push(u);
            }
        }
        dirty.extend_from_slice(&affected);
        dirty.sort_unstable();
        dirty.dedup();

        for &u in &dirty {
            if eliminated[u] {
                continue;
            }
            let new_cost = compute_fill(matrix, n, u, &eliminated);
            if new_cost != fill_cost[u] {
                fill_cost[u] = new_cost;
                epoch[u] = epoch[u].wrapping_add(1);
                heap.push(Reverse((new_cost, u as u32, epoch[u])));
            }
        }
    }

    Some((bags, width))
}

/// Counts the number of missing edges among alive neighbors of `v`.
fn compute_fill(matrix: &[bool], n: usize, v: usize, eliminated: &[bool]) -> u32 {
    let row = &matrix[v * n..(v + 1) * n];
    let mut fill = 0u32;
    for a in 0..n {
        if a == v || eliminated[a] || !row[a] {
            continue;
        }
        for b in (a + 1)..n {
            if b == v || eliminated[b] || !row[b] {
                continue;
            }
            if !matrix[a * n + b] {
                fill += 1;
            }
        }
    }
    fill
}

/// Ascending alive neighbours of `v` (separator at elimination time).
fn collect_sep(matrix: &[bool], n: usize, v: usize, eliminated: &[bool]) -> Vec<u32> {
    (0..n)
        .filter(|&u| u != v && !eliminated[u] && matrix[v * n + u])
        .map(|u| u as u32)
        .collect()
}

/// Makes `sep` vertices pairwise adjacent (clique fill-in).
fn apply_fill_in(matrix: &mut [bool], n: usize, sep: &[u32]) {
    for i in 0..sep.len() {
        for j in (i + 1)..sep.len() {
            let (a, b) = (sep[i] as usize, sep[j] as usize);
            matrix[a * n + b] = true;
            matrix[b * n + a] = true;
        }
    }
}

/// A node in the rooted elimination tree.
struct TreeNode {
    elim: u32,
    sep: Vec<u32>,
    children: Vec<u32>,
}

/// Connects elimination bags into a rooted tree (reversed elimination order).
/// The last-eliminated vertex becomes the root (index 0).
fn root_tree(bags: &[Bag]) -> Vec<TreeNode> {
    let n = bags.len();
    if n == 0 {
        return Vec::new();
    }

    let max_v = bags.iter().map(|b| b.elim as usize).max().unwrap_or(0) + 1;
    let mut vertex_bag = vec![0u32; max_v];
    for (i, bag) in bags.iter().enumerate() {
        vertex_bag[bag.elim as usize] = i as u32;
    }

    let mut nodes: Vec<TreeNode> = bags
        .iter()
        .rev()
        .map(|b| TreeNode {
            elim: b.elim,
            sep: b.sep.clone(),
            children: Vec::new(),
        })
        .collect();

    for (elim_i, bag) in bags.iter().enumerate().take(n - 1) {
        let parent_elim_i = bag
            .sep
            .iter()
            .map(|&v| vertex_bag[v as usize] as usize)
            .min()
            .unwrap_or(n - 1);
        let child_rev = n - 1 - elim_i;
        let parent_rev = n - 1 - parent_elim_i;
        nodes[parent_rev].children.push(child_rev as u32);
    }

    nodes
}

/// Runs the full tree-decomposition DP and writes the GMEC into `result`.
fn tree_dp(
    tree: &[TreeNode],
    local_self: &[Vec<f32>],
    pair_e: &PairEnergyTable,
    graph: &ContactGraph,
    gi: impl Fn(usize) -> usize,
    work_edges: &[(u32, u32, u32)],
    result: &mut [usize],
) {
    let n_nodes = tree.len();
    if n_nodes == 0 {
        return;
    }

    let cn = local_self.len();
    let edge_map = build_edge_map(work_edges, &gi, graph);

    let (alive_data, alive_off) = build_alive_table(local_self);

    let max_cands = local_self.iter().map(|se| se.len()).max().unwrap_or(0);
    let mut rot_to_idx = vec![u32::MAX; cn * max_cands];
    for ci in 0..cn {
        let alive = &alive_data[alive_off[ci]..alive_off[ci + 1]];
        for (idx, &r) in alive.iter().enumerate() {
            rot_to_idx[ci * max_cands + r] = idx as u32;
        }
    }

    let nodes: Vec<NodeInfo> = tree
        .iter()
        .map(|tn| NodeInfo::new(tn, &alive_off))
        .collect();

    let node_edges: Vec<Vec<EdgeInfo>> = nodes
        .iter()
        .map(|nd| {
            nd.sep_cis
                .iter()
                .enumerate()
                .filter_map(|(d, &sci)| {
                    let elim_is_lo = nd.elim_ci < sci;
                    let key = if elim_is_lo {
                        (nd.elim_ci as u32, sci as u32)
                    } else {
                        (sci as u32, nd.elim_ci as u32)
                    };
                    let &(edge_idx, lo_is_left) = edge_map.get(&key)?;
                    let is_left = if elim_is_lo { lo_is_left } else { !lo_is_left };
                    let mat = pair_e.matrix(edge_idx);
                    let stride = pair_e.dims(edge_idx).1;
                    Some(EdgeInfo {
                        sep_dim: d,
                        mat,
                        stride,
                        is_left,
                    })
                })
                .collect()
        })
        .collect();

    const ELIM_MARKER: u32 = u32::MAX;
    let child_maps: Vec<Vec<Vec<u32>>> = (0..n_nodes)
        .map(|ni| {
            tree[ni]
                .children
                .iter()
                .map(|&cni| {
                    let child_node = &nodes[cni as usize];
                    child_node
                        .sep_cis
                        .iter()
                        .map(|&c_sci| {
                            if c_sci == nodes[ni].elim_ci {
                                ELIM_MARKER
                            } else {
                                nodes[ni].sep_cis.iter().position(|&x| x == c_sci).unwrap() as u32
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut messages: Vec<Vec<f32>> = (0..n_nodes).map(|_| Vec::new()).collect();
    let mut tracebacks: Vec<Vec<usize>> = (0..n_nodes).map(|_| Vec::new()).collect();

    let topo = topo_order(tree);

    let max_sep = nodes.iter().map(|nd| nd.sep_cis.len()).max().unwrap_or(0);
    let mut sep_rots_buf = vec![0usize; max_sep];

    for &ni in &topo {
        let nd = &nodes[ni];
        let elim_alive = &alive_data[alive_off[nd.elim_ci]..alive_off[nd.elim_ci + 1]];

        let mut msg = vec![PRUNED; nd.sep_total];
        let mut tb = vec![0usize; nd.sep_total];

        let sep_rots = &mut sep_rots_buf[..nd.sep_cis.len()];

        for sep_flat in 0..nd.sep_total {
            for (d, &ci) in nd.sep_cis.iter().enumerate() {
                let alive = &alive_data[alive_off[ci]..alive_off[ci + 1]];
                sep_rots[d] = alive[(sep_flat / nd.sep_strides[d]) % nd.sep_dims[d]];
            }

            let mut best_e = PRUNED;
            let mut best_r = 0usize;

            for &er in elim_alive {
                let mut e = local_self[nd.elim_ci][er];

                for ei in &node_edges[ni] {
                    e += pair_val(ei.mat, ei.stride, ei.is_left, er, sep_rots[ei.sep_dim]);
                }

                for (ci_idx, &child_ni) in tree[ni].children.iter().enumerate() {
                    let child_nd = &nodes[child_ni as usize];
                    let mapping = &child_maps[ni][ci_idx];

                    let mut child_flat = 0usize;
                    for (d, &src) in mapping.iter().enumerate() {
                        let rot = if src == ELIM_MARKER {
                            er
                        } else {
                            sep_rots[src as usize]
                        };
                        let alive_idx = rot_to_idx[child_nd.sep_cis[d] * max_cands + rot] as usize;
                        child_flat += alive_idx * child_nd.sep_strides[d];
                    }

                    e += messages[child_ni as usize][child_flat];
                }

                if e < best_e {
                    best_e = e;
                    best_r = er;
                }
            }

            msg[sep_flat] = best_e;
            tb[sep_flat] = best_r;
        }

        messages[ni] = msg;
        tracebacks[ni] = tb;
    }

    debug_assert!(
        nodes[0].sep_cis.is_empty(),
        "root separator must be empty in a valid elimination tree"
    );

    result[nodes[0].elim_ci] = tracebacks[0][0];

    backtrack(
        tree,
        &nodes,
        &child_maps,
        &rot_to_idx,
        max_cands,
        &tracebacks,
        result,
    );
}

/// Top-down backtracking: assigns rotamers from root to leaves.
fn backtrack(
    tree: &[TreeNode],
    nodes: &[NodeInfo],
    child_maps: &[Vec<Vec<u32>>],
    rot_to_idx: &[u32],
    max_cands: usize,
    tracebacks: &[Vec<usize>],
    result: &mut [usize],
) {
    const ELIM_MARKER: u32 = u32::MAX;

    let mut stack: Vec<(usize, usize)> = Vec::new();

    for (ci_idx, &child_ni) in tree[0].children.iter().enumerate() {
        stack.push((0, ci_idx));
        let _ = child_ni;
    }

    while let Some((parent_ni, ci_idx)) = stack.pop() {
        let child_ni = tree[parent_ni].children[ci_idx] as usize;
        let child_nd = &nodes[child_ni];
        let mapping = &child_maps[parent_ni][ci_idx];

        let parent_elim_ci = nodes[parent_ni].elim_ci;
        let mut child_flat = 0usize;
        for (d, &src) in mapping.iter().enumerate() {
            let rot = if src == ELIM_MARKER {
                result[parent_elim_ci]
            } else {
                result[nodes[parent_ni].sep_cis[src as usize]]
            };
            let alive_idx = rot_to_idx[child_nd.sep_cis[d] * max_cands + rot] as usize;
            child_flat += alive_idx * child_nd.sep_strides[d];
        }

        result[child_nd.elim_ci] = tracebacks[child_ni][child_flat];

        for (grandchild_idx, _) in tree[child_ni].children.iter().enumerate() {
            stack.push((child_ni, grandchild_idx));
        }
    }
}

/// Pre-computed per-node info for the DP inner loop.
struct NodeInfo {
    elim_ci: usize,
    sep_cis: Vec<usize>,
    sep_dims: Vec<usize>,
    sep_strides: Vec<usize>,
    sep_total: usize,
}

impl NodeInfo {
    fn new(tn: &TreeNode, alive_off: &[usize]) -> Self {
        let sep_cis: Vec<usize> = tn.sep.iter().map(|&v| v as usize).collect();
        let sep_dims: Vec<usize> = sep_cis
            .iter()
            .map(|&ci| alive_off[ci + 1] - alive_off[ci])
            .collect();
        let sep_total = sep_dims.iter().copied().product::<usize>().max(1);

        let mut sep_strides = vec![1usize; sep_dims.len()];
        for i in (0..sep_dims.len().saturating_sub(1)).rev() {
            sep_strides[i] = sep_strides[i + 1] * sep_dims[i + 1];
        }

        Self {
            elim_ci: tn.elim as usize,
            sep_cis,
            sep_dims,
            sep_strides,
            sep_total,
        }
    }
}

/// Cached edge info for the elim<->sep pair energy in the DP inner loop.
struct EdgeInfo<'a> {
    sep_dim: usize,
    mat: &'a [f32],
    stride: usize,
    is_left: bool,
}

/// Builds a HashMap for O(1) edge lookup between component-local slot pairs.
fn build_edge_map(
    work_edges: &[(u32, u32, u32)],
    gi: impl Fn(usize) -> usize,
    graph: &ContactGraph,
) -> HashMap<(u32, u32), (usize, bool)> {
    let mut map = HashMap::with_capacity(work_edges.len());
    for &(ca, cb, eidx) in work_edges {
        let eidx = eidx as usize;
        let (ga, _) = graph.edges()[eidx];
        let ca_is_left = gi(ca as usize) == ga as usize;
        let key = (ca.min(cb), ca.max(cb));
        map.insert(key, (eidx, ca_is_left));
    }
    map
}

/// Builds a flat alive-rotamer table: `alive_data[off[ci]..off[ci+1]]` holds
/// the alive rotamer indices for component-local slot `ci`.
fn build_alive_table(local_self: &[Vec<f32>]) -> (Vec<usize>, Vec<usize>) {
    let cn = local_self.len();
    let mut offsets = vec![0usize; cn + 1];
    for (ci, se) in local_self.iter().enumerate() {
        offsets[ci + 1] = offsets[ci] + se.iter().filter(|&&e| e != PRUNED).count();
    }
    let mut data = vec![0usize; offsets[cn]];
    for (ci, se) in local_self.iter().enumerate() {
        let mut pos = offsets[ci];
        for (r, &e) in se.iter().enumerate() {
            if e != PRUNED {
                data[pos] = r;
                pos += 1;
            }
        }
    }
    (data, offsets)
}

/// Returns a bottom-up topological order of the elimination tree nodes.
fn topo_order(tree: &[TreeNode]) -> Vec<usize> {
    let n = tree.len();
    if n == 0 {
        return Vec::new();
    }

    let mut order = Vec::with_capacity(n);
    let mut queue = std::collections::VecDeque::with_capacity(n);
    queue.push_back(0usize);
    while let Some(ni) = queue.pop_front() {
        order.push(ni);
        for &child in &tree[ni].children {
            queue.push_back(child as usize);
        }
    }
    order.reverse();
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_matrix(n: usize, edges: &[(usize, usize)]) -> Vec<bool> {
        let mut m = vec![false; n * n];
        for &(a, b) in edges {
            m[a * n + b] = true;
            m[b * n + a] = true;
        }
        m
    }

    fn make_adj(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<u32>> {
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a].push(b as u32);
            adj[b].push(a as u32);
        }
        adj
    }

    fn two_slot(
        counts: [u16; 2],
        pair_data: &[f32],
    ) -> (SelfEnergyTable, PairEnergyTable, ContactGraph) {
        let self_e = SelfEnergyTable::new(&counts);
        let mut pair_e = PairEnergyTable::new(&[(counts[0], counts[1])]);
        let graph = ContactGraph::build(2, [(0u32, 1u32)]);
        let (ni, nj) = (counts[0] as usize, counts[1] as usize);
        for ri in 0..ni {
            for rj in 0..nj {
                pair_e.set(0, ri, rj, pair_data[ri * nj + rj]);
            }
        }
        (self_e, pair_e, graph)
    }

    #[test]
    fn mcs_on_path_produces_peo() {
        let m = make_matrix(3, &[(0, 1), (1, 2)]);
        let order = mcs_order(&m, 3);
        assert!(is_peo(&m, 3, &order));
    }

    #[test]
    fn mcs_on_triangle_produces_peo() {
        let m = make_matrix(3, &[(0, 1), (0, 2), (1, 2)]);
        let order = mcs_order(&m, 3);
        assert!(is_peo(&m, 3, &order));
    }

    #[test]
    fn mcs_on_four_cycle_fails_peo() {
        let m = make_matrix(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let order = mcs_order(&m, 4);
        assert!(!is_peo(&m, 4, &order));
    }

    #[test]
    fn is_peo_rejects_non_peo_order_on_chordal_graph() {
        let m = make_matrix(3, &[(0, 1), (1, 2)]);
        assert!(is_peo(&m, 3, &[0, 1, 2]));
        assert!(!is_peo(&m, 3, &[1, 0, 2]));
    }

    #[test]
    fn eliminate_empty_graph_returns_zero_width() {
        let (bags, width) = eliminate(0, &[], 5).unwrap();
        assert!(bags.is_empty());
        assert_eq!(width, 0);
    }

    #[test]
    fn eliminate_chordal_path_produces_width_one() {
        let adj = make_adj(3, &[(0, 1), (1, 2)]);
        let (bags, width) = eliminate(3, &adj, 5).unwrap();
        assert_eq!(bags.len(), 3);
        assert_eq!(width, 1);
    }

    #[test]
    fn eliminate_non_chordal_cycle_uses_min_fill() {
        let adj = make_adj(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let (bags, width) = eliminate(4, &adj, 5).unwrap();
        assert_eq!(bags.len(), 4);
        assert_eq!(width, 2);
    }

    #[test]
    fn eliminate_returns_none_when_width_exceeds_cutoff() {
        let adj = make_adj(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        assert!(eliminate(4, &adj, 1).is_none());
    }

    #[test]
    fn eliminate_isolated_vertices_produce_empty_separators() {
        let adj: Vec<Vec<u32>> = vec![vec![], vec![], vec![]];
        let (bags, width) = eliminate(3, &adj, 0).unwrap();
        assert_eq!(bags.len(), 3);
        assert_eq!(width, 0);
        for bag in &bags {
            assert!(bag.sep.is_empty());
        }
    }

    #[test]
    fn root_tree_single_bag_has_no_children() {
        let bags = vec![Bag {
            elim: 0,
            sep: vec![],
        }];
        let tree = root_tree(&bags);
        assert_eq!(tree.len(), 1);
        assert!(tree[0].children.is_empty());
        assert!(tree[0].sep.is_empty());
    }

    #[test]
    fn root_tree_chain_preserves_parent_child_structure() {
        let bags = vec![
            Bag {
                elim: 2,
                sep: vec![1],
            },
            Bag {
                elim: 1,
                sep: vec![0],
            },
            Bag {
                elim: 0,
                sep: vec![],
            },
        ];
        let tree = root_tree(&bags);
        assert_eq!(tree.len(), 3);
        assert_eq!(tree[0].elim, 0);
        assert_eq!(tree[0].children, [1]);
        assert_eq!(tree[1].elim, 1);
        assert_eq!(tree[1].children, [2]);
        assert!(tree[2].children.is_empty());
    }

    #[test]
    fn root_tree_root_has_empty_separator() {
        let adj = make_adj(4, &[(0, 1), (1, 2), (2, 3)]);
        let (bags, _) = eliminate(4, &adj, 5).unwrap();
        let tree = root_tree(&bags);
        assert!(tree[0].sep.is_empty());
    }

    #[test]
    fn edge_decompose_removes_exact_rank1_edge() {
        let (_self_e, pair_e, graph) = two_slot([2, 2], &[1.0, 3.0, 3.0, 5.0]);

        let mut adj = vec![vec![1u32], vec![0u32]];
        let mut work_edges = vec![(0u32, 1u32, 0u32)];
        let mut local_self = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        edge_decompose(
            &mut adj,
            &mut work_edges,
            &mut local_self,
            |ci| ci,
            &pair_e,
            &graph,
            0.5,
        );

        assert!(work_edges.is_empty(), "rank-1 edge should be removed");
        assert!(adj[0].is_empty());

        assert_abs_diff_eq!(local_self[0][0], 0.5, epsilon = 1e-5);
        assert_abs_diff_eq!(local_self[0][1], 2.5, epsilon = 1e-5);
        assert_abs_diff_eq!(local_self[1][0], 0.5, epsilon = 1e-5);
        assert_abs_diff_eq!(local_self[1][1], 2.5, epsilon = 1e-5);
    }

    #[test]
    fn edge_decompose_keeps_non_rank1_edge() {
        let (_self_e, pair_e, graph) = two_slot([2, 2], &[10.0, 0.0, 0.0, 10.0]);

        let mut adj = vec![vec![1u32], vec![0u32]];
        let mut work_edges = vec![(0u32, 1u32, 0u32)];
        let mut local_self = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        edge_decompose(
            &mut adj,
            &mut work_edges,
            &mut local_self,
            |ci| ci,
            &pair_e,
            &graph,
            0.5,
        );

        assert_eq!(work_edges.len(), 1, "non-rank-1 edge must survive");
        assert_abs_diff_eq!(local_self[0][0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn dp_all_single_candidate_returns_zeros() {
        let mut self_e = SelfEnergyTable::new(&[1, 1]);
        let pair_e = PairEnergyTable::new(&[(1, 1)]);
        let graph = ContactGraph::build(2, [(0u32, 1u32)]);

        assert_eq!(dp(&mut self_e, &pair_e, &graph), [0, 0]);
    }

    #[test]
    fn dp_independent_slots_pick_minimum_self_energy() {
        let mut self_e = SelfEnergyTable::new(&[3, 3]);
        self_e.set(0, 0, 5.0);
        self_e.set(0, 1, 1.0);
        self_e.set(0, 2, 3.0);
        self_e.set(1, 0, 4.0);
        self_e.set(1, 1, 2.0);
        self_e.set(1, 2, 6.0);

        let pair_e = PairEnergyTable::new(&[(3, 3)]);
        let graph = ContactGraph::build(2, [(0u32, 1u32)]);

        let result = dp(&mut self_e, &pair_e, &graph);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn dp_two_slot_chain_picks_gmec() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[0.0, 10.0, 10.0, 0.0]);
        self_e.set(0, 1, 1.0);
        let result = dp(&mut self_e, &pair_e, &graph);
        assert_eq!(result, [0, 0]);
    }

    #[test]
    fn dp_pair_energy_overrides_self_preference() {
        let (mut self_e, pair_e, graph) = two_slot([2, 2], &[20.0, 0.0, 0.0, 0.0]);
        self_e.set(0, 1, 5.0);
        self_e.set(1, 1, 5.0);

        let result = dp(&mut self_e, &pair_e, &graph);
        let e = self_e.get(0, result[0])
            + self_e.get(1, result[1])
            + pair_val(pair_e.matrix(0), 2, true, result[0], result[1]);
        assert_abs_diff_eq!(e, 5.0, epsilon = 1e-5);
    }

    #[test]
    fn dp_three_slot_triangle_prefers_matching_rotamers() {
        let mut self_e = SelfEnergyTable::new(&[2, 2, 2]);
        let mut pair_e = PairEnergyTable::new(&[(2, 2), (2, 2), (2, 2)]);
        let graph = ContactGraph::build(3, [(0u32, 1u32), (0u32, 2u32), (1u32, 2u32)]);

        for edge in 0..3 {
            pair_e.set(edge, 0, 1, 10.0);
            pair_e.set(edge, 1, 0, 10.0);
        }

        let result = dp(&mut self_e, &pair_e, &graph);
        assert_eq!(result[0], result[1]);
        assert_eq!(result[1], result[2]);
    }

    #[test]
    fn dp_skips_pruned_candidates() {
        let (mut self_e, pair_e, graph) = two_slot([3, 2], &[0.0, 0.0, 0.0, 0.0, 5.0, 5.0]);
        self_e.set(0, 1, 2.0);
        self_e.set(0, 2, 3.0);
        self_e.prune(0, 0);

        let result = dp(&mut self_e, &pair_e, &graph);
        assert_ne!(result[0], 0, "pruned candidate must not be selected");
    }
}
