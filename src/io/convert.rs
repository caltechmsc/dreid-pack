use super::config::{
    BasisType, ChargeConfig, CleanConfig, DampingStrategy, EmbeddedQeqConfig, ForceFieldConfig,
    Format, HeteroQeqMethod, HisStrategy, NucleicScheme, PackingScope, ProteinScheme,
    ProtonationConfig, QeqConfig, ReadConfig, ResidueSelector, SolverOptions, TopologyConfig,
    VdwPotential, WaterScheme,
};
use super::error::Error;
use super::order;
use super::session::{
    AtomRef, Bond, FixedAtom, MobileSidechain, ResidueCategory, ResiduePosition, Session,
    StandardResidue, SystemMetadata,
};
use crate::model::{
    residue::ResidueType,
    system::{
        BuckMatrix, BuckPair, FixedAtomPool, ForceFieldParams, HBondParams, LjMatrix, LjPair,
        Residue, SidechainAtoms, System, VdwMatrix,
    },
    types::{TypeIdx, Vec3},
};
use dreid_forge::{AtomResidueInfo, BioMetadata, ForgedSystem, VdwPairPotential};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};

/// Reads a biomolecular structure and builds a packing [`Session`].
///
/// Parses the input PDB/mmCIF data, cleans, protonates, builds topology,
/// parameterizes with the DREIDING force field, and partitions atoms into
/// fixed scaffold and mobile sidechains.
///
/// # Errors
///
/// Returns [`Error::Parse`] if the structure data is malformed,
/// [`Error::Forge`] if force-field parameterization fails,
/// [`Error::Scope`] if the packing scope is invalid, or
/// [`Error::Io`] on OS-level read failure.
pub fn read<R: BufRead>(reader: R, fmt: Format, cfg: &ReadConfig) -> Result<Session, Error> {
    let df_sys = dreid_forge::io::BioReader::new(reader, bio_format(fmt))
        .clean(to_clean(&cfg.clean))
        .protonate(to_protonation(&cfg.protonation))
        .topology(to_topology(&cfg.topology))
        .read()
        .map_err(|e| Error::Parse(e.to_string()))?;

    let forged = dreid_forge::forge(&df_sys, &to_forge_config(&cfg.ff))
        .map_err(|e| Error::Forge(e.to_string()))?;

    build(forged, &cfg.scope)
}

/// Writes a packing [`Session`] back to a biomolecular structure file.
///
/// # Errors
///
/// Returns [`Error::Io`] on OS-level write failure.
pub fn write<W: Write>(writer: W, session: &Session, fmt: Format) -> Result<(), Error> {
    let df_sys = reconstruct(session);
    dreid_forge::io::BioWriter::new(writer, bio_format(fmt))
        .write(&df_sys)
        .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))
}

// ---------------------------------------------------------------------------
// Group A — config translation (pure, infallible)
// ---------------------------------------------------------------------------

fn bio_format(fmt: Format) -> dreid_forge::io::Format {
    match fmt {
        Format::Pdb => dreid_forge::io::Format::Pdb,
        Format::Mmcif => dreid_forge::io::Format::Mmcif,
    }
}

fn to_clean(c: &CleanConfig) -> dreid_forge::io::CleanConfig {
    dreid_forge::io::CleanConfig {
        remove_water: c.remove_water,
        remove_ions: c.remove_ions,
        remove_hydrogens: false,
        remove_hetero: c.remove_hetero,
        remove_residue_names: c.remove_residue_names.clone(),
        keep_residue_names: c.keep_residue_names.clone(),
    }
}

fn to_protonation(c: &ProtonationConfig) -> dreid_forge::io::ProtonationConfig {
    dreid_forge::io::ProtonationConfig {
        target_ph: c.target_ph,
        remove_existing_h: true,
        his_strategy: to_his_strategy(c.his_strategy),
        his_salt_bridge: c.his_salt_bridge,
    }
}

fn to_his_strategy(s: HisStrategy) -> dreid_forge::io::HisStrategy {
    match s {
        HisStrategy::Hid => dreid_forge::io::HisStrategy::DirectHID,
        HisStrategy::Hie => dreid_forge::io::HisStrategy::DirectHIE,
        HisStrategy::Random => dreid_forge::io::HisStrategy::Random,
        HisStrategy::HbNetwork => dreid_forge::io::HisStrategy::HbNetwork,
    }
}

fn to_topology(c: &TopologyConfig) -> dreid_forge::io::TopologyConfig {
    dreid_forge::io::TopologyConfig {
        hetero_templates: c.templates.iter().map(|t| t.0.clone()).collect(),
        disulfide_bond_cutoff: c.disulfide_cutoff,
    }
}

fn to_forge_config(c: &ForceFieldConfig) -> dreid_forge::ForgeConfig {
    dreid_forge::ForgeConfig {
        rules: c.rules.clone(),
        params: c.params.clone(),
        charge_method: to_charge_method(&c.charge),
        bond_potential: dreid_forge::BondPotentialType::Harmonic,
        angle_potential: dreid_forge::AnglePotentialType::Cosine,
        vdw_potential: match c.vdw {
            VdwPotential::Buckingham => dreid_forge::VdwPotentialType::Buckingham,
            VdwPotential::LennardJones => dreid_forge::VdwPotentialType::LennardJones,
        },
    }
}

fn to_charge_method(c: &ChargeConfig) -> dreid_forge::ChargeMethod {
    dreid_forge::ChargeMethod::Hybrid(dreid_forge::HybridConfig {
        protein_scheme: to_protein_scheme(c.protein_scheme),
        nucleic_scheme: to_nucleic_scheme(c.nucleic_scheme),
        water_scheme: to_water_scheme(c.water_scheme),
        ligand_configs: c
            .hetero_configs
            .iter()
            .map(|hc| dreid_forge::LigandChargeConfig {
                selector: dreid_forge::ResidueSelector::new(
                    &hc.selector.chain_id,
                    hc.selector.residue_id,
                    hc.selector.insertion_code,
                ),
                method: to_ligand_method(&hc.method),
            })
            .collect(),
        default_ligand_method: to_ligand_method(&c.default_hetero_method),
    })
}

fn to_ligand_method(m: &HeteroQeqMethod) -> dreid_forge::LigandQeqMethod {
    match m {
        HeteroQeqMethod::Vacuum(q) => dreid_forge::LigandQeqMethod::Vacuum(to_qeq_config(q)),
        HeteroQeqMethod::Embedded(e) => dreid_forge::LigandQeqMethod::Embedded(to_embedded_qeq(e)),
    }
}

fn to_qeq_config(c: &QeqConfig) -> dreid_forge::QeqConfig {
    dreid_forge::QeqConfig {
        total_charge: c.total_charge,
        solver_options: to_solver_options(&c.solver_options),
    }
}

fn to_embedded_qeq(c: &EmbeddedQeqConfig) -> dreid_forge::EmbeddedQeqConfig {
    dreid_forge::EmbeddedQeqConfig {
        cutoff_radius: c.cutoff_radius,
        qeq: to_qeq_config(&c.qeq),
    }
}

fn to_solver_options(s: &SolverOptions) -> dreid_forge::SolverOptions {
    dreid_forge::SolverOptions {
        tolerance: s.tolerance,
        max_iterations: s.max_iterations,
        lambda_scale: s.lambda_scale,
        hydrogen_scf: s.hydrogen_scf,
        basis_type: to_basis_type(s.basis_type),
        damping: to_damping(s.damping),
    }
}

fn to_protein_scheme(s: ProteinScheme) -> dreid_forge::ProteinScheme {
    match s {
        ProteinScheme::AmberFFSB => dreid_forge::ProteinScheme::AmberFFSB,
        ProteinScheme::AmberFF03 => dreid_forge::ProteinScheme::AmberFF03,
        ProteinScheme::Charmm => dreid_forge::ProteinScheme::Charmm,
    }
}

fn to_nucleic_scheme(s: NucleicScheme) -> dreid_forge::NucleicScheme {
    match s {
        NucleicScheme::Amber => dreid_forge::NucleicScheme::Amber,
        NucleicScheme::Charmm => dreid_forge::NucleicScheme::Charmm,
    }
}

fn to_water_scheme(s: WaterScheme) -> dreid_forge::WaterScheme {
    match s {
        WaterScheme::Tip3p => dreid_forge::WaterScheme::Tip3p,
        WaterScheme::Tip3pFb => dreid_forge::WaterScheme::Tip3pFb,
        WaterScheme::Spc => dreid_forge::WaterScheme::Spc,
        WaterScheme::SpcE => dreid_forge::WaterScheme::SpcE,
        WaterScheme::Opc3 => dreid_forge::WaterScheme::Opc3,
    }
}

fn to_basis_type(b: BasisType) -> dreid_forge::BasisType {
    match b {
        BasisType::Gto => dreid_forge::BasisType::Gto,
        BasisType::Sto => dreid_forge::BasisType::Sto,
    }
}

fn to_damping(d: DampingStrategy) -> dreid_forge::DampingStrategy {
    match d {
        DampingStrategy::None => dreid_forge::DampingStrategy::None,
        DampingStrategy::Fixed(f) => dreid_forge::DampingStrategy::Fixed(f),
        DampingStrategy::Auto { initial } => dreid_forge::DampingStrategy::Auto { initial },
    }
}

struct ResidueGroup {
    chain_id: String,
    residue_id: i32,
    insertion_code: Option<char>,
    residue_name: String,
    standard_name: Option<StandardResidue>,
    category: ResidueCategory,
    position: ResiduePosition,
    atom_indices: Vec<usize>,
}

struct MobileMetadata {
    group_idx: usize,
    res_type: ResidueType,
    n_idx: usize,
    ca_idx: usize,
    c_idx: usize,
    sc_df_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Group B — ForgedSystem -> Session (read path)
// ---------------------------------------------------------------------------

fn build(forged: ForgedSystem, scope: &PackingScope) -> Result<Session, Error> {
    let bio = forged
        .system
        .bio_metadata
        .as_ref()
        .ok_or_else(|| Error::Parse("input system lacks biological metadata".into()))?;

    let groups = group_residues(bio);
    let mobile_metas = classify_mobile(&groups, bio, &forged.system.atoms, scope)?;
    let n_atoms = forged.system.atoms.len();
    let atom_to_ref = build_atom_to_ref(n_atoms, &mobile_metas);
    let (ff, h_types) = build_ff(forged.atom_types.len(), &forged.potentials)?;
    let phi_psi_omega = compute_phi_psi_omega(&forged.system.atoms, bio, &mobile_metas, &groups);
    let (fixed_pool, fixed_meta) = build_fixed_pool(&forged, bio, &atom_to_ref, &h_types);
    let (mobile, mobile_meta) =
        build_mobile(&forged, &groups, &mobile_metas, &phi_psi_omega, &h_types);
    let bonds = build_bonds(&forged.system.bonds, &atom_to_ref);

    let system = System {
        mobile,
        fixed: fixed_pool,
        ff,
    };
    let metadata = SystemMetadata {
        box_vectors: forged.system.box_vectors,
        bonds,
        fixed_atoms: fixed_meta,
        mobile_residues: mobile_meta,
    };
    Ok(Session::new(system, metadata))
}

fn group_residues(bio: &BioMetadata) -> Vec<ResidueGroup> {
    let mut groups: Vec<ResidueGroup> = Vec::new();

    for (i, info) in bio.atom_info.iter().enumerate() {
        let same = groups.last().is_some_and(|g| {
            g.chain_id == info.chain_id
                && g.residue_id == info.residue_id
                && g.insertion_code == info.insertion_code
        });

        if same {
            groups.last_mut().unwrap().atom_indices.push(i);
        } else {
            groups.push(ResidueGroup {
                chain_id: info.chain_id.clone(),
                residue_id: info.residue_id,
                insertion_code: info.insertion_code,
                residue_name: info.residue_name.clone(),
                standard_name: info.standard_name,
                category: info.category,
                position: info.position,
                atom_indices: vec![i],
            });
        }
    }

    groups
}

fn classify_mobile(
    groups: &[ResidueGroup],
    bio: &BioMetadata,
    atoms: &[dreid_forge::Atom],
    scope: &PackingScope,
) -> Result<Vec<MobileMetadata>, Error> {
    let candidates: Vec<(usize, ResidueType)> = groups
        .iter()
        .enumerate()
        .filter_map(|(gi, g)| {
            if g.category != ResidueCategory::Standard {
                return None;
            }
            residue_name_to_type(&g.residue_name)
                .filter(|rt| rt.is_packable())
                .map(|rt| (gi, rt))
        })
        .collect();

    let selected = apply_scope(candidates, groups, atoms, scope)?;

    let mut metas = Vec::with_capacity(selected.len());
    for (gi, rt) in selected {
        let g = &groups[gi];

        let n_idx = find_backbone_atom(g, bio, "N").ok_or_else(|| {
            Error::Parse(format!(
                "missing backbone N in {} {} {}",
                g.chain_id, g.residue_name, g.residue_id,
            ))
        })?;
        let ca_idx = find_backbone_atom(g, bio, "CA").ok_or_else(|| {
            Error::Parse(format!(
                "missing backbone CA in {} {} {}",
                g.chain_id, g.residue_name, g.residue_id,
            ))
        })?;
        let c_idx = find_backbone_atom(g, bio, "C").ok_or_else(|| {
            Error::Parse(format!(
                "missing backbone C in {} {} {}",
                g.chain_id, g.residue_name, g.residue_id,
            ))
        })?;

        let name_to_idx: HashMap<&str, usize> = g
            .atom_indices
            .iter()
            .map(|&i| (bio.atom_info[i].atom_name.as_str(), i))
            .collect();

        let sc_names = order::sidechain(rt);
        let mut sc_df_indices = Vec::with_capacity(sc_names.len());
        for &name in sc_names {
            let &df_i = name_to_idx.get(name).ok_or_else(|| {
                Error::Parse(format!(
                    "missing sidechain atom {name:?} in {} {} {}",
                    g.chain_id, g.residue_name, g.residue_id,
                ))
            })?;
            sc_df_indices.push(df_i);
        }

        metas.push(MobileMetadata {
            group_idx: gi,
            res_type: rt,
            n_idx,
            ca_idx,
            c_idx,
            sc_df_indices,
        });
    }

    if metas.is_empty() {
        return Err(Error::Scope(
            "no mobile residues selected by packing scope".into(),
        ));
    }

    Ok(metas)
}

fn residue_name_to_type(name: &str) -> Option<ResidueType> {
    match name {
        "GLY" => Some(ResidueType::Gly),
        "ALA" => Some(ResidueType::Ala),
        "VAL" => Some(ResidueType::Val),
        "CYM" => Some(ResidueType::Cym),
        "CYX" => Some(ResidueType::Cyx),
        "CYS" => Some(ResidueType::Cys),
        "SER" => Some(ResidueType::Ser),
        "THR" => Some(ResidueType::Thr),
        "PRO" => Some(ResidueType::Pro),
        "ASP" => Some(ResidueType::Asp),
        "ASN" => Some(ResidueType::Asn),
        "ILE" => Some(ResidueType::Ile),
        "LEU" => Some(ResidueType::Leu),
        "PHE" => Some(ResidueType::Phe),
        "TYM" => Some(ResidueType::Tym),
        "HID" => Some(ResidueType::Hid),
        "HIE" => Some(ResidueType::Hie),
        "HIP" => Some(ResidueType::Hip),
        "TRP" => Some(ResidueType::Trp),
        "ASH" => Some(ResidueType::Ash),
        "TYR" => Some(ResidueType::Tyr),
        "MET" => Some(ResidueType::Met),
        "GLU" => Some(ResidueType::Glu),
        "GLN" => Some(ResidueType::Gln),
        "GLH" => Some(ResidueType::Glh),
        "ARG" => Some(ResidueType::Arg),
        "ARN" => Some(ResidueType::Arn),
        "LYN" => Some(ResidueType::Lyn),
        "LYS" => Some(ResidueType::Lys),
        _ => None,
    }
}

fn find_backbone_atom(group: &ResidueGroup, bio: &BioMetadata, name: &str) -> Option<usize> {
    group
        .atom_indices
        .iter()
        .copied()
        .find(|&i| bio.atom_info[i].atom_name == name)
}

fn apply_scope(
    candidates: Vec<(usize, ResidueType)>,
    groups: &[ResidueGroup],
    atoms: &[dreid_forge::Atom],
    scope: &PackingScope,
) -> Result<Vec<(usize, ResidueType)>, Error> {
    match scope {
        PackingScope::Full => Ok(candidates),
        PackingScope::Pocket { anchor, radius } => {
            filter_pocket(candidates, groups, atoms, anchor, *radius)
        }
        PackingScope::Interface {
            groups: chain_groups,
            cutoff,
        } => filter_interface(candidates, groups, atoms, chain_groups, *cutoff),
        PackingScope::List(selectors) => Ok(filter_list(candidates, groups, selectors)),
    }
}

fn filter_list(
    candidates: Vec<(usize, ResidueType)>,
    groups: &[ResidueGroup],
    selectors: &[ResidueSelector],
) -> Vec<(usize, ResidueType)> {
    let set: HashSet<(&str, i32, Option<char>)> = selectors
        .iter()
        .map(|s| (s.chain_id.as_str(), s.residue_id, s.insertion_code))
        .collect();
    candidates
        .into_iter()
        .filter(|&(gi, _)| {
            let g = &groups[gi];
            set.contains(&(g.chain_id.as_str(), g.residue_id, g.insertion_code))
        })
        .collect()
}

fn filter_pocket(
    candidates: Vec<(usize, ResidueType)>,
    groups: &[ResidueGroup],
    atoms: &[dreid_forge::Atom],
    anchor: &ResidueSelector,
    radius: f32,
) -> Result<Vec<(usize, ResidueType)>, Error> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(Error::Scope(
            "pocket radius must be finite and positive".into(),
        ));
    }

    let anchor_group = groups
        .iter()
        .find(|g| {
            g.chain_id == anchor.chain_id
                && g.residue_id == anchor.residue_id
                && g.insertion_code == anchor.insertion_code
        })
        .ok_or_else(|| {
            Error::Scope(format!(
                "anchor residue {} not found in structure",
                fmt_selector(anchor),
            ))
        })?;

    let cell_size = radius as f64;
    let cutoff_sq = (radius as f64) * (radius as f64);
    let cells = build_cell_list(anchor_group.atom_indices.iter().copied(), atoms, cell_size);

    if cells.is_empty() {
        return Err(Error::Scope("anchor residue has no heavy atoms".into()));
    }

    Ok(candidates
        .into_iter()
        .filter(|&(gi, _)| any_heavy_atom_near(&groups[gi], atoms, &cells, cell_size, cutoff_sq))
        .collect())
}

fn filter_interface(
    candidates: Vec<(usize, ResidueType)>,
    groups: &[ResidueGroup],
    atoms: &[dreid_forge::Atom],
    chain_groups: &[Vec<String>; 2],
    cutoff: f32,
) -> Result<Vec<(usize, ResidueType)>, Error> {
    if chain_groups[0].is_empty() || chain_groups[1].is_empty() {
        return Err(Error::Scope("interface group must not be empty".into()));
    }
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(Error::Scope(
            "interface cutoff must be finite and positive".into(),
        ));
    }

    let set_a: HashSet<&str> = chain_groups[0].iter().map(|s| s.as_str()).collect();
    let set_b: HashSet<&str> = chain_groups[1].iter().map(|s| s.as_str()).collect();

    for chain in &set_a {
        if set_b.contains(chain) {
            return Err(Error::Scope(format!(
                "chain {chain:?} appears in both interface groups",
            )));
        }
    }

    let all_chains: HashSet<&str> = groups.iter().map(|g| g.chain_id.as_str()).collect();
    for chain in set_a.iter().chain(set_b.iter()) {
        if !all_chains.contains(chain) {
            return Err(Error::Scope(format!(
                "chain {chain:?} not found in structure",
            )));
        }
    }

    let cell_size = cutoff as f64;
    let cutoff_sq = (cutoff as f64) * (cutoff as f64);

    let cells_a = build_cell_list(
        groups
            .iter()
            .filter(|g| set_a.contains(g.chain_id.as_str()))
            .flat_map(|g| g.atom_indices.iter().copied()),
        atoms,
        cell_size,
    );
    let cells_b = build_cell_list(
        groups
            .iter()
            .filter(|g| set_b.contains(g.chain_id.as_str()))
            .flat_map(|g| g.atom_indices.iter().copied()),
        atoms,
        cell_size,
    );

    Ok(candidates
        .into_iter()
        .filter(|&(gi, _)| {
            let chain = groups[gi].chain_id.as_str();
            if set_a.contains(chain) {
                any_heavy_atom_near(&groups[gi], atoms, &cells_b, cell_size, cutoff_sq)
            } else if set_b.contains(chain) {
                any_heavy_atom_near(&groups[gi], atoms, &cells_a, cell_size, cutoff_sq)
            } else {
                false
            }
        })
        .collect())
}

fn build_cell_list(
    atom_indices: impl Iterator<Item = usize>,
    atoms: &[dreid_forge::Atom],
    cell_size: f64,
) -> HashMap<(i32, i32, i32), Vec<[f64; 3]>> {
    let mut cells: HashMap<(i32, i32, i32), Vec<[f64; 3]>> = HashMap::new();
    for i in atom_indices {
        if atoms[i].element == dreid_forge::Element::H {
            continue;
        }
        let pos = atoms[i].position;
        cells.entry(cell_key(pos, cell_size)).or_default().push(pos);
    }
    cells
}

fn any_heavy_atom_near(
    group: &ResidueGroup,
    atoms: &[dreid_forge::Atom],
    cells: &HashMap<(i32, i32, i32), Vec<[f64; 3]>>,
    cell_size: f64,
    cutoff_sq: f64,
) -> bool {
    for &ai in &group.atom_indices {
        if atoms[ai].element == dreid_forge::Element::H {
            continue;
        }
        let pos = atoms[ai].position;
        let (cx, cy, cz) = cell_key(pos, cell_size);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(bucket) = cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &other in bucket {
                            if dist_sq(pos, other) <= cutoff_sq {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

fn build_atom_to_ref(n_atoms: usize, mobile_metas: &[MobileMetadata]) -> Vec<AtomRef> {
    let mut refs = vec![AtomRef::Fixed(0); n_atoms];
    let mut is_mobile = vec![false; n_atoms];

    for (r, m) in mobile_metas.iter().enumerate() {
        for (l, &df_i) in m.sc_df_indices.iter().enumerate() {
            refs[df_i] = AtomRef::Mobile {
                residue: r as u32,
                local: l as u8,
            };
            is_mobile[df_i] = true;
        }
    }

    let mut fixed_k = 0u32;
    for i in 0..n_atoms {
        if !is_mobile[i] {
            refs[i] = AtomRef::Fixed(fixed_k);
            fixed_k += 1;
        }
    }

    refs
}

fn build_ff(
    n_types: usize,
    potentials: &dreid_forge::Potentials,
) -> Result<(ForceFieldParams, HashSet<TypeIdx>), Error> {
    if n_types > 256 {
        return Err(Error::Parse(format!(
            "atom type count ({n_types}) exceeds u8 capacity"
        )));
    }

    let vdw = build_vdw_matrix(n_types, &potentials.vdw_pairs)?;

    let mut h_types = HashSet::new();
    let mut acc_types = HashSet::new();
    let mut params = HashMap::new();
    for hb in &potentials.h_bonds {
        let d = TypeIdx(hb.donor_type_idx as u8);
        let h = TypeIdx(hb.hydrogen_type_idx as u8);
        let a = TypeIdx(hb.acceptor_type_idx as u8);
        h_types.insert(h);
        acc_types.insert(a);
        params.insert((d, h, a), (hb.d_hb as f32, hb.r_hb_sq as f32));
    }

    let hbond = HBondParams::new(h_types.clone(), acc_types, params);
    let ff = ForceFieldParams { vdw, hbond };
    Ok((ff, h_types))
}

fn build_vdw_matrix(n: usize, vdw_pairs: &[VdwPairPotential]) -> Result<VdwMatrix, Error> {
    if vdw_pairs.is_empty() {
        let zero = LjPair {
            d0: 0.0,
            r0_sq: 0.0,
        };
        return Ok(VdwMatrix::LennardJones(LjMatrix::new(n, vec![zero; n * n])));
    }

    let is_lj = matches!(vdw_pairs[0], VdwPairPotential::LennardJones { .. });

    if is_lj {
        let mut data = vec![
            LjPair {
                d0: 0.0,
                r0_sq: 0.0
            };
            n * n
        ];
        for p in vdw_pairs {
            match *p {
                VdwPairPotential::LennardJones {
                    type1_idx,
                    type2_idx,
                    d0,
                    r0_sq,
                } => {
                    let pair = LjPair {
                        d0: d0 as f32,
                        r0_sq: r0_sq as f32,
                    };
                    data[type1_idx * n + type2_idx] = pair;
                    data[type2_idx * n + type1_idx] = pair;
                }
                _ => return Err(Error::Parse("mixed VdW potential variants".into())),
            }
        }
        Ok(VdwMatrix::LennardJones(LjMatrix::new(n, data)))
    } else {
        let zero = BuckPair {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            r_max_sq: 0.0,
            two_e_max: 0.0,
        };
        let mut data = vec![zero; n * n];
        for p in vdw_pairs {
            match *p {
                VdwPairPotential::Buckingham {
                    type1_idx,
                    type2_idx,
                    a,
                    b,
                    c,
                    r_max_sq,
                    two_e_max,
                } => {
                    let pair = BuckPair {
                        a: a as f32,
                        b: b as f32,
                        c: c as f32,
                        r_max_sq: r_max_sq as f32,
                        two_e_max: two_e_max as f32,
                    };
                    data[type1_idx * n + type2_idx] = pair;
                    data[type2_idx * n + type1_idx] = pair;
                }
                _ => return Err(Error::Parse("mixed VdW potential variants".into())),
            }
        }
        Ok(VdwMatrix::Buckingham(BuckMatrix::new(n, data)))
    }
}

fn compute_phi_psi_omega(
    atoms: &[dreid_forge::Atom],
    bio: &BioMetadata,
    mobile_metas: &[MobileMetadata],
    groups: &[ResidueGroup],
) -> Vec<(f32, f32, f32)> {
    let mut chain_groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (gi, g) in groups.iter().enumerate() {
        chain_groups
            .entry(g.chain_id.as_str())
            .or_default()
            .push(gi);
    }

    let mut group_chain_pos: HashMap<usize, usize> = HashMap::new();
    for indices in chain_groups.values() {
        for (pos, &gi) in indices.iter().enumerate() {
            group_chain_pos.insert(gi, pos);
        }
    }

    mobile_metas
        .iter()
        .map(|m| {
            let g = &groups[m.group_idx];
            let chain_idx = &chain_groups[g.chain_id.as_str()];
            let pos = group_chain_pos[&m.group_idx];

            let phi = if pos > 0 {
                let prev_g = &groups[chain_idx[pos - 1]];
                find_backbone_atom(prev_g, bio, "C")
                    .map(|ci| {
                        dihedral(
                            atoms[ci].position,
                            atoms[m.n_idx].position,
                            atoms[m.ca_idx].position,
                            atoms[m.c_idx].position,
                        )
                    })
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let psi = if pos + 1 < chain_idx.len() {
                let next_g = &groups[chain_idx[pos + 1]];
                find_backbone_atom(next_g, bio, "N")
                    .map(|ni| {
                        dihedral(
                            atoms[m.n_idx].position,
                            atoms[m.ca_idx].position,
                            atoms[m.c_idx].position,
                            atoms[ni].position,
                        )
                    })
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let omega = if pos > 0 {
                let prev_g = &groups[chain_idx[pos - 1]];
                find_backbone_atom(prev_g, bio, "CA")
                    .and_then(|prev_ca| {
                        find_backbone_atom(prev_g, bio, "C").map(|prev_c| {
                            dihedral(
                                atoms[prev_ca].position,
                                atoms[prev_c].position,
                                atoms[m.n_idx].position,
                                atoms[m.ca_idx].position,
                            )
                        })
                    })
                    .unwrap_or(std::f32::consts::PI)
            } else {
                std::f32::consts::PI
            };

            (phi, psi, omega)
        })
        .collect()
}

fn build_fixed_pool(
    forged: &ForgedSystem,
    bio: &BioMetadata,
    atom_to_ref: &[AtomRef],
    h_types: &HashSet<TypeIdx>,
) -> (FixedAtomPool, Vec<FixedAtom>) {
    let n_fixed = atom_to_ref
        .iter()
        .filter(|r| matches!(r, AtomRef::Fixed(_)))
        .count();

    let mut positions = Vec::with_capacity(n_fixed);
    let mut types = Vec::with_capacity(n_fixed);
    let mut charges = Vec::with_capacity(n_fixed);
    let mut donor_for_h = vec![u32::MAX; n_fixed];
    let mut fixed_meta = Vec::with_capacity(n_fixed);

    for (i, atom_ref) in atom_to_ref.iter().enumerate() {
        if !matches!(atom_ref, AtomRef::Fixed(_)) {
            continue;
        }
        let pos = forged.system.atoms[i].position;
        positions.push(Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32));
        types.push(TypeIdx(forged.atom_properties[i].type_idx as u8));
        charges.push(forged.atom_properties[i].charge as f32);

        let info = &bio.atom_info[i];
        fixed_meta.push(FixedAtom {
            atom_name: info.atom_name.clone(),
            residue_name: info.residue_name.clone(),
            residue_id: info.residue_id,
            chain_id: info.chain_id.clone(),
            insertion_code: info.insertion_code,
            standard_name: info.standard_name,
            category: info.category,
            position: info.position,
            element: forged.system.atoms[i].element,
        });
    }

    for b in &forged.system.bonds {
        if let (AtomRef::Fixed(fa), AtomRef::Fixed(fb)) = (atom_to_ref[b.i], atom_to_ref[b.j]) {
            let (fa, fb) = (fa as usize, fb as usize);
            if h_types.contains(&types[fa]) {
                donor_for_h[fa] = fb as u32;
            }
            if h_types.contains(&types[fb]) {
                donor_for_h[fb] = fa as u32;
            }
        }
    }

    let pool = FixedAtomPool {
        positions,
        types,
        charges,
        donor_for_h,
    };
    (pool, fixed_meta)
}

fn build_mobile(
    forged: &ForgedSystem,
    groups: &[ResidueGroup],
    mobile_metas: &[MobileMetadata],
    phi_psi_omega: &[(f32, f32, f32)],
    h_types: &HashSet<TypeIdx>,
) -> (Vec<Residue>, Vec<MobileSidechain>) {
    let mut df_to_local: HashMap<usize, (usize, u8)> = HashMap::new();
    for (r, m) in mobile_metas.iter().enumerate() {
        for (l, &df_i) in m.sc_df_indices.iter().enumerate() {
            df_to_local.insert(df_i, (r, l as u8));
        }
    }

    let mut donors: Vec<Vec<u8>> = mobile_metas
        .iter()
        .map(|m| vec![u8::MAX; m.sc_df_indices.len()])
        .collect();

    for b in &forged.system.bonds {
        let (opt_a, opt_b) = (df_to_local.get(&b.i), df_to_local.get(&b.j));
        if let (Some(&(ra, la)), Some(&(rb, lb))) = (opt_a, opt_b) {
            if ra != rb {
                continue;
            }
            let type_a = TypeIdx(forged.atom_properties[b.i].type_idx as u8);
            let type_b = TypeIdx(forged.atom_properties[b.j].type_idx as u8);
            if h_types.contains(&type_a) {
                donors[ra][la as usize] = lb;
            }
            if h_types.contains(&type_b) {
                donors[rb][lb as usize] = la;
            }
        }
    }

    let mut residues = Vec::with_capacity(mobile_metas.len());
    let mut meta = Vec::with_capacity(mobile_metas.len());

    for (r, m) in mobile_metas.iter().enumerate() {
        let g = &groups[m.group_idx];
        let (phi, psi, omega) = phi_psi_omega[r];
        let anchor = [
            to_vec3(forged.system.atoms[m.n_idx].position),
            to_vec3(forged.system.atoms[m.ca_idx].position),
            to_vec3(forged.system.atoms[m.c_idx].position),
        ];

        let n_sc = m.sc_df_indices.len();
        let mut coords = Vec::with_capacity(n_sc);
        let mut types = Vec::with_capacity(n_sc);
        let mut charges = Vec::with_capacity(n_sc);

        for &df_i in &m.sc_df_indices {
            coords.push(to_vec3(forged.system.atoms[df_i].position));
            types.push(TypeIdx(forged.atom_properties[df_i].type_idx as u8));
            charges.push(forged.atom_properties[df_i].charge as f32);
        }

        let residue = Residue::new(
            m.res_type,
            anchor,
            phi,
            psi,
            omega,
            SidechainAtoms {
                coords: &coords,
                types: &types,
                charges: &charges,
                donor_of_h: &donors[r],
            },
        )
        .expect("packable ResidueType must produce Some");
        residues.push(residue);

        meta.push(MobileSidechain {
            residue_name: g.residue_name.clone(),
            residue_id: g.residue_id,
            chain_id: g.chain_id.clone(),
            insertion_code: g.insertion_code,
            standard_name: g.standard_name,
            category: g.category,
            position: g.position,
            elements: m
                .sc_df_indices
                .iter()
                .map(|&df_i| forged.system.atoms[df_i].element)
                .collect(),
        });
    }

    (residues, meta)
}

fn build_bonds(df_bonds: &[dreid_forge::Bond], atom_to_ref: &[AtomRef]) -> Vec<Bond> {
    df_bonds
        .iter()
        .map(|b| Bond {
            a: atom_to_ref[b.i],
            b: atom_to_ref[b.j],
            order: b.order,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Group C — Session -> dreid_forge::System (write path)
// ---------------------------------------------------------------------------

fn reconstruct(session: &Session) -> dreid_forge::System {
    let meta = session.metadata();
    let sys = &session.system;
    let n_fixed = sys.fixed.positions.len();
    let n_mobile_atoms: usize = sys.mobile.iter().map(|r| r.sidechain().len()).sum();
    let n_total = n_fixed + n_mobile_atoms;

    let mut atoms = Vec::with_capacity(n_total);
    let mut atom_info = Vec::with_capacity(n_total);

    for (k, fa) in meta.fixed_atoms.iter().enumerate() {
        let p = sys.fixed.positions[k];
        atoms.push(dreid_forge::Atom::new(
            fa.element,
            [p.x as f64, p.y as f64, p.z as f64],
        ));
        atom_info.push(
            AtomResidueInfo::builder(&fa.atom_name, &fa.residue_name, fa.residue_id, &fa.chain_id)
                .insertion_code_opt(fa.insertion_code)
                .standard_name(fa.standard_name)
                .category(fa.category)
                .position(fa.position)
                .build(),
        );
    }

    for (residue, ms) in sys.mobile.iter().zip(meta.mobile_residues.iter()) {
        let rt = residue_name_to_type(&ms.residue_name)
            .expect("MobileSidechain always maps to a valid ResidueType");
        let sc_names = order::sidechain(rt);
        for (l, &name) in sc_names.iter().enumerate() {
            let pos = residue.sidechain()[l];
            atoms.push(dreid_forge::Atom::new(
                ms.elements[l],
                [pos.x as f64, pos.y as f64, pos.z as f64],
            ));
            atom_info.push(
                AtomResidueInfo::builder(name, &ms.residue_name, ms.residue_id, &ms.chain_id)
                    .insertion_code_opt(ms.insertion_code)
                    .standard_name(ms.standard_name)
                    .category(ms.category)
                    .position(ms.position)
                    .build(),
            );
        }
    }

    let mobile_offsets: Vec<usize> = sys
        .mobile
        .iter()
        .scan(0usize, |acc, r| {
            let offset = *acc;
            *acc += r.sidechain().len();
            Some(offset)
        })
        .collect();

    let atom_ref_to_flat = |ar: AtomRef| -> usize {
        match ar {
            AtomRef::Fixed(k) => k as usize,
            AtomRef::Mobile { residue, local } => {
                n_fixed + mobile_offsets[residue as usize] + local as usize
            }
        }
    };

    let bonds: Vec<dreid_forge::Bond> = meta
        .bonds
        .iter()
        .map(|b| dreid_forge::Bond {
            i: atom_ref_to_flat(b.a),
            j: atom_ref_to_flat(b.b),
            order: b.order,
        })
        .collect();

    let mut bio = BioMetadata::with_capacity(n_total);
    bio.atom_info = atom_info;

    dreid_forge::System {
        atoms,
        bonds,
        box_vectors: meta.box_vectors,
        bio_metadata: Some(bio),
    }
}

// ---------------------------------------------------------------------------
// Group D — pure helpers
// ---------------------------------------------------------------------------

fn dihedral(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f32 {
    let (a, b, c, d) = (to_vec3(a), to_vec3(b), to_vec3(c), to_vec3(d));
    let b1 = b - a;
    let b2 = c - b;
    let b3 = d - c;
    let n1 = b1.cross(b2);
    let n2 = b2.cross(b3);
    let x = n1.dot(n2) * b2.len();
    let y = n1.cross(b2).dot(n2);
    y.atan2(x)
}

fn to_vec3(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn cell_key(pos: [f64; 3], cell_size: f64) -> (i32, i32, i32) {
    (
        (pos[0] / cell_size).floor() as i32,
        (pos[1] / cell_size).floor() as i32,
        (pos[2] / cell_size).floor() as i32,
    )
}

fn fmt_selector(s: &ResidueSelector) -> String {
    match s.insertion_code {
        Some(ic) => format!("{} {}{}", s.chain_id, s.residue_id, ic),
        None => format!("{} {}", s.chain_id, s.residue_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use dreid_forge::BondOrder;

    fn ai(name: &str, res: &str, resid: i32, chain: &str) -> AtomResidueInfo {
        AtomResidueInfo::builder(name, res, resid, chain).build()
    }

    fn ai_icode(
        name: &str,
        res: &str,
        resid: i32,
        chain: &str,
        icode: Option<char>,
    ) -> AtomResidueInfo {
        AtomResidueInfo::builder(name, res, resid, chain)
            .insertion_code_opt(icode)
            .build()
    }

    fn ai_full(
        name: &str,
        res: &str,
        resid: i32,
        chain: &str,
        icode: Option<char>,
        std_name: Option<StandardResidue>,
        cat: ResidueCategory,
        pos: ResiduePosition,
    ) -> AtomResidueInfo {
        AtomResidueInfo::builder(name, res, resid, chain)
            .insertion_code_opt(icode)
            .standard_name(std_name)
            .category(cat)
            .position(pos)
            .build()
    }

    fn bio_with(infos: Vec<AtomResidueInfo>) -> BioMetadata {
        let mut bio = BioMetadata::new();
        bio.atom_info = infos;
        bio
    }

    fn mobile_meta(sc: Vec<usize>) -> MobileMetadata {
        MobileMetadata {
            group_idx: 0,
            res_type: ResidueType::Ser,
            n_idx: 0,
            ca_idx: 0,
            c_idx: 0,
            sc_df_indices: sc,
        }
    }

    fn group_at(chain: &str, resid: i32, name: &str, indices: Vec<usize>) -> ResidueGroup {
        ResidueGroup {
            chain_id: chain.into(),
            residue_id: resid,
            insertion_code: None,
            residue_name: name.into(),
            standard_name: None,
            category: ResidueCategory::Standard,
            position: ResiduePosition::None,
            atom_indices: indices,
        }
    }

    fn group_hetero(chain: &str, resid: i32, name: &str, indices: Vec<usize>) -> ResidueGroup {
        ResidueGroup {
            chain_id: chain.into(),
            residue_id: resid,
            insertion_code: None,
            residue_name: name.into(),
            standard_name: None,
            category: ResidueCategory::Hetero,
            position: ResiduePosition::None,
            atom_indices: indices,
        }
    }

    fn atom(pos: [f64; 3]) -> dreid_forge::Atom {
        dreid_forge::Atom::new(dreid_forge::Element::C, pos)
    }

    fn hydrogen(pos: [f64; 3]) -> dreid_forge::Atom {
        dreid_forge::Atom::new(dreid_forge::Element::H, pos)
    }

    fn sel(chain: &str, resid: i32) -> ResidueSelector {
        ResidueSelector {
            chain_id: chain.into(),
            residue_id: resid,
            insertion_code: None,
        }
    }

    #[test]
    fn bio_format_maps_pdb() {
        assert_eq!(bio_format(Format::Pdb), dreid_forge::io::Format::Pdb);
    }

    #[test]
    fn bio_format_maps_mmcif() {
        assert_eq!(bio_format(Format::Mmcif), dreid_forge::io::Format::Mmcif);
    }

    #[test]
    fn group_residues_empty() {
        assert!(group_residues(&bio_with(vec![])).is_empty());
    }

    #[test]
    fn group_residues_single_all_fields() {
        let bio = bio_with(vec![
            ai_full(
                "N",
                "SER",
                5,
                "B",
                Some('X'),
                Some(StandardResidue::SER),
                ResidueCategory::Standard,
                ResiduePosition::NTerminal,
            ),
            ai_icode("CA", "SER", 5, "B", Some('X')),
            ai_icode("C", "SER", 5, "B", Some('X')),
        ]);
        let groups = group_residues(&bio);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.chain_id, "B");
        assert_eq!(g.residue_id, 5);
        assert_eq!(g.insertion_code, Some('X'));
        assert_eq!(g.residue_name, "SER");
        assert_eq!(g.standard_name, Some(StandardResidue::SER));
        assert_eq!(g.category, ResidueCategory::Standard);
        assert_eq!(g.position, ResiduePosition::NTerminal);
        assert_eq!(g.atom_indices, vec![0, 1, 2]);
    }

    #[test]
    fn group_residues_merge() {
        let bio = bio_with(vec![
            ai("N", "ALA", 1, "A"),
            ai("CA", "ALA", 1, "A"),
            ai("C", "ALA", 1, "A"),
            ai("O", "ALA", 1, "A"),
            ai("CB", "ALA", 1, "A"),
        ]);
        let groups = group_residues(&bio);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].atom_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn group_residues_split_by_chain() {
        let bio = bio_with(vec![ai("N", "ALA", 1, "A"), ai("CA", "ALA", 1, "B")]);
        let groups = group_residues(&bio);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].chain_id, "A");
        assert_eq!(groups[0].atom_indices, vec![0]);
        assert_eq!(groups[1].chain_id, "B");
        assert_eq!(groups[1].atom_indices, vec![1]);
    }

    #[test]
    fn group_residues_split_by_resid() {
        let bio = bio_with(vec![ai("N", "SER", 1, "A"), ai("N", "SER", 2, "A")]);
        let groups = group_residues(&bio);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].residue_id, 1);
        assert_eq!(groups[0].atom_indices, vec![0]);
        assert_eq!(groups[1].residue_id, 2);
        assert_eq!(groups[1].atom_indices, vec![1]);
    }

    #[test]
    fn group_residues_split_by_icode() {
        let bio = bio_with(vec![
            ai_icode("N", "SER", 1, "A", None),
            ai_icode("N", "SER", 1, "A", Some('A')),
        ]);
        let groups = group_residues(&bio);
        assert_eq!(groups.len(), 2);
        assert!(groups[0].insertion_code.is_none());
        assert_eq!(groups[0].atom_indices, vec![0]);
        assert_eq!(groups[1].insertion_code, Some('A'));
        assert_eq!(groups[1].atom_indices, vec![1]);
    }

    #[test]
    fn residue_name_all_mappings_correct() {
        let cases = [
            ("GLY", ResidueType::Gly),
            ("ALA", ResidueType::Ala),
            ("VAL", ResidueType::Val),
            ("CYM", ResidueType::Cym),
            ("CYX", ResidueType::Cyx),
            ("CYS", ResidueType::Cys),
            ("SER", ResidueType::Ser),
            ("THR", ResidueType::Thr),
            ("PRO", ResidueType::Pro),
            ("ASP", ResidueType::Asp),
            ("ASN", ResidueType::Asn),
            ("ILE", ResidueType::Ile),
            ("LEU", ResidueType::Leu),
            ("PHE", ResidueType::Phe),
            ("TYM", ResidueType::Tym),
            ("HID", ResidueType::Hid),
            ("HIE", ResidueType::Hie),
            ("HIP", ResidueType::Hip),
            ("TRP", ResidueType::Trp),
            ("ASH", ResidueType::Ash),
            ("TYR", ResidueType::Tyr),
            ("MET", ResidueType::Met),
            ("GLU", ResidueType::Glu),
            ("GLN", ResidueType::Gln),
            ("GLH", ResidueType::Glh),
            ("ARG", ResidueType::Arg),
            ("ARN", ResidueType::Arn),
            ("LYN", ResidueType::Lyn),
            ("LYS", ResidueType::Lys),
        ];
        for (name, expected) in cases {
            assert_eq!(residue_name_to_type(name), Some(expected), "{name}");
        }
    }

    #[test]
    fn residue_name_unknown_returns_none() {
        assert!(residue_name_to_type("XYZ").is_none());
        assert!(residue_name_to_type("").is_none());
        assert!(residue_name_to_type("HOH").is_none());
        assert!(residue_name_to_type("ACE").is_none());
    }

    #[test]
    fn residue_name_case_sensitive() {
        assert!(residue_name_to_type("gly").is_none());
        assert!(residue_name_to_type("Gly").is_none());
        assert!(residue_name_to_type("GLy").is_none());
    }

    #[test]
    fn find_backbone_present() {
        let bio = bio_with(vec![
            ai("N", "SER", 1, "A"),
            ai("CA", "SER", 1, "A"),
            ai("C", "SER", 1, "A"),
        ]);
        let g = group_at("A", 1, "SER", vec![0, 1, 2]);
        assert_eq!(find_backbone_atom(&g, &bio, "CA"), Some(1));
    }

    #[test]
    fn find_backbone_absent() {
        let bio = bio_with(vec![ai("N", "SER", 1, "A"), ai("CA", "SER", 1, "A")]);
        let g = group_at("A", 1, "SER", vec![0, 1]);
        assert_eq!(find_backbone_atom(&g, &bio, "CB"), None);
    }

    #[test]
    fn find_backbone_first_occurrence() {
        let bio = bio_with(vec![
            ai("CA", "SER", 1, "A"),
            ai("N", "SER", 1, "A"),
            ai("CA", "SER", 1, "A"),
        ]);
        let g = group_at("A", 1, "SER", vec![0, 1, 2]);
        assert_eq!(find_backbone_atom(&g, &bio, "CA"), Some(0));
    }

    #[test]
    fn apply_scope_full_passthrough() {
        let candidates = vec![(0, ResidueType::Ser), (1, ResidueType::Val)];
        let groups = [
            group_at("A", 1, "SER", vec![0]),
            group_at("A", 2, "VAL", vec![1]),
        ];
        let atoms = [atom([0.0, 0.0, 0.0]), atom([1.0, 0.0, 0.0])];
        let result = apply_scope(candidates.clone(), &groups, &atoms, &PackingScope::Full).unwrap();
        assert_eq!(result, candidates);
    }

    #[test]
    fn filter_list_all_match() {
        let groups = [
            group_at("A", 1, "SER", vec![0]),
            group_at("A", 2, "VAL", vec![1]),
        ];
        let candidates = vec![(0, ResidueType::Ser), (1, ResidueType::Val)];
        let selectors = [sel("A", 1), sel("A", 2)];
        let result = filter_list(candidates, &groups, &selectors);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn filter_list_partial_match() {
        let groups = [
            group_at("A", 1, "SER", vec![0]),
            group_at("A", 2, "VAL", vec![1]),
        ];
        let candidates = vec![(0, ResidueType::Ser), (1, ResidueType::Val)];
        let selectors = [sel("A", 2)];
        let result = filter_list(candidates, &groups, &selectors);
        assert_eq!(result, vec![(1, ResidueType::Val)]);
    }

    #[test]
    fn filter_list_none_match() {
        let groups = [group_at("A", 1, "SER", vec![0])];
        let candidates = vec![(0, ResidueType::Ser)];
        let selectors = [sel("B", 99)];
        let result = filter_list(candidates, &groups, &selectors);
        assert!(result.is_empty());
    }

    #[test]
    fn filter_list_respects_insertion_code() {
        let mut g = group_at("A", 1, "SER", vec![0]);
        g.insertion_code = Some('A');
        let groups = [g];
        let candidates = vec![(0, ResidueType::Ser)];

        let miss = filter_list(candidates.clone(), &groups, &[sel("A", 1)]);
        assert!(miss.is_empty());

        let hit = ResidueSelector {
            chain_id: "A".into(),
            residue_id: 1,
            insertion_code: Some('A'),
        };
        let got = filter_list(candidates, &groups, &[hit]);
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn filter_pocket_zero_radius() {
        let err = filter_pocket(vec![], &[], &[], &sel("A", 1), 0.0).unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_pocket_negative_radius() {
        let err = filter_pocket(vec![], &[], &[], &sel("A", 1), -1.0).unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_pocket_infinite_radius() {
        let err = filter_pocket(vec![], &[], &[], &sel("A", 1), f32::INFINITY).unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_pocket_nan_radius() {
        let err = filter_pocket(vec![], &[], &[], &sel("A", 1), f32::NAN).unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_pocket_anchor_not_found() {
        let groups = [group_at("A", 1, "SER", vec![0])];
        let atoms = [atom([0.0, 0.0, 0.0])];
        let err = filter_pocket(vec![], &groups, &atoms, &sel("B", 99), 5.0).unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_pocket_anchor_only_hydrogen() {
        let groups = [
            group_hetero("A", 100, "LIG", vec![0]),
            group_at("A", 1, "SER", vec![1]),
        ];
        let atoms = [hydrogen([0.0, 0.0, 0.0]), atom([3.0, 0.0, 0.0])];
        let candidates = vec![(1, ResidueType::Ser)];
        let err = filter_pocket(candidates, &groups, &atoms, &sel("A", 100), 5.0).unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_pocket_keeps_nearby() {
        let groups = [
            group_hetero("A", 100, "LIG", vec![0]),
            group_at("A", 1, "SER", vec![1]),
        ];
        let atoms = [atom([0.0, 0.0, 0.0]), atom([3.0, 0.0, 0.0])];
        let candidates = vec![(1, ResidueType::Ser)];
        let result = filter_pocket(candidates, &groups, &atoms, &sel("A", 100), 5.0).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn filter_pocket_excludes_distant() {
        let groups = [
            group_hetero("A", 100, "LIG", vec![0]),
            group_at("A", 1, "SER", vec![1]),
        ];
        let atoms = [atom([0.0, 0.0, 0.0]), atom([10.0, 0.0, 0.0])];
        let candidates = vec![(1, ResidueType::Ser)];
        let result = filter_pocket(candidates, &groups, &atoms, &sel("A", 100), 5.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn filter_pocket_mixed_distances() {
        let groups = [
            group_hetero("A", 100, "LIG", vec![0]),
            group_at("A", 1, "SER", vec![1]),
            group_at("A", 2, "SER", vec![2]),
        ];
        let atoms = [
            atom([0.0, 0.0, 0.0]),
            atom([3.0, 0.0, 0.0]),
            atom([10.0, 0.0, 0.0]),
        ];
        let candidates = vec![(1, ResidueType::Ser), (2, ResidueType::Ser)];
        let result = filter_pocket(candidates, &groups, &atoms, &sel("A", 100), 5.0).unwrap();
        assert_eq!(result, vec![(1, ResidueType::Ser)]);
    }

    #[test]
    fn filter_interface_empty_group() {
        let err = filter_interface(vec![], &[], &[], &[vec![], vec!["B".into()]], 5.0).unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_interface_zero_cutoff() {
        let err = filter_interface(vec![], &[], &[], &[vec!["A".into()], vec!["B".into()]], 0.0)
            .unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_interface_groups_overlap() {
        let groups = [group_at("A", 1, "SER", vec![0])];
        let atoms = [atom([0.0, 0.0, 0.0])];
        let err = filter_interface(
            vec![],
            &groups,
            &atoms,
            &[vec!["A".into()], vec!["A".into()]],
            5.0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_interface_unknown_chain() {
        let groups = [group_at("A", 1, "SER", vec![0])];
        let atoms = [atom([0.0, 0.0, 0.0])];
        let err = filter_interface(
            vec![],
            &groups,
            &atoms,
            &[vec!["A".into()], vec!["Z".into()]],
            5.0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Scope(_)));
    }

    #[test]
    fn filter_interface_keeps_contacts() {
        let groups = [
            group_at("A", 1, "SER", vec![0]),
            group_at("B", 1, "SER", vec![1]),
        ];
        let atoms = [atom([0.0, 0.0, 0.0]), atom([3.0, 0.0, 0.0])];
        let candidates = vec![(0, ResidueType::Ser), (1, ResidueType::Ser)];
        let result = filter_interface(
            candidates,
            &groups,
            &atoms,
            &[vec!["A".into()], vec!["B".into()]],
            5.0,
        )
        .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn filter_interface_excludes_no_contacts() {
        let groups = [
            group_at("A", 1, "SER", vec![0]),
            group_at("B", 1, "SER", vec![1]),
        ];
        let atoms = [atom([0.0, 0.0, 0.0]), atom([100.0, 0.0, 0.0])];
        let candidates = vec![(0, ResidueType::Ser), (1, ResidueType::Ser)];
        let result = filter_interface(
            candidates,
            &groups,
            &atoms,
            &[vec!["A".into()], vec!["B".into()]],
            5.0,
        )
        .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn filter_interface_both_sides() {
        let groups = [
            group_at("A", 1, "SER", vec![0]),
            group_at("A", 2, "SER", vec![1]),
            group_at("B", 1, "SER", vec![2]),
        ];
        let atoms = [
            atom([0.0, 0.0, 0.0]),
            atom([100.0, 0.0, 0.0]),
            atom([3.0, 0.0, 0.0]),
        ];
        let candidates = vec![
            (0, ResidueType::Ser),
            (1, ResidueType::Ser),
            (2, ResidueType::Ser),
        ];
        let result = filter_interface(
            candidates,
            &groups,
            &atoms,
            &[vec!["A".into()], vec!["B".into()]],
            5.0,
        )
        .unwrap();
        assert_eq!(result, vec![(0, ResidueType::Ser), (2, ResidueType::Ser)]);
    }

    #[test]
    fn filter_interface_excludes_unaffiliated_chain() {
        let groups = [
            group_at("A", 1, "SER", vec![0]),
            group_at("B", 1, "SER", vec![1]),
            group_at("C", 1, "SER", vec![2]),
        ];
        let atoms = [
            atom([0.0, 0.0, 0.0]),
            atom([3.0, 0.0, 0.0]),
            atom([1.0, 0.0, 0.0]),
        ];
        let candidates = vec![
            (0, ResidueType::Ser),
            (1, ResidueType::Ser),
            (2, ResidueType::Ser),
        ];
        let result = filter_interface(
            candidates,
            &groups,
            &atoms,
            &[vec!["A".into()], vec!["B".into()]],
            5.0,
        )
        .unwrap();
        assert_eq!(result, vec![(0, ResidueType::Ser), (1, ResidueType::Ser)]);
    }

    #[test]
    fn build_cell_list_empty() {
        let cells = build_cell_list(std::iter::empty(), &[], 5.0);
        assert!(cells.is_empty());
    }

    #[test]
    fn build_cell_list_skips_hydrogen() {
        let atoms = [hydrogen([0.0, 0.0, 0.0]), hydrogen([1.0, 1.0, 1.0])];
        let cells = build_cell_list(0..2, &atoms, 5.0);
        assert!(cells.is_empty());
    }

    #[test]
    fn build_cell_list_bins_heavy_atoms() {
        let atoms = [
            atom([0.5, 0.5, 0.5]),
            atom([5.5, 0.5, 0.5]),
            hydrogen([0.5, 0.5, 0.5]),
        ];
        let cells = build_cell_list(0..3, &atoms, 5.0);
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[&(0, 0, 0)].len(), 1);
        assert_eq!(cells[&(1, 0, 0)].len(), 1);
    }

    #[test]
    fn any_heavy_atom_near_within() {
        let atoms = [atom([0.0, 0.0, 0.0]), atom([1.0, 0.0, 0.0])];
        let cells = build_cell_list(0..1, &atoms, 5.0);
        let g = group_at("A", 1, "SER", vec![1]);
        assert!(any_heavy_atom_near(&g, &atoms, &cells, 5.0, 4.0));
    }

    #[test]
    fn any_heavy_atom_near_outside() {
        let atoms = [atom([0.0, 0.0, 0.0]), atom([10.0, 0.0, 0.0])];
        let cells = build_cell_list(0..1, &atoms, 5.0);
        let g = group_at("A", 1, "SER", vec![1]);
        assert!(!any_heavy_atom_near(&g, &atoms, &cells, 5.0, 4.0));
    }

    #[test]
    fn any_heavy_atom_near_ignores_hydrogen() {
        let atoms = [atom([0.0, 0.0, 0.0]), hydrogen([0.5, 0.0, 0.0])];
        let cells = build_cell_list(0..1, &atoms, 5.0);
        let g = group_at("A", 1, "SER", vec![1]);
        assert!(!any_heavy_atom_near(&g, &atoms, &cells, 5.0, 4.0));
    }

    #[test]
    fn any_heavy_atom_near_cross_cell() {
        let atoms = [atom([2.0, 0.0, 0.0]), atom([6.0, 0.0, 0.0])];
        let cells = build_cell_list(0..1, &atoms, 5.0);
        let g = group_at("A", 1, "SER", vec![1]);
        assert!(any_heavy_atom_near(&g, &atoms, &cells, 5.0, 25.0));
    }

    #[test]
    fn any_heavy_atom_near_exact_boundary() {
        let atoms = [atom([0.0, 0.0, 0.0]), atom([3.0, 0.0, 0.0])];
        let cells = build_cell_list(0..1, &atoms, 5.0);
        let g = group_at("A", 1, "SER", vec![1]);
        assert!(any_heavy_atom_near(&g, &atoms, &cells, 5.0, 9.0));
    }

    #[test]
    fn any_heavy_atom_near_multi_atom_group() {
        let atoms = [
            atom([0.0, 0.0, 0.0]),
            atom([10.0, 0.0, 0.0]),
            atom([1.0, 0.0, 0.0]),
        ];
        let cells = build_cell_list(0..1, &atoms, 5.0);
        let g = group_at("A", 1, "SER", vec![1, 2]);
        assert!(any_heavy_atom_near(&g, &atoms, &cells, 5.0, 4.0));
    }

    #[test]
    fn atom_to_ref_no_mobile() {
        let refs = build_atom_to_ref(3, &[]);
        assert_eq!(
            refs,
            vec![AtomRef::Fixed(0), AtomRef::Fixed(1), AtomRef::Fixed(2)]
        );
    }

    #[test]
    fn atom_to_ref_mobile_refs() {
        let metas = [mobile_meta(vec![1, 3])];
        let refs = build_atom_to_ref(5, &metas);
        assert_eq!(
            refs[1],
            AtomRef::Mobile {
                residue: 0,
                local: 0
            }
        );
        assert_eq!(
            refs[3],
            AtomRef::Mobile {
                residue: 0,
                local: 1
            }
        );
    }

    #[test]
    fn atom_to_ref_contiguous_fixed() {
        let metas = [mobile_meta(vec![1, 3])];
        let refs = build_atom_to_ref(5, &metas);
        assert_eq!(refs[0], AtomRef::Fixed(0));
        assert_eq!(refs[2], AtomRef::Fixed(1));
        assert_eq!(refs[4], AtomRef::Fixed(2));
    }

    #[test]
    fn atom_to_ref_local_order() {
        let metas = [mobile_meta(vec![3, 1])];
        let refs = build_atom_to_ref(5, &metas);
        assert_eq!(
            refs[3],
            AtomRef::Mobile {
                residue: 0,
                local: 0
            }
        );
        assert_eq!(
            refs[1],
            AtomRef::Mobile {
                residue: 0,
                local: 1
            }
        );
    }

    #[test]
    fn atom_to_ref_two_residues() {
        let metas = [mobile_meta(vec![1, 2]), mobile_meta(vec![4, 5])];
        let refs = build_atom_to_ref(6, &metas);
        assert_eq!(refs[0], AtomRef::Fixed(0));
        assert_eq!(
            refs[1],
            AtomRef::Mobile {
                residue: 0,
                local: 0
            }
        );
        assert_eq!(
            refs[2],
            AtomRef::Mobile {
                residue: 0,
                local: 1
            }
        );
        assert_eq!(refs[3], AtomRef::Fixed(1));
        assert_eq!(
            refs[4],
            AtomRef::Mobile {
                residue: 1,
                local: 0
            }
        );
        assert_eq!(
            refs[5],
            AtomRef::Mobile {
                residue: 1,
                local: 1
            }
        );
    }

    #[test]
    fn vdw_empty_yields_zero_lj() {
        let m = build_vdw_matrix(2, &[]).unwrap();
        let VdwMatrix::LennardJones(lj) = m else {
            panic!("expected LJ");
        };
        let zero = LjPair {
            d0: 0.0,
            r0_sq: 0.0,
        };
        assert_eq!(lj.get(TypeIdx(0), TypeIdx(0)), zero);
        assert_eq!(lj.get(TypeIdx(0), TypeIdx(1)), zero);
        assert_eq!(lj.get(TypeIdx(1), TypeIdx(0)), zero);
        assert_eq!(lj.get(TypeIdx(1), TypeIdx(1)), zero);
    }

    #[test]
    fn vdw_lj_symmetric() {
        let pairs = vec![VdwPairPotential::LennardJones {
            type1_idx: 0,
            type2_idx: 1,
            d0: 3.0,
            r0_sq: 9.0,
        }];
        let m = build_vdw_matrix(2, &pairs).unwrap();
        let VdwMatrix::LennardJones(lj) = m else {
            panic!("expected LJ");
        };
        let p01 = lj.get(TypeIdx(0), TypeIdx(1));
        let p10 = lj.get(TypeIdx(1), TypeIdx(0));
        assert_eq!(p01, p10);
        assert_ne!(p01.d0, 0.0);
    }

    #[test]
    fn vdw_lj_values() {
        let pairs = vec![VdwPairPotential::LennardJones {
            type1_idx: 0,
            type2_idx: 1,
            d0: 2.5,
            r0_sq: 6.25,
        }];
        let m = build_vdw_matrix(2, &pairs).unwrap();
        let VdwMatrix::LennardJones(lj) = m else {
            panic!("expected LJ");
        };
        let p = lj.get(TypeIdx(0), TypeIdx(1));
        assert_eq!(p.d0, 2.5f32);
        assert_eq!(p.r0_sq, 6.25f32);
    }

    #[test]
    fn vdw_buck_symmetric() {
        let pairs = vec![VdwPairPotential::Buckingham {
            type1_idx: 0,
            type2_idx: 1,
            a: 1.0,
            b: 2.0,
            c: 3.0,
            r_max_sq: 4.0,
            two_e_max: 5.0,
        }];
        let m = build_vdw_matrix(2, &pairs).unwrap();
        let VdwMatrix::Buckingham(buck) = m else {
            panic!("expected Buck");
        };
        let p01 = buck.get(TypeIdx(0), TypeIdx(1));
        let p10 = buck.get(TypeIdx(1), TypeIdx(0));
        assert_eq!(p01, p10);
        assert_ne!(p01.a, 0.0);
    }

    #[test]
    fn vdw_buck_values() {
        let pairs = vec![VdwPairPotential::Buckingham {
            type1_idx: 0,
            type2_idx: 0,
            a: 10.0,
            b: 20.0,
            c: 30.0,
            r_max_sq: 40.0,
            two_e_max: 50.0,
        }];
        let m = build_vdw_matrix(1, &pairs).unwrap();
        let VdwMatrix::Buckingham(buck) = m else {
            panic!("expected Buck");
        };
        let p = buck.get(TypeIdx(0), TypeIdx(0));
        assert_eq!(p.a, 10.0f32);
        assert_eq!(p.b, 20.0f32);
        assert_eq!(p.c, 30.0f32);
        assert_eq!(p.r_max_sq, 40.0f32);
        assert_eq!(p.two_e_max, 50.0f32);
    }

    #[test]
    fn vdw_mixed_returns_error() {
        let pairs = vec![
            VdwPairPotential::LennardJones {
                type1_idx: 0,
                type2_idx: 0,
                d0: 1.0,
                r0_sq: 1.0,
            },
            VdwPairPotential::Buckingham {
                type1_idx: 0,
                type2_idx: 1,
                a: 1.0,
                b: 1.0,
                c: 1.0,
                r_max_sq: 1.0,
                two_e_max: 1.0,
            },
        ];
        assert!(build_vdw_matrix(2, &pairs).is_err());
    }

    #[test]
    fn build_bonds_empty() {
        assert!(build_bonds(&[], &[]).is_empty());
    }

    #[test]
    fn build_bonds_fixed_fixed() {
        let refs = vec![AtomRef::Fixed(0), AtomRef::Fixed(1)];
        let df = vec![dreid_forge::Bond {
            i: 0,
            j: 1,
            order: BondOrder::Single,
        }];
        let bonds = build_bonds(&df, &refs);
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].a, AtomRef::Fixed(0));
        assert_eq!(bonds[0].b, AtomRef::Fixed(1));
        assert_eq!(bonds[0].order, BondOrder::Single);
    }

    #[test]
    fn build_bonds_mobile_mobile() {
        let refs = vec![
            AtomRef::Mobile {
                residue: 0,
                local: 0,
            },
            AtomRef::Mobile {
                residue: 0,
                local: 1,
            },
        ];
        let df = vec![dreid_forge::Bond {
            i: 0,
            j: 1,
            order: BondOrder::Double,
        }];
        let bonds = build_bonds(&df, &refs);
        assert_eq!(bonds.len(), 1);
        assert_eq!(
            bonds[0].a,
            AtomRef::Mobile {
                residue: 0,
                local: 0
            }
        );
        assert_eq!(
            bonds[0].b,
            AtomRef::Mobile {
                residue: 0,
                local: 1
            }
        );
        assert_eq!(bonds[0].order, BondOrder::Double);
    }

    #[test]
    fn build_bonds_fixed_mobile() {
        let refs = vec![
            AtomRef::Fixed(0),
            AtomRef::Mobile {
                residue: 0,
                local: 0,
            },
        ];
        let df = vec![dreid_forge::Bond {
            i: 0,
            j: 1,
            order: BondOrder::Single,
        }];
        let bonds = build_bonds(&df, &refs);
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].a, AtomRef::Fixed(0));
        assert_eq!(
            bonds[0].b,
            AtomRef::Mobile {
                residue: 0,
                local: 0
            }
        );
        assert_eq!(bonds[0].order, BondOrder::Single);
    }

    #[test]
    fn build_bonds_order_preserved() {
        let refs = vec![AtomRef::Fixed(0), AtomRef::Fixed(1)];
        for order in [
            BondOrder::Single,
            BondOrder::Double,
            BondOrder::Triple,
            BondOrder::Aromatic,
        ] {
            let df = vec![dreid_forge::Bond { i: 0, j: 1, order }];
            let bonds = build_bonds(&df, &refs);
            assert_eq!(bonds.len(), 1);
            assert_eq!(bonds[0].order, order);
        }
    }

    #[test]
    fn dihedral_trans() {
        let angle = dihedral(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        );
        assert_abs_diff_eq!(angle, std::f32::consts::PI, epsilon = 1e-6);
    }

    #[test]
    fn dihedral_cis() {
        let angle = dihedral(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        );
        assert_abs_diff_eq!(angle, 0.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn dihedral_right_angle() {
        let angle = dihedral(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        );
        assert_abs_diff_eq!(angle, std::f32::consts::FRAC_PI_2, epsilon = 1e-6);
    }

    #[test]
    fn dihedral_sign_flips_on_reflection() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let above = dihedral(a, b, c, [0.0, 1.0, 1.0]);
        let below = dihedral(a, b, c, [0.0, 1.0, -1.0]);
        assert_abs_diff_eq!(above, -below, epsilon = 1e-6);
    }

    #[test]
    fn to_vec3_zeros() {
        assert_eq!(to_vec3([0.0, 0.0, 0.0]), Vec3::zero());
    }

    #[test]
    fn to_vec3_values() {
        assert_eq!(to_vec3([1.5, -2.5, 3.5]), Vec3::new(1.5, -2.5, 3.5));
    }

    #[test]
    fn dist_sq_same_point() {
        assert_eq!(dist_sq([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn dist_sq_unit_axis() {
        assert_eq!(dist_sq([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 1.0);
    }

    #[test]
    fn dist_sq_symmetric() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dist_sq(a, b), dist_sq(b, a));
    }

    #[test]
    fn dist_sq_diagonal() {
        assert_eq!(dist_sq([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]), 3.0);
    }

    #[test]
    fn cell_key_origin() {
        assert_eq!(cell_key([0.5, 0.5, 0.5], 1.0), (0, 0, 0));
    }

    #[test]
    fn cell_key_positive() {
        assert_eq!(cell_key([1.5, 2.5, 3.5], 1.0), (1, 2, 3));
    }

    #[test]
    fn cell_key_negative() {
        assert_eq!(cell_key([-0.5, -1.5, -2.5], 1.0), (-1, -2, -3));
    }

    #[test]
    fn cell_key_boundary() {
        assert_eq!(cell_key([1.0, 0.0, 0.0], 1.0), (1, 0, 0));
    }

    #[test]
    fn fmt_selector_without_icode() {
        assert_eq!(fmt_selector(&sel("A", 42)), "A 42");
    }

    #[test]
    fn fmt_selector_with_icode() {
        let s = ResidueSelector {
            chain_id: "B".into(),
            residue_id: 5,
            insertion_code: Some('X'),
        };
        assert_eq!(fmt_selector(&s), "B 5X");
    }
}
