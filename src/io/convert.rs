use super::config::{
    BasisType, ChargeConfig, CleanConfig, DampingStrategy, EmbeddedQeqConfig, ForceFieldConfig,
    Format, HeteroQeqMethod, HisStrategy, NucleicScheme, ProteinScheme, ProtonationConfig,
    QeqConfig, ReadConfig, SolverOptions, TopologyConfig, VdwPotential, WaterScheme,
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
/// parameterises with the DREIDING force field, and partitions atoms into
/// fixed scaffold and mobile sidechains.
///
/// # Errors
///
/// Returns [`Error::Parse`] if the structure data is malformed,
/// [`Error::Forge`] if force-field parameterisation fails, or
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

    build(forged)
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

fn build(forged: ForgedSystem) -> Result<Session, Error> {
    let bio = forged
        .system
        .bio_metadata
        .as_ref()
        .ok_or_else(|| Error::Parse("input system lacks biological metadata".into()))?;

    let groups = group_residues(bio);
    let mobile_metas = classify_mobile(&groups, bio)?;
    let n_atoms = forged.system.atoms.len();
    let atom_to_ref = build_atom_to_ref(n_atoms, &mobile_metas);
    let (ff, h_types) = build_ff(forged.atom_types.len(), &forged.potentials)?;
    let phi_psi = compute_phi_psi(&forged.system.atoms, bio, &mobile_metas, &groups);
    let (fixed_pool, fixed_meta) = build_fixed_pool(&forged, bio, &atom_to_ref, &h_types);
    let (mobile, mobile_meta) = build_mobile(&forged, &groups, &mobile_metas, &phi_psi, &h_types);
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
) -> Result<Vec<MobileMetadata>, Error> {
    let mut metas = Vec::new();

    for (gi, g) in groups.iter().enumerate() {
        if g.category != ResidueCategory::Standard {
            continue;
        }

        let rt = match residue_name_to_type(&g.residue_name) {
            Some(rt) if rt.is_packable() => rt,
            _ => continue,
        };

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

fn compute_phi_psi(
    atoms: &[dreid_forge::Atom],
    bio: &BioMetadata,
    mobile_metas: &[MobileMetadata],
    groups: &[ResidueGroup],
) -> Vec<(f32, f32)> {
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

            (phi, psi)
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
    phi_psi: &[(f32, f32)],
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
        let (phi, psi) = phi_psi[r];
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
// Group D — math helpers
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
