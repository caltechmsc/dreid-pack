use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use dreid_pack::PackConfig;
use dreid_pack::io::{
    BasisType, ChargeConfig, CleanConfig, DampingStrategy, EmbeddedQeqConfig, ForceFieldConfig,
    Format, HeteroQeqMethod, HeteroTemplate, HisStrategy, NucleicScheme, PackingScope,
    ProteinScheme, ProtonationConfig, QeqConfig, ReadConfig, ResidueSelector, SolverOptions,
    TopologyConfig, VdwPotential, WaterScheme,
};

use crate::args::{
    BasisArg, ChargesArgs, HeteroQeqArg, HisArg, InterfaceArgs, IoArgs, ListArgs,
    NucleicChargesArg, PackingArgs, PocketArgs, ProteinChargesArg, StructureArgs, SubCmd, VdwArg,
    WaterChargesArg,
};

pub fn format_from_path(path: &Path) -> Result<Format> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("pdb") => Ok(Format::Pdb),
        Some("cif" | "mmcif") => Ok(Format::Mmcif),
        Some(ext) => bail!("unsupported file extension: .{ext}"),
        None => bail!("missing file extension"),
    }
}

pub fn resolve_output(io: &IoArgs) -> PathBuf {
    if let Some(ref out) = io.output {
        return out.clone();
    }
    let stem = io.input.file_stem().unwrap_or_default().to_string_lossy();
    let ext = io.input.extension().unwrap_or_default().to_string_lossy();
    io.input.with_file_name(format!("{stem}-packed.{ext}"))
}

pub fn read_config(
    s: &StructureArgs,
    c: &ChargesArgs,
    p: &PackingArgs,
    scope: PackingScope,
) -> Result<ReadConfig> {
    Ok(ReadConfig {
        clean: CleanConfig {
            remove_water: s.no_water,
            remove_ions: s.no_ions,
            remove_hetero: s.no_hetero,
            ..CleanConfig::default()
        },
        protonation: ProtonationConfig {
            target_ph: s.ph,
            his_strategy: to_his(s.his),
            his_salt_bridge: true,
        },
        topology: TopologyConfig {
            templates: load_templates(&p.templates)?,
            disulfide_cutoff: 2.2,
        },
        ff: ForceFieldConfig {
            rules: read_optional_file(&s.ff_rules)?,
            params: read_optional_file(&s.ff_params)?,
            vdw: to_vdw(s.vdw),
            charge: p.electrostatics.map(|_| to_charge_config(c)),
        },
        scope,
    })
}

pub fn pack_config(p: &PackingArgs) -> PackConfig {
    PackConfig {
        electrostatics: p.electrostatics,
        sample_polar_h: !p.no_polar_h,
        include_input_conformation: p.include_input,
        self_energy_threshold: p.self_energy,
        rotamer_prob_cutoff: p.prob_cutoff,
    }
}

pub fn packing_scope(cmd: &SubCmd) -> Result<PackingScope> {
    match cmd {
        SubCmd::Full(_) => Ok(PackingScope::Full),
        SubCmd::Pocket(a) => pocket_scope(a),
        SubCmd::Interface(a) => interface_scope(a),
        SubCmd::List(a) => list_scope(a),
    }
}

fn pocket_scope(a: &PocketArgs) -> Result<PackingScope> {
    let anchor = a
        .ligand
        .as_deref()
        .map(parse_selector)
        .transpose()?
        .context("pocket mode requires --ligand CHAIN:RESID[:INS]")?;
    Ok(PackingScope::Pocket {
        anchor,
        radius: a.radius,
    })
}

fn interface_scope(a: &InterfaceArgs) -> Result<PackingScope> {
    let ga: Vec<String> = a
        .group_a
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    let gb: Vec<String> = a
        .group_b
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    if ga.is_empty() || gb.is_empty() {
        bail!("both --group-a and --group-b must be non-empty");
    }
    Ok(PackingScope::Interface {
        groups: [ga, gb],
        cutoff: a.cutoff,
    })
}

fn list_scope(a: &ListArgs) -> Result<PackingScope> {
    let selectors = a
        .residues
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(parse_selector)
        .collect::<Result<Vec<_>>>()?;
    if selectors.is_empty() {
        bail!("--residues must contain at least one selector");
    }
    Ok(PackingScope::List(selectors))
}

fn parse_selector(s: &str) -> Result<ResidueSelector> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        2 => Ok(ResidueSelector {
            chain_id: parts[0].to_owned(),
            residue_id: parts[1]
                .parse()
                .with_context(|| format!("invalid residue number: {}", parts[1]))?,
            insertion_code: None,
        }),
        3 => Ok(ResidueSelector {
            chain_id: parts[0].to_owned(),
            residue_id: parts[1]
                .parse()
                .with_context(|| format!("invalid residue number: {}", parts[1]))?,
            insertion_code: parts[2].chars().next(),
        }),
        _ => bail!("invalid selector '{s}': expected CHAIN:RESID or CHAIN:RESID:INS"),
    }
}

fn to_his(h: HisArg) -> HisStrategy {
    match h {
        HisArg::Network => HisStrategy::HbNetwork,
        HisArg::Hid => HisStrategy::Hid,
        HisArg::Hie => HisStrategy::Hie,
        HisArg::Random => HisStrategy::Random,
    }
}

fn to_vdw(v: VdwArg) -> VdwPotential {
    match v {
        VdwArg::Exp => VdwPotential::Buckingham,
        VdwArg::Lj => VdwPotential::LennardJones,
    }
}

fn to_charge_config(c: &ChargesArgs) -> ChargeConfig {
    ChargeConfig {
        protein_scheme: to_protein(c.protein_charges),
        nucleic_scheme: to_nucleic(c.nucleic_charges),
        water_scheme: to_water(c.water_charges),
        hetero_configs: Vec::new(),
        default_hetero_method: to_hetero_method(c),
    }
}

fn to_protein(p: ProteinChargesArg) -> ProteinScheme {
    match p {
        ProteinChargesArg::AmberFf14sb => ProteinScheme::AmberFFSB,
        ProteinChargesArg::AmberFf03 => ProteinScheme::AmberFF03,
        ProteinChargesArg::Charmm => ProteinScheme::Charmm,
    }
}

fn to_nucleic(n: NucleicChargesArg) -> NucleicScheme {
    match n {
        NucleicChargesArg::Amber => NucleicScheme::Amber,
        NucleicChargesArg::Charmm => NucleicScheme::Charmm,
    }
}

fn to_water(w: WaterChargesArg) -> WaterScheme {
    match w {
        WaterChargesArg::Tip3p => WaterScheme::Tip3p,
        WaterChargesArg::Tip3pFb => WaterScheme::Tip3pFb,
        WaterChargesArg::Spc => WaterScheme::Spc,
        WaterChargesArg::SpcE => WaterScheme::SpcE,
        WaterChargesArg::Opc3 => WaterScheme::Opc3,
    }
}

fn to_hetero_method(c: &ChargesArgs) -> HeteroQeqMethod {
    let solver = SolverOptions {
        tolerance: c.qeq_tol,
        max_iterations: c.qeq_iter,
        lambda_scale: c.qeq_lambda,
        hydrogen_scf: !c.no_h_scf,
        basis_type: to_basis(c.qeq_basis),
        damping: parse_damping(&c.qeq_damp),
    };
    let qeq = QeqConfig {
        total_charge: c.qeq_charge,
        solver_options: solver,
    };
    match c.hetero_qeq {
        HeteroQeqArg::Vacuum => HeteroQeqMethod::Vacuum(qeq),
        HeteroQeqArg::Embedded => HeteroQeqMethod::Embedded(EmbeddedQeqConfig {
            cutoff_radius: c.qeq_shell,
            qeq,
        }),
    }
}

fn to_basis(b: BasisArg) -> BasisType {
    match b {
        BasisArg::Sto => BasisType::Sto,
        BasisArg::Gto => BasisType::Gto,
    }
}

fn parse_damping(s: &str) -> DampingStrategy {
    if s == "none" {
        return DampingStrategy::None;
    }
    if let Some(rest) = s.strip_prefix("fixed:")
        && let Ok(d) = rest.parse::<f64>()
    {
        return DampingStrategy::Fixed(d);
    }
    if let Some(rest) = s.strip_prefix("auto:")
        && let Ok(d) = rest.parse::<f64>()
    {
        return DampingStrategy::Auto { initial: d };
    }
    DampingStrategy::Auto { initial: 0.4 }
}

fn load_templates(paths: &[PathBuf]) -> Result<Vec<HeteroTemplate>> {
    paths
        .iter()
        .map(|p| {
            let file = std::fs::File::open(p)
                .with_context(|| format!("cannot open template '{}'", p.display()))?;
            HeteroTemplate::read_mol2(BufReader::new(file))
                .with_context(|| format!("bad template '{}'", p.display()))
        })
        .collect()
}

fn read_optional_file(path: &Option<PathBuf>) -> Result<Option<String>> {
    match path {
        None => Ok(None),
        Some(p) => std::fs::read_to_string(p)
            .map(Some)
            .with_context(|| format!("cannot read '{}'", p.display())),
    }
}
