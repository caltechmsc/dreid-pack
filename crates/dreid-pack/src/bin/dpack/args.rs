use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "dpack", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: SubCmd,

    /// Suppress all progress output.
    #[arg(short, long, global = true)]
    pub quiet: bool,
}

#[derive(Subcommand)]
pub enum SubCmd {
    /// Pack all standard residues.
    #[command(alias = "f")]
    Full(FullArgs),

    /// Pack residues near a ligand pocket.
    #[command(alias = "p")]
    Pocket(PocketArgs),

    /// Pack residues at a protein-protein interface.
    #[command(alias = "i")]
    Interface(InterfaceArgs),

    /// Pack an explicit residue list.
    #[command(alias = "l")]
    List(ListArgs),
}

impl SubCmd {
    pub fn common(&self) -> (&IoArgs, &StructureArgs, &ChargesArgs, &PackingArgs) {
        match self {
            Self::Full(a) => (&a.io, &a.structure, &a.charges, &a.packing),
            Self::Pocket(a) => (&a.io, &a.structure, &a.charges, &a.packing),
            Self::Interface(a) => (&a.io, &a.structure, &a.charges, &a.packing),
            Self::List(a) => (&a.io, &a.structure, &a.charges, &a.packing),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Full(_) | Self::List(_) => "packed",
            Self::Pocket(_) => "pocket",
            Self::Interface(_) => "interface",
        }
    }
}

#[derive(Args)]
pub struct FullArgs {
    #[command(flatten)]
    pub io: IoArgs,
    #[command(flatten)]
    pub structure: StructureArgs,
    #[command(flatten)]
    pub charges: ChargesArgs,
    #[command(flatten)]
    pub packing: PackingArgs,
}

#[derive(Args)]
pub struct PocketArgs {
    #[command(flatten)]
    pub io: IoArgs,
    #[command(flatten)]
    pub structure: StructureArgs,
    #[command(flatten)]
    pub charges: ChargesArgs,
    #[command(flatten)]
    pub packing: PackingArgs,

    /// Anchor ligand selector (CHAIN:RESID or CHAIN:RESID:INS).
    #[arg(short = 'L', long = "ligand")]
    pub ligand: Option<String>,

    /// Pocket heavy-atom distance cutoff (Å).
    #[arg(short, long, default_value_t = 8.0)]
    pub radius: f32,
}

#[derive(Args)]
pub struct InterfaceArgs {
    #[command(flatten)]
    pub io: IoArgs,
    #[command(flatten)]
    pub structure: StructureArgs,
    #[command(flatten)]
    pub charges: ChargesArgs,
    #[command(flatten)]
    pub packing: PackingArgs,

    /// First chain group (comma-separated, e.g. A,B).
    #[arg(short = 'A', long = "group-a")]
    pub group_a: String,

    /// Second chain group (comma-separated, e.g. C,D).
    #[arg(short = 'B', long = "group-b")]
    pub group_b: String,

    /// Interface heavy-atom distance cutoff (Å).
    #[arg(short, long, default_value_t = 6.0)]
    pub cutoff: f32,
}

#[derive(Args)]
pub struct ListArgs {
    #[command(flatten)]
    pub io: IoArgs,
    #[command(flatten)]
    pub structure: StructureArgs,
    #[command(flatten)]
    pub charges: ChargesArgs,
    #[command(flatten)]
    pub packing: PackingArgs,

    /// Residue selectors (comma-separated, CHAIN:RESID or CHAIN:RESID:INS).
    #[arg(short, long = "residues")]
    pub residues: String,
}

#[derive(Args)]
pub struct IoArgs {
    /// Input structure file (PDB or mmCIF).
    pub input: PathBuf,

    /// Output structure file [default: <INPUT_STEM>-packed.<EXT>].
    pub output: Option<PathBuf>,
}

#[derive(Args)]
#[command(next_help_heading = "STRUCTURE")]
pub struct StructureArgs {
    /// Target pH for protonation state assignment.
    #[arg(long)]
    pub ph: Option<f64>,

    /// Histidine tautomer strategy.
    #[arg(long, default_value = "network")]
    pub his: HisArg,

    /// Remove crystallographic water molecules.
    #[arg(long)]
    pub no_water: bool,

    /// Remove monoatomic ions.
    #[arg(long)]
    pub no_ions: bool,

    /// Remove non-ion HETATM residues (ligands, cofactors).
    #[arg(long)]
    pub no_hetero: bool,

    /// VdW potential form.
    #[arg(long, default_value = "exp")]
    pub vdw: VdwArg,

    /// Custom atom-typing rules (TOML file).
    #[arg(long)]
    pub ff_rules: Option<PathBuf>,

    /// Custom force-field parameters (TOML file).
    #[arg(long)]
    pub ff_params: Option<PathBuf>,
}

#[derive(Args)]
#[command(next_help_heading = "CHARGES")]
pub struct ChargesArgs {
    /// Protein residue charge scheme.
    #[arg(long, default_value = "amber-ff14sb")]
    pub protein_charges: ProteinChargesArg,

    /// Nucleic acid residue charge scheme.
    #[arg(long, default_value = "amber")]
    pub nucleic_charges: NucleicChargesArg,

    /// Water molecule charge model.
    #[arg(long, default_value = "tip3p")]
    pub water_charges: WaterChargesArg,

    /// Default QEq method for hetero residues.
    #[arg(long, default_value = "embedded")]
    pub hetero_qeq: HeteroQeqArg,

    /// Embedded QEq environment shell radius (Å).
    #[arg(long, default_value_t = 10.0)]
    pub qeq_shell: f64,

    /// QEq target net charge (e).
    #[arg(long, default_value_t = 0.0)]
    pub qeq_charge: f64,

    /// QEq basis function type.
    #[arg(long, default_value = "sto")]
    pub qeq_basis: BasisArg,

    /// QEq orbital screening scale factor λ.
    #[arg(long, default_value_t = 0.5)]
    pub qeq_lambda: f64,

    /// QEq SCF convergence tolerance (RMS Δq).
    #[arg(long, default_value_t = 1e-6)]
    pub qeq_tol: f64,

    /// QEq maximum SCF iterations.
    #[arg(long, default_value_t = 2000)]
    pub qeq_iter: u32,

    /// QEq damping strategy (none | fixed:D | auto:D).
    #[arg(long, default_value = "auto:0.4")]
    pub qeq_damp: String,

    /// Disable hydrogen nonlinear SCF iterations.
    #[arg(long)]
    pub no_h_scf: bool,
}

#[derive(Args)]
#[command(next_help_heading = "PACKING")]
pub struct PackingArgs {
    /// Enable Coulomb electrostatics with ε(r) = D·r dielectric model.
    #[arg(short, long)]
    pub electrostatics: Option<f32>,

    /// Disable polar-hydrogen torsion sampling.
    #[arg(long)]
    pub no_polar_h: bool,

    /// Include input conformation as an additional candidate.
    #[arg(long)]
    pub include_input: bool,

    /// Non-standard residue template (MOL2 file, repeatable).
    #[arg(short = 'T', long = "template")]
    pub templates: Vec<PathBuf>,

    /// Self-energy pruning threshold (kcal/mol).
    #[arg(short = 'E', long, default_value_t = 30.0)]
    pub self_energy: f32,

    /// Minimum Dunbrack rotamer probability cutoff.
    #[arg(short = 'p', long, default_value_t = 0.0)]
    pub prob_cutoff: f32,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum HisArg {
    Network,
    Hid,
    Hie,
    Random,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum VdwArg {
    Exp,
    Lj,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ProteinChargesArg {
    #[value(name = "amber-ff14sb")]
    AmberFf14sb,
    #[value(name = "amber-ff03")]
    AmberFf03,
    Charmm,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum NucleicChargesArg {
    Amber,
    Charmm,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum WaterChargesArg {
    Tip3p,
    #[value(name = "tip3p-fb")]
    Tip3pFb,
    Spc,
    #[value(name = "spc-e")]
    SpcE,
    Opc3,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum HeteroQeqArg {
    Embedded,
    Vacuum,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum BasisArg {
    Sto,
    Gto,
}
