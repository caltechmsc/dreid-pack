use std::collections::HashSet;

/// Structure file format for biomolecular systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Protein Data Bank format (`.pdb`).
    Pdb,
    /// Macromolecular Crystallographic Information File format (`.cif`).
    Mmcif,
}

/// Configuration for [`read()`](super::read).
#[derive(Debug, Clone, Default)]
pub struct ReadConfig {
    /// Atom and residue removal policy.
    pub clean: CleanConfig,
    /// Hydrogen addition and protonation state assignment.
    pub protonation: ProtonationConfig,
    /// Bond perception and hetero residue template matching.
    pub topology: TopologyConfig,
    /// Force-field parameterisation settings.
    pub ff: ForceFieldConfig,
}

/// Atom and residue removal policy applied before protonation.
#[derive(Debug, Clone, Default)]
pub struct CleanConfig {
    /// Remove all water molecules (HOH, WAT, etc.).
    pub remove_water: bool,
    /// Remove monoatomic ions (Na⁺, Cl⁻, etc.).
    pub remove_ions: bool,
    /// Remove all non-ion HETATM residues (ligands, cofactors).
    pub remove_hetero: bool,
    /// Remove residues whose names are in this set (case-sensitive).
    pub remove_residue_names: HashSet<String>,
    /// If non-empty, keep only residues whose names are in this set.
    pub keep_residue_names: HashSet<String>,
}

/// Hydrogen addition and protonation state assignment.
#[derive(Debug, Clone)]
pub struct ProtonationConfig {
    /// Target pH for protonation state assignment (e.g. 7.4).
    ///
    /// `None` adds only missing hydrogens using the default protonation
    /// states already present in the input structure.
    pub target_ph: Option<f64>,
    /// Histidine tautomer strategy.
    ///
    /// Default [`HisStrategy::HbNetwork`] optimises tautomer selection via
    /// hydrogen-bond network analysis.
    pub his_strategy: HisStrategy,
    /// Favour doubly-protonated HIP for histidines near carboxylate groups.
    ///
    /// When `true`, histidines forming salt bridges with ASP⁻, GLU⁻, or the
    /// C-terminal COO⁻ are assigned the HIP form before tautomer selection.
    ///
    /// Default `true`.
    pub his_salt_bridge: bool,
}

impl Default for ProtonationConfig {
    fn default() -> Self {
        Self {
            target_ph: None,
            his_strategy: HisStrategy::default(),
            his_salt_bridge: true,
        }
    }
}

/// Histidine protonation tautomer strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HisStrategy {
    /// Force the Nδ-protonated (HID) tautomer for all histidines.
    Hid,
    /// Force the Nε-protonated (HIE) tautomer for all histidines.
    Hie,
    /// Assign HID or HIE at random with equal probability.
    Random,
    /// Optimise tautomer selection via hydrogen-bond network analysis.
    #[default]
    HbNetwork,
}

/// Bond perception and hetero residue template matching.
#[derive(Debug, Clone)]
pub struct TopologyConfig {
    /// Templates for non-standard (HETATM) residue types.
    pub templates: Vec<HeteroTemplate>,
    /// S–S distance cutoff for disulfide bond detection (Å).
    pub disulfide_cutoff: f64,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            templates: Vec::new(),
            disulfide_cutoff: 2.2,
        }
    }
}

/// Topology template for a non-standard (HETATM) residue type.
#[derive(Debug, Clone)]
pub struct HeteroTemplate(#[allow(dead_code)] pub(crate) dreid_forge::io::Template); // FIXME: Remove allow dead code when this is actually used.

impl HeteroTemplate {
    /// Parses a MOL2 reader into a hetero residue template.
    ///
    /// # Errors
    ///
    /// Returns [`dreid_forge::io::Error`] if the MOL2 data is malformed or
    /// contains an invalid structure definition.
    pub fn read_mol2<R: std::io::BufRead>(reader: R) -> Result<Self, dreid_forge::io::Error> {
        // FIXME: Use Error type from this crate instead of re-exporting from dreid-forge.
        dreid_forge::io::read_mol2_template(reader).map(Self)
    }
}

/// Force-field parameterisation settings.
#[derive(Debug, Clone, Default)]
pub struct ForceFieldConfig {
    /// Custom atom-typing rules in TOML format.
    pub rules: Option<String>,
    /// Custom force-field parameters in TOML format.
    pub params: Option<String>,
    /// Van der Waals non-bonded potential form.
    ///
    /// Default [`VdwPotential::Buckingham`] (Buckingham exp-6 potential).
    pub vdw: VdwPotential,
    /// Partial charge assignment settings.
    pub charge: ChargeConfig,
}

/// Van der Waals non-bonded potential form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VdwPotential {
    /// Buckingham exponential-6 potential.
    #[default]
    Buckingham,
    /// Lennard-Jones 12-6 potential.
    LennardJones,
}

/// Partial charge assignment settings for the hybrid biomolecule/QEq scheme.
#[derive(Debug, Clone)]
pub struct ChargeConfig {
    /// Charge scheme for standard amino acid residues.
    ///
    /// Default [`ProteinScheme::AmberFFSB`] (AMBER ff99SB/ff14SB/ff19SB).
    pub protein_scheme: ProteinScheme,
    /// Charge scheme for standard nucleic acid residues.
    ///
    /// Default [`NucleicScheme::Amber`] (AMBER OL15/OL21/OL24/bsc1/OL3).
    pub nucleic_scheme: NucleicScheme,
    /// Charge model for water molecules.
    ///
    /// Default [`WaterScheme::Tip3p`] (TIP3P).
    pub water_scheme: WaterScheme,
    /// Per-residue QEq method overrides for specific hetero residues.
    pub hetero_configs: Vec<HeteroChargeConfig>,
    /// QEq method for hetero residues not listed in [`hetero_configs`](Self::hetero_configs).
    ///
    /// Default [`HeteroQeqMethod::Embedded`] (embedded QEq, polarised).
    pub default_hetero_method: HeteroQeqMethod,
}

impl Default for ChargeConfig {
    fn default() -> Self {
        Self {
            protein_scheme: ProteinScheme::default(),
            nucleic_scheme: NucleicScheme::default(),
            water_scheme: WaterScheme::default(),
            hetero_configs: Vec::new(),
            default_hetero_method: HeteroQeqMethod::default(),
        }
    }
}

/// Classical partial charge scheme for protein residues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProteinScheme {
    /// AMBER ff99SB/ff14SB/ff19SB.
    #[default]
    AmberFFSB,
    /// AMBER ff03.
    AmberFF03,
    /// CHARMM22/27/36/36m.
    Charmm,
}

/// Classical partial charge scheme for nucleic acid residues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NucleicScheme {
    /// AMBER OL15/OL21/OL24/bsc1/OL3.
    #[default]
    Amber,
    /// CHARMM C27/C36.
    Charmm,
}

/// Partial charge model for water molecules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaterScheme {
    /// TIP3P.
    #[default]
    Tip3p,
    /// TIP3P-FB.
    Tip3pFb,
    /// SPC.
    Spc,
    /// SPC/E.
    SpcE,
    /// OPC3.
    Opc3,
}

/// Residue selector for identifying a specific residue by chain position.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidueSelector {
    /// Chain identifier.
    pub chain_id: String,
    /// Residue sequence number.
    pub residue_id: i32,
    /// Insertion code, if any.
    pub insertion_code: Option<char>,
}

/// Per-residue QEq method override for a specific hetero residue.
#[derive(Debug, Clone)]
pub struct HeteroChargeConfig {
    /// Selector identifying the target residue.
    pub selector: ResidueSelector,
    /// QEq method to use for this residue.
    ///
    /// Default [`HeteroQeqMethod::Embedded`] (embedded QEq, polarised).
    pub method: HeteroQeqMethod,
}

/// QEq charge assignment method for hetero residues.
#[derive(Debug, Clone)]
pub enum HeteroQeqMethod {
    /// Vacuum QEq — residue treated as an isolated molecule.
    Vacuum(QeqConfig),
    /// Embedded QEq — residue polarised by surrounding fixed charges.
    Embedded(EmbeddedQeqConfig),
}

impl Default for HeteroQeqMethod {
    fn default() -> Self {
        Self::Embedded(EmbeddedQeqConfig::default())
    }
}

/// Configuration for QEq charge equilibration.
#[derive(Debug, Clone)]
pub struct QeqConfig {
    /// Target net charge of the system in elementary charge units.
    ///
    /// Default `0.0` (neutral).
    pub total_charge: f64,
    /// Numerical solver options.
    pub solver_options: SolverOptions,
}

impl Default for QeqConfig {
    fn default() -> Self {
        Self {
            total_charge: 0.0,
            solver_options: SolverOptions::default(),
        }
    }
}

/// Configuration for embedded QEq calculations.
#[derive(Debug, Clone)]
pub struct EmbeddedQeqConfig {
    /// Radius of the environment atom shell (Å).
    ///
    /// Default `10.0` Å.
    pub cutoff_radius: f64,
    /// QEq solver configuration for the hetero residue.
    pub qeq: QeqConfig,
}

impl Default for EmbeddedQeqConfig {
    fn default() -> Self {
        Self {
            cutoff_radius: 10.0,
            qeq: QeqConfig::default(),
        }
    }
}

/// Numerical options for the QEq solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverOptions {
    /// RMS change in partial charges used as the convergence criterion.
    ///
    /// Default `1e-6`.
    pub tolerance: f64,
    /// Maximum number of SCF iterations before early termination.
    ///
    /// Default `2000`.
    pub max_iterations: u32,
    /// Orbital screening scale factor λ (Rappé & Goddard, 1991).
    ///
    /// Default `0.5`.
    pub lambda_scale: f64,
    /// Update hydrogen hardness each iteration (nonlinear SCF term).
    ///
    /// Default `true`.
    pub hydrogen_scf: bool,
    /// Basis function type for Coulomb integral evaluation.
    ///
    /// Default [`BasisType::Sto`] (Slater-type orbitals).
    pub basis_type: BasisType,
    /// Charge update damping strategy.
    ///
    /// Default [`DampingStrategy::Auto`] (adaptive damping with initial factor 0.4).
    pub damping: DampingStrategy,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-6,
            max_iterations: 2000,
            lambda_scale: 0.5,
            hydrogen_scf: true,
            basis_type: BasisType::default(),
            damping: DampingStrategy::Auto { initial: 0.4 },
        }
    }
}

/// Basis function type for Coulomb integral evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BasisType {
    /// Gaussian-type orbitals — fast, slight approximation.
    Gto,
    /// Slater-type orbitals — exact, higher cost.
    #[default]
    Sto,
}

/// Charge update damping strategy for SCF convergence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DampingStrategy {
    /// No damping — fast but may oscillate.
    None,
    /// Fixed damping factor `d` where `0.0 < d ≤ 1.0`.
    Fixed(f64),
    /// Adaptive damping — adjusts automatically based on convergence behaviour.
    Auto { initial: f64 },
}
