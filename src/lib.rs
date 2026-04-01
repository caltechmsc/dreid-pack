//! # DREID-Pack
//!
//! **Full-atom protein side-chain packing powered by the DREIDING force field.**
//!
//! # Overview
//!
//! DREID-Pack determines the Global Minimum Energy Conformation (GMEC) of protein side chains.
//! The full pipeline:
//!
//! 1. **Structure Preparation** — Parse PDB/mmCIF, rebuild missing heavy atoms (Kabsch SVD
//!    alignment), detect disulfide bonds, assign protonation states by pH and pKa, resolve
//!    histidine tautomers via H-bond network analysis, add hydrogens, build covalent topology.
//! 2. **Force-Field Parameterization** — Perceive molecular graph (rings, aromaticity,
//!    hybridization), assign DREIDING atom types via rule engine, compute partial charges
//!    (AMBER/CHARMM lookup for biopolymers, QEq with exact STO integrals for ligands),
//!    generate force-field parameters.
//! 3. **Rotamer Sampling** — Dunbrack 2010 backbone-dependent library with cis-Pro detection
//!    and optional polar-hydrogen expansion; NERF coordinate generation.
//! 4. **Self-Energy Pruning** — Sidechain vs fixed-scaffold energy + rotamer preference bias;
//!    dead conformers discarded by threshold.
//! 5. **Pair-Energy Computation** — Sidechain vs sidechain for every edge in the spatial
//!    contact graph.
//! 6. **Dead-End Elimination** — Goldstein + Split DEE, iterated to convergence with
//!    single-survivor absorption.
//! 7. **Tree-Decomposition DP** — MCS/min-fill elimination ordering; exact GMEC for treewidth
//!    ≤ 5, rank-1 edge decomposition fallback.
//!
//! The force field is DREIDING (VdW + hydrogen bond), with optional distance-dependent Coulomb
//! electrostatics. VdW supports both Buckingham (exp-6) and Lennard-Jones (12-6) forms.
//!
//! ## Why DREID-Pack
//!
//! **Physics.** DREIDING is a transferable, all-atom force field — atom types are automatically
//! assigned from chemical-environment graph topology, not a hand-coded residue-specific lookup
//! table. Buckingham exp-6 (default) gives a softer, more physical repulsive wall than LJ 12-6;
//! both forms are available. Explicit D–H···A hydrogen bonding evaluates all-atom
//! polar-hydrogen geometry. Optional distance-dependent Coulomb electrostatics add charge-based
//! discrimination.
//!
//! **Chemistry.** Missing heavy atoms are rebuilt via SVD-based template alignment before any
//! packing begins. 29 residue types with full protonation state coverage
//! (Hid/Hie/Hip, Ash/Glh, Cys/Cym/Cyx, Lyn/Arn). All titratable residues
//! (Asp, Glu, Lys, Arg, Cys, Tyr) are assigned states by pH and pKa; histidine tautomers are
//! resolved via hydrogen-bond network scoring with salt-bridge priority override
//! (Nδ/Nε near COO⁻ → Hip). Disulfide bonds are detected by Sγ–Sγ distance and relabeled to
//! CYX. Polar-hydrogen torsions (Ser, Thr, Cys, Tyr, Ash, Glh, Lys, Lyn) are explicitly
//! sampled as discrete candidates. cis-Proline is detected by ω angle and dispatches to the
//! dedicated Dunbrack cis-Pro library.
//!
//! **Generality.** Any molecule — ligands, cofactors, nucleic acids, solvent, ions — is
//! parameterized automatically and participates as a fixed-scaffold atom. Biopolymer charges
//! come from AMBER/CHARMM lookup tables (29 protein residues × 5 terminal positions × 3
//! schemes, plus nucleic acids and water models); ligand charges are computed dynamically via
//! charge equilibration (QEq) with exact Slater-type orbital integrals, optionally embedded in
//! the electrostatic field of the surrounding protein. Four packing scopes: full protein, ligand
//! pocket, protein–protein interface, or explicit residue list.
//!
//! **Algorithm.** Self-energy threshold pruning eliminates dead conformers *before* the O(n²)
//! pair-energy phase. Spatial grid acceleration replaces brute-force all-pairs distance
//! computation. If the pruned interaction graph has treewidth > 5, rank-1 edge decomposition
//! progressively factors weak pair couplings into self-energy until the graph becomes tractable
//! — no bag-size explosion. Connected components are solved in parallel.
//!
//! **Engineering.** Design philosophy: maximize compile-time computation, then setup-time
//! precomputation, then minimize runtime cost.
//!
//! - **Rotamer library** (`dunbrack`): the build script precomputes sin/cos of all 740K χ mean
//!   angles across the full φ/ψ grid and bakes the entire Dunbrack 2010 database (~28 MB) into
//!   `.rodata`. At query time, bilinear interpolation uses circular weighted means on the
//!   precomputed sin/cos pairs — the only runtime trig is a single branchless `atan2f` per χ
//!   angle.
//! - **Coordinate builder** (`rotamer`): the build script code-generates a straight-line NERF
//!   `build()` function per residue type, with all fixed torsion and bond-angle sin/cos baked as
//!   `f32` immediates. Only the runtime-variable χ and polar-H angles call `sincosf` — for Arg
//!   (18 atoms, 4χ), 4 of 36 trig evaluations remain at runtime; for simpler residues, zero.
//! - **Energy kernels** (`dreid-kernel`): stateless, `#[inline(always)]` potential functions.
//!   `precompute()` converts physical constants (D₀, R₀, V, φ₀) into optimized parameter
//!   tuples at system-setup time, avoiding repeated sqrt/trig/exp in the energy hot loop.
//! - **QEq integrals** (`cheq`/`sto-ns`): the QEq J-matrix uses exact two-center Coulomb
//!   integrals over Slater-type ns orbitals via ellipsoidal coordinate expansion
//!   (Rappé & Goddard, 1991) — no Gaussian approximation.
//! - **Dispatch**: Buckingham vs LJ and Coulomb on/off are resolved at compile time via `const`
//!   generics — zero runtime branching in the inner loop.
//! - **Parallelism**: every compute-intensive phase — sampling, self-energy, pair-energy, DEE
//!   convergence, subgraph DP — runs on `rayon`'s work-stealing thread pool.
//! - **I/O**: PDB and mmCIF, both read and write.
//!
//! # Installation
//!
//! ```toml
//! [dependencies]
//! dreid-pack = "0.1.0"
//! ```
//!
//! Set `default-features = false` to exclude the CLI dependencies (`clap`, `anyhow`,
//! `indicatif`, `console`).
//!
//! # Usage
//!
//! ## End-to-End Example
//!
//! ```no_run
//! use dreid_pack::io::{self, Format, ReadConfig};
//! use dreid_pack::{PackConfig, pack};
//! use std::io::{BufReader, BufWriter};
//!
//! // Read and parameterize
//! let reader = BufReader::new(std::fs::File::open("input.pdb")?);
//! let mut session = io::read(reader, Format::Pdb, &ReadConfig::default())?;
//!
//! // Pack with default settings
//! pack::<()>(&mut session.system, &PackConfig::default());
//!
//! // Write result
//! let writer = BufWriter::new(std::fs::File::create("output.pdb")?);
//! io::write(writer, &session, Format::Pdb)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`pack::<()>`][pack()] uses zero-cost no-op progress tracking. Implement the [`Progress`]
//! trait for custom phase-level callbacks.
//!
//! ## Key Items
//!
//! | Item | Role |
//! |:-----|:-----|
//! | [`pack()`] | Pack all mobile side chains to GMEC |
//! | [`PackConfig`] | Packing algorithm settings (electrostatics, thresholds, polar-H) |
//! | [`Progress`] | Phase-level progress callback trait |
//! | [`System`] | Molecular system: mobile residues + fixed scaffold + FF params |
//! | [`Residue`] | One packable residue slot with backbone geometry and sidechain atoms |
//!
//! For io items, see the [`io`] module documentation.

mod model;
mod pack;

pub mod io;

pub use model::residue::ResidueType;
pub use model::system::{
    BuckMatrix, BuckPair, FixedAtomPool, ForceFieldParams, HBondParams, LjMatrix, LjPair, Residue,
    SidechainAtoms, System, VdwMatrix,
};
pub use model::types::{TypeIdx, Vec3};

pub use pack::PackConfig;
pub use pack::Progress;
pub use pack::pack;
