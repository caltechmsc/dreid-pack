//! Packing session I/O — reading and writing biomolecular structures.
//!
//! This module converts between molecular structure files and the [`Session`]
//! type used by the packing engine. Reading parses the file, runs force-field
//! parameterization via `dreid-forge`, and partitions atoms into a fixed
//! scaffold and a set of mobile sidechains ready for repacking. Writing
//! reconstructs coordinates from a packed session and serializes them back to
//! the original file format.
//!
//! # Supported Formats
//!
//! | Format | Read | Write | Use Case |
//! |-|-|-|-|
//! | PDB | ✓ | ✓ | Biomolecules |
//! | mmCIF | ✓ | ✓ | Biomolecules (modern PDB) |
//!
//! # Entry Points
//!
//! - [`read()`] — Parse a structure file into a [`Session`].
//! - [`write()`] — Serialize a [`Session`] back to a structure file.
//!
//! # Configuration
//!
//! All read-time options are bundled in [`ReadConfig`]:
//!
//! - [`CleanConfig`] — Remove water, ions, and hetero residues.
//! - [`ProtonationConfig`] — Add missing hydrogens; resolve His tautomers.
//! - [`TopologyConfig`] — Bond perception and hetero residue templates.
//! - [`ForceFieldConfig`] — DREIDING parameterization and charge method.

mod config;
mod convert;
mod error;
mod order;
mod session;

pub use config::{
    BasisType, ChargeConfig, CleanConfig, DampingStrategy, EmbeddedQeqConfig, ForceFieldConfig,
    Format, HeteroChargeConfig, HeteroQeqMethod, HeteroTemplate, HisStrategy, NucleicScheme,
    ProteinScheme, ProtonationConfig, QeqConfig, ReadConfig, ResidueSelector, SolverOptions,
    TopologyConfig, VdwPotential, WaterScheme,
};
pub use convert::{read, write};
pub use error::Error;
pub use session::Session;
