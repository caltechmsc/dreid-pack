use thiserror::Error;

/// Errors that can occur during structure file I/O.
#[derive(Debug, Error)]
pub enum Error {
    /// OS-level read or write failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// PDB/mmCIF bio-structure load failure.
    #[error("parse error: {0}")]
    Parse(String),

    /// MOL2 hetero template parse failure.
    #[error("template parse error: {0}")]
    Template(String),

    /// Force-field parameterization failure.
    #[error("force-field parameterization error: {0}")]
    Forge(String),
}
