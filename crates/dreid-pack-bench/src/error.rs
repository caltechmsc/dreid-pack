use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Pack(#[from] dreid_pack::io::Error),

    #[error("b-factor: {0}")]
    Bfactor(String),

    #[error("bio-forge: {0}")]
    BioForge(String),
}
