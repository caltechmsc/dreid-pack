use bio_forge::Structure;
use bio_forge::io::IoContext;
use bio_forge::ops::{CleanConfig, clean_structure};

use crate::error::Error;

pub use dreid_pack::io::Format;

fn read(bytes: &[u8], fmt: Format) -> Result<Structure, Error> {
    let reader = std::io::BufReader::new(std::io::Cursor::new(bytes));
    let ctx = IoContext::default();
    match fmt {
        Format::Pdb => bio_forge::io::read_pdb_structure(reader, &ctx)
            .map_err(|e| Error::BioForge(e.to_string())),
        Format::Mmcif => bio_forge::io::read_mmcif_structure(reader, &ctx)
            .map_err(|e| Error::BioForge(e.to_string())),
    }
}

pub fn read_pair(
    crystal_bytes: &[u8],
    packed_bytes: &[u8],
    fmt: Format,
) -> Result<(Structure, Structure), Error> {
    let clean_cfg = CleanConfig {
        remove_water: true,
        remove_ions: true,
        remove_hydrogens: true,
        remove_hetero: true,
        ..Default::default()
    };

    let mut crystal = read(crystal_bytes, fmt)?;
    clean_structure(&mut crystal, &clean_cfg).map_err(|e| Error::BioForge(e.to_string()))?;

    let mut packed = read(packed_bytes, fmt)?;
    clean_structure(&mut packed, &clean_cfg).map_err(|e| Error::BioForge(e.to_string()))?;

    Ok((crystal, packed))
}
