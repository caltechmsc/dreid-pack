use std::io::Cursor;

use bio_forge::Structure;
use bio_forge::io::IoContext;

use crate::error::Error;

pub use dreid_pack::io::Format;

pub fn bio_read(bytes: &[u8], fmt: Format) -> Result<Structure, Error> {
    let reader = Cursor::new(bytes);
    let ctx = IoContext::default();
    match fmt {
        Format::Pdb => bio_forge::io::read_pdb_structure(std::io::BufReader::new(reader), &ctx)
            .map_err(|e| Error::BioForge(e.to_string())),
        Format::Mmcif => bio_forge::io::read_mmcif_structure(std::io::BufReader::new(reader), &ctx)
            .map_err(|e| Error::BioForge(e.to_string())),
    }
}
