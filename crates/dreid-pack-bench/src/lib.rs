mod bfactor;
mod chi;
mod config;
mod error;
mod format;
mod output;
mod residue;
mod rmsd;

use std::io::{BufRead, BufReader};
use std::time::Instant;

pub use config::BenchConfig;
pub use error::Error;
pub use format::Format;
pub use output::{BenchOutput, Residue, ResidueTable};
pub use residue::AminoAcid;

pub fn bench(
    mut reader: impl BufRead,
    fmt: Format,
    config: &BenchConfig,
) -> Result<BenchOutput, Error> {
    // ── P1: read entire stream into memory
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|e| Error::Pack(dreid_pack::io::Error::Io(e)))?;

    // ── P2: dreid-pack session
    let mut session = dreid_pack::io::read(
        BufReader::new(std::io::Cursor::new(&bytes)),
        fmt,
        &config.read,
    )?;

    let locators: Vec<(String, i32, Option<char>)> = session
        .mobile_residues()
        .map(|(chain, id, ins)| (chain.to_owned(), id, ins))
        .collect();

    // ── P3: B-factor filter
    let retain = bfactor::retain(&bytes, fmt, &locators, config.bfactor_percentile)?;

    // ── P4: pack
    let start = Instant::now();
    dreid_pack::pack::<()>(&mut session.system, &config.pack);
    let elapsed = start.elapsed();

    // ── P5: bio-forge structures
    let mut packed_buf = Vec::new();
    dreid_pack::io::write(&mut packed_buf, &session, fmt).map_err(Error::Pack)?;

    let (crystal, packed) = format::read_pair(&bytes, &packed_buf, fmt)?;

    // ── P6: per-residue analysis
    let mut table = ResidueTable::new();

    for (i, (chain, id, ins)) in locators.iter().enumerate() {
        if !retain[i] {
            continue;
        }

        let aa = match crystal
            .find_residue(chain, *id, *ins)
            .and_then(|r| r.standard_name)
            .and_then(residue::from_standard)
        {
            Some(aa) => aa,
            None => continue,
        };

        let chi_diff = chi::diff(&crystal, &packed, chain, *id, *ins, aa);
        let sc_rmsd = rmsd::sidechain(&crystal, &packed, chain, *id, *ins);

        table.push(aa, Residue { chi_diff, sc_rmsd });
    }

    Ok(BenchOutput { table, elapsed })
}
