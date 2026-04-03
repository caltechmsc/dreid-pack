use std::collections::HashMap;

use dreid_pack::io::Format;

use crate::error::Error;

pub fn retain(
    bytes: &[u8],
    fmt: Format,
    locators: &[(String, i32, Option<char>)],
    percentile: f32,
) -> Result<Vec<bool>, Error> {
    if percentile >= 1.0 {
        return Ok(vec![true; locators.len()]);
    }

    let map = parse(bytes, fmt)?;

    let values: Vec<f32> = locators
        .iter()
        .filter_map(|loc| map.get(loc).copied())
        .collect();

    let cutoff = percentile_value(&values, percentile);

    Ok(locators
        .iter()
        .map(|loc| match map.get(loc) {
            Some(&bf) => bf < cutoff,
            None => true,
        })
        .collect())
}

type BfactorMap = HashMap<(String, i32, Option<char>), f32>;

fn parse(bytes: &[u8], fmt: Format) -> Result<BfactorMap, Error> {
    match fmt {
        Format::Pdb => parse_pdb(bytes),
        Format::Mmcif => parse_mmcif(bytes),
    }
}

fn percentile_value(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f32 - 1.0) * p.clamp(0.0, 1.0)).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn parse_pdb(bytes: &[u8]) -> Result<BfactorMap, Error> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error::Bfactor(e.to_string()))?;

    let mut accum: HashMap<(String, i32, Option<char>), (f64, u32)> = HashMap::new();

    for line in text.lines() {
        let bytes = line.as_bytes();
        if bytes.len() < 66 {
            continue;
        }
        if !line[..6].starts_with("ATOM") {
            continue;
        }

        let element = if bytes.len() >= 78 {
            line[76..78].trim()
        } else {
            let name = line[12..16].trim();
            if name.starts_with('H') || name.starts_with('D') || name == "HN" {
                continue;
            }
            ""
        };
        if element == "H" || element == "D" {
            continue;
        }

        let chain_id = line[21..22].to_string();
        let res_seq: i32 = line[22..26]
            .trim()
            .parse()
            .map_err(|e: std::num::ParseIntError| Error::Bfactor(e.to_string()))?;
        let ins_code = {
            let c = bytes[26] as char;
            if c == ' ' { None } else { Some(c) }
        };
        let bfactor: f64 = line[60..66]
            .trim()
            .parse()
            .map_err(|e: std::num::ParseFloatError| Error::Bfactor(e.to_string()))?;

        let entry = accum
            .entry((chain_id, res_seq, ins_code))
            .or_insert((0.0, 0));
        entry.0 += bfactor;
        entry.1 += 1;
    }

    Ok(accum
        .into_iter()
        .map(|(key, (sum, count))| (key, (sum / count as f64) as f32))
        .collect())
}

fn parse_mmcif(bytes: &[u8]) -> Result<BfactorMap, Error> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error::Bfactor(e.to_string()))?;

    let mut in_atom_site = false;
    let mut columns: Vec<String> = Vec::new();
    let mut data_lines: Vec<&str> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("loop_") {
            in_atom_site = false;
            columns.clear();
            data_lines.clear();
            continue;
        }

        if trimmed.starts_with("_atom_site.") {
            in_atom_site = true;
            columns.push(trimmed.to_string());
            continue;
        }

        if in_atom_site {
            if trimmed.starts_with('_') || trimmed.starts_with("loop_") || trimmed.starts_with('#')
            {
                break;
            }
            if trimmed.is_empty() {
                continue;
            }
            data_lines.push(trimmed);
        }
    }

    if columns.is_empty() {
        return Ok(HashMap::new());
    }

    let col = |name: &str| -> Option<usize> { columns.iter().position(|c| c == name) };

    let i_group = col("_atom_site.group_PDB");
    let i_element = col("_atom_site.type_symbol");
    let i_chain = col("_atom_site.auth_asym_id").or_else(|| col("_atom_site.label_asym_id"));
    let i_resid = col("_atom_site.auth_seq_id").or_else(|| col("_atom_site.label_seq_id"));
    let i_ins = col("_atom_site.pdbx_PDB_ins_code");
    let i_bfactor = col("_atom_site.B_iso_or_equiv");

    let i_chain = match i_chain {
        Some(i) => i,
        None => return Err(Error::Bfactor("missing chain ID column".into())),
    };
    let i_resid = match i_resid {
        Some(i) => i,
        None => return Err(Error::Bfactor("missing residue ID column".into())),
    };
    let i_bfactor = match i_bfactor {
        Some(i) => i,
        None => return Err(Error::Bfactor("missing B_iso_or_equiv column".into())),
    };

    let mut accum: HashMap<(String, i32, Option<char>), (f64, u32)> = HashMap::new();

    for line in &data_lines {
        let fields: Vec<&str> = split_cif_fields(line);
        if fields.len() <= i_bfactor || fields.len() <= i_chain || fields.len() <= i_resid {
            continue;
        }

        if let Some(ig) = i_group
            && fields.get(ig).is_some_and(|&g| g != "ATOM")
        {
            continue;
        }

        if let Some(ie) = i_element
            && let Some(&el) = fields.get(ie)
            && (el == "H" || el == "D")
        {
            continue;
        }

        let chain_id = fields[i_chain].to_string();
        let res_id: i32 = fields[i_resid]
            .parse()
            .map_err(|e: std::num::ParseIntError| Error::Bfactor(e.to_string()))?;
        let ins_code = i_ins.and_then(|i| {
            fields.get(i).and_then(|&s| {
                if s == "?" || s == "." {
                    None
                } else {
                    s.chars().next()
                }
            })
        });
        let bfactor: f64 = fields[i_bfactor]
            .parse()
            .map_err(|e: std::num::ParseFloatError| Error::Bfactor(e.to_string()))?;

        let entry = accum
            .entry((chain_id, res_id, ins_code))
            .or_insert((0.0, 0));
        entry.0 += bfactor;
        entry.1 += 1;
    }

    Ok(accum
        .into_iter()
        .map(|(key, (sum, count))| (key, (sum / count as f64) as f32))
        .collect())
}

fn split_cif_fields(line: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        while i < len && bytes[i] == b' ' {
            i += 1;
        }
        if i >= len {
            break;
        }

        if bytes[i] == b'\'' {
            let start = i + 1;
            i = start;
            while i < len && !(bytes[i] == b'\'' && (i + 1 >= len || bytes[i + 1] == b' ')) {
                i += 1;
            }
            fields.push(&line[start..i]);
            if i < len {
                i += 1;
            }
        } else {
            let start = i;
            while i < len && bytes[i] != b' ' {
                i += 1;
            }
            fields.push(&line[start..i]);
        }
    }

    fields
}
