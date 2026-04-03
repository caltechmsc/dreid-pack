use bio_forge::Structure;

use crate::chi::geometry::dihedral;
use crate::residue::AminoAcid;

type Quad = [&'static str; 4];

const CHI_ATOMS: [&[Quad]; AminoAcid::COUNT] = {
    const ALA: &[Quad] = &[];

    const ARG: &[Quad] = &[
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ];

    const ASN: &[Quad] = &[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]];

    const ASP: &[Quad] = &[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]];

    const CYS: &[Quad] = &[["N", "CA", "CB", "SG"]];

    const GLN: &[Quad] = &[
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ];

    const GLU: &[Quad] = &[
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ];

    const GLY: &[Quad] = &[];

    const HIS: &[Quad] = &[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]];

    const ILE: &[Quad] = &[["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]];

    const LEU: &[Quad] = &[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]];

    const LYS: &[Quad] = &[
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ];

    const MET: &[Quad] = &[
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ];

    const PHE: &[Quad] = &[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]];

    const PRO: &[Quad] = &[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]];

    const SER: &[Quad] = &[["N", "CA", "CB", "OG"]];

    const THR: &[Quad] = &[["N", "CA", "CB", "OG1"]];

    const TRP: &[Quad] = &[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]];

    const TYR: &[Quad] = &[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]];

    const VAL: &[Quad] = &[["N", "CA", "CB", "CG1"]];

    [
        ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP,
        TYR, VAL,
    ]
};

pub fn extract(
    structure: &Structure,
    chain_id: &str,
    res_id: i32,
    ins_code: Option<char>,
    aa: AminoAcid,
) -> Vec<Option<f64>> {
    let quads = CHI_ATOMS[aa as u8 as usize];

    let residue = match structure.find_residue(chain_id, res_id, ins_code) {
        Some(r) => r,
        None => return vec![None; quads.len()],
    };

    quads
        .iter()
        .map(|[a, b, c, d]| {
            let pa = residue.atom(a)?;
            let pb = residue.atom(b)?;
            let pc = residue.atom(c)?;
            let pd = residue.atom(d)?;
            Some(dihedral(&pos(pa), &pos(pb), &pos(pc), &pos(pd)))
        })
        .collect()
}

fn pos(atom: &bio_forge::Atom) -> [f64; 3] {
    [atom.pos.x, atom.pos.y, atom.pos.z]
}
