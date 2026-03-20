use crate::model::residue::ResidueType;

/// Sidechain atom names for each residue type, in the same order as `rotamer::SidechainCoords::as_slice()`.
#[rustfmt::skip]
pub fn sidechain(rt: ResidueType) -> &'static [&'static str] {
    match rt {
        ResidueType::Gly => &[],
        ResidueType::Ala => &["CB", "HB1", "HB2", "HB3"],
        ResidueType::Val => &["CB", "CG1", "CG2", "HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23"],
        ResidueType::Cym => &["CB", "SG", "HB2", "HB3"],
        ResidueType::Cyx => &["CB", "SG", "HB2", "HB3"],
        ResidueType::Cys => &["CB", "SG", "HB2", "HB3", "HG"],
        ResidueType::Ser => &["CB", "OG", "HB2", "HB3", "HG"],
        ResidueType::Thr => &["CB", "OG1", "CG2", "HB", "HG1", "HG21", "HG22", "HG23"],
        ResidueType::Pro => &["CB", "CG", "CD", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3"],
        ResidueType::Asp => &["CB", "CG", "OD1", "OD2", "HB2", "HB3"],
        ResidueType::Asn => &["CB", "CG", "OD1", "ND2", "HB2", "HB3", "HD21", "HD22"],
        ResidueType::Ile => &["CB", "CG1", "CG2", "CD1", "HB", "HG12", "HG13", "HG21", "HG22", "HG23", "HD11", "HD12", "HD13"],
        ResidueType::Leu => &["CB", "CG", "CD1", "CD2", "HB2", "HB3", "HG", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23"],
        ResidueType::Phe => &["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HZ"],
        ResidueType::Tym => &["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2"],
        ResidueType::Hid => &["CB", "CG", "ND1", "CD2", "CE1", "NE2", "HB2", "HB3", "HD1", "HD2", "HE1"],
        ResidueType::Hie => &["CB", "CG", "ND1", "CD2", "CE1", "NE2", "HB2", "HB3", "HD2", "HE1", "HE2"],
        ResidueType::Hip => &["CB", "CG", "ND1", "CD2", "CE1", "NE2", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2"],
        ResidueType::Trp => &["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2", "HB2", "HB3", "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"],
        ResidueType::Ash => &["CB", "CG", "OD1", "OD2", "HB2", "HB3", "HD2"],
        ResidueType::Tyr => &["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HH"],
        ResidueType::Met => &["CB", "CG", "SD", "CE", "HB2", "HB3", "HG2", "HG3", "HE1", "HE2", "HE3"],
        ResidueType::Glu => &["CB", "CG", "CD", "OE1", "OE2", "HB2", "HB3", "HG2", "HG3"],
        ResidueType::Gln => &["CB", "CG", "CD", "OE1", "NE2", "HB2", "HB3", "HG2", "HG3", "HE21", "HE22"],
        ResidueType::Glh => &["CB", "CG", "CD", "OE1", "OE2", "HB2", "HB3", "HG2", "HG3", "HE2"],
        ResidueType::Arg => &["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE", "HH11", "HH12", "HH21", "HH22"],
        ResidueType::Arn => &["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE", "HH11", "HH12", "HH21"],
        ResidueType::Lyn => &["CB", "CG", "CD", "CE", "NZ", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE2", "HE3", "HZ1", "HZ2"],
        ResidueType::Lys => &["CB", "CG", "CD", "CE", "NZ", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE2", "HE3", "HZ1", "HZ2", "HZ3"],
    }
}
