use bio_forge::StandardResidue;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AminoAcid {
    Ala = 0,
    Arg,
    Asn,
    Asp,
    Cys,
    Gln,
    Glu,
    Gly,
    His,
    Ile,
    Leu,
    Lys,
    Met,
    Phe,
    Pro,
    Ser,
    Thr,
    Trp,
    Tyr,
    Val,
}

impl AminoAcid {
    pub const COUNT: usize = 20;

    pub const ALL: [Self; Self::COUNT] = [
        Self::Ala,
        Self::Arg,
        Self::Asn,
        Self::Asp,
        Self::Cys,
        Self::Gln,
        Self::Glu,
        Self::Gly,
        Self::His,
        Self::Ile,
        Self::Leu,
        Self::Lys,
        Self::Met,
        Self::Phe,
        Self::Pro,
        Self::Ser,
        Self::Thr,
        Self::Trp,
        Self::Tyr,
        Self::Val,
    ];

    pub const fn code(self) -> &'static str {
        match self {
            Self::Ala => "ALA",
            Self::Arg => "ARG",
            Self::Asn => "ASN",
            Self::Asp => "ASP",
            Self::Cys => "CYS",
            Self::Gln => "GLN",
            Self::Glu => "GLU",
            Self::Gly => "GLY",
            Self::His => "HIS",
            Self::Ile => "ILE",
            Self::Leu => "LEU",
            Self::Lys => "LYS",
            Self::Met => "MET",
            Self::Phe => "PHE",
            Self::Pro => "PRO",
            Self::Ser => "SER",
            Self::Thr => "THR",
            Self::Trp => "TRP",
            Self::Tyr => "TYR",
            Self::Val => "VAL",
        }
    }

    pub const fn n_chi(self) -> u8 {
        match self {
            Self::Gly | Self::Ala => 0,
            Self::Val | Self::Cys | Self::Ser | Self::Thr => 1,
            Self::Pro
            | Self::Asp
            | Self::Asn
            | Self::Ile
            | Self::Leu
            | Self::Phe
            | Self::His
            | Self::Trp
            | Self::Tyr => 2,
            Self::Met | Self::Glu | Self::Gln => 3,
            Self::Arg | Self::Lys => 4,
        }
    }
}

pub fn from_standard(sr: StandardResidue) -> Option<AminoAcid> {
    match sr {
        StandardResidue::ALA => Some(AminoAcid::Ala),
        StandardResidue::ARG => Some(AminoAcid::Arg),
        StandardResidue::ASN => Some(AminoAcid::Asn),
        StandardResidue::ASP => Some(AminoAcid::Asp),
        StandardResidue::CYS => Some(AminoAcid::Cys),
        StandardResidue::GLN => Some(AminoAcid::Gln),
        StandardResidue::GLU => Some(AminoAcid::Glu),
        StandardResidue::GLY => Some(AminoAcid::Gly),
        StandardResidue::HIS => Some(AminoAcid::His),
        StandardResidue::ILE => Some(AminoAcid::Ile),
        StandardResidue::LEU => Some(AminoAcid::Leu),
        StandardResidue::LYS => Some(AminoAcid::Lys),
        StandardResidue::MET => Some(AminoAcid::Met),
        StandardResidue::PHE => Some(AminoAcid::Phe),
        StandardResidue::PRO => Some(AminoAcid::Pro),
        StandardResidue::SER => Some(AminoAcid::Ser),
        StandardResidue::THR => Some(AminoAcid::Thr),
        StandardResidue::TRP => Some(AminoAcid::Trp),
        StandardResidue::TYR => Some(AminoAcid::Tyr),
        StandardResidue::VAL => Some(AminoAcid::Val),
        _ => None,
    }
}
