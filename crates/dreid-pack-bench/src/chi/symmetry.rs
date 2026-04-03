use crate::residue::AminoAcid;

pub fn has_symmetric_last_chi(aa: AminoAcid) -> bool {
    matches!(
        aa,
        AminoAcid::Asp
            | AminoAcid::Glu
            | AminoAcid::Phe
            | AminoAcid::Tyr
            | AminoAcid::Arg
            | AminoAcid::Asn
            | AminoAcid::Gln
            | AminoAcid::His
    )
}
