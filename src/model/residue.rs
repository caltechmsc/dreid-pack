/// The 29 amino acid sidechain types (including different protonation state variants).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidueType {
    Gly = 0,
    Ala,
    Val,
    Cym,
    Cyx,
    Cys,
    Ser,
    Thr,
    Pro,
    Asp,
    Asn,
    Ile,
    Leu,
    Phe,
    Tym,
    Hid,
    Hie,
    Hip,
    Trp,
    Ash,
    Tyr,
    Met,
    Glu,
    Gln,
    Glh,
    Arg,
    Arn,
    Lyn,
    Lys,
}

impl ResidueType {
    /// Number of sidechain atoms (heavy + H). Maximum is 18 (Trp, Arg).
    pub const fn n_atoms(self) -> u8 {
        match self {
            Self::Gly => 0,
            Self::Ala | Self::Cym | Self::Cyx => 4,
            Self::Cys | Self::Ser => 5,
            Self::Asp => 6,
            Self::Ash => 7,
            Self::Thr | Self::Asn => 8,
            Self::Pro | Self::Glu => 9,
            Self::Val | Self::Glh => 10,
            Self::Hid | Self::Hie | Self::Met | Self::Gln => 11,
            Self::Hip => 12,
            Self::Ile | Self::Leu => 13,
            Self::Phe | Self::Tym => 14,
            Self::Tyr | Self::Lyn => 15,
            Self::Lys => 16,
            Self::Arn => 17,
            Self::Trp | Self::Arg => 18,
        }
    }

    /// Number of heavy-atom χ dihedral angles.
    pub const fn n_chi(self) -> u8 {
        match self {
            Self::Gly | Self::Ala => 0,
            Self::Val | Self::Cym | Self::Cyx | Self::Cys | Self::Ser | Self::Thr => 1,
            Self::Pro
            | Self::Asp
            | Self::Asn
            | Self::Ile
            | Self::Leu
            | Self::Phe
            | Self::Tym
            | Self::Hid
            | Self::Hie
            | Self::Hip
            | Self::Trp
            | Self::Ash
            | Self::Tyr => 2,
            Self::Met | Self::Glu | Self::Gln | Self::Glh => 3,
            Self::Arg | Self::Arn | Self::Lyn | Self::Lys => 4,
        }
    }

    /// Number of independently rotatable polar-hydrogen torsions.
    pub const fn n_polar_h(self) -> u8 {
        match self {
            Self::Cys
            | Self::Ser
            | Self::Thr
            | Self::Ash
            | Self::Tyr
            | Self::Glh
            | Self::Lyn
            | Self::Lys => 1,
            _ => 0,
        }
    }

    /// Whether this residue participates in rotamer packing (`n_chi > 0`).
    pub const fn is_packable(self) -> bool {
        self.n_chi() > 0
    }

    /// Conservative upper bound on sidechain radius (Å, from Cα).
    pub const fn reach(self) -> f32 {
        match self {
            Self::Gly => 0.0,
            Self::Ala => 2.5,
            Self::Pro => 3.5,
            Self::Val | Self::Cym | Self::Cyx | Self::Cys | Self::Ser | Self::Thr => 4.0,
            Self::Asp
            | Self::Asn
            | Self::Ile
            | Self::Leu
            | Self::Hid
            | Self::Hie
            | Self::Hip
            | Self::Ash => 5.0,
            Self::Phe | Self::Tym => 5.5,
            Self::Met | Self::Glu | Self::Gln | Self::Glh => 6.0,
            Self::Tyr => 6.5,
            Self::Trp => 7.0,
            Self::Lyn | Self::Lys => 7.5,
            Self::Arg | Self::Arn => 8.0,
        }
    }
}
