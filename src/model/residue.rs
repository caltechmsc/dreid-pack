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

    /// Whether this residue participates in rotamer packing.
    pub const fn is_packable(self) -> bool {
        !matches!(self, Self::Gly | Self::Ala | Self::Cyx) // TODO: To support disulfide optimization, replace this with `self.n_chi() > 0`.
    }

    /// Symmetry period of the polar-hydrogen torsion (radians).
    ///
    /// Returns `0.0` for types with no rotatable polar hydrogen.
    pub const fn polar_h_period(self) -> f32 {
        use core::f32::consts::{PI, TAU};
        match self {
            Self::Lys => TAU / 3.0,
            Self::Tyr => PI,
            Self::Ser | Self::Thr | Self::Cys | Self::Ash | Self::Glh | Self::Lyn => TAU,
            _ => 0.0,
        }
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

#[cfg(test)]
mod tests {
    use super::ResidueType;

    #[test]
    fn n_atoms_bounds() {
        assert_eq!(ResidueType::Gly.n_atoms(), 0);
        assert_eq!(ResidueType::Trp.n_atoms(), 18);
        assert_eq!(ResidueType::Arg.n_atoms(), 18);
        assert_eq!(ResidueType::Ser.n_atoms(), 5);
    }

    #[test]
    fn is_packable_gly_ala() {
        assert!(!ResidueType::Gly.is_packable());
        assert!(!ResidueType::Ala.is_packable());
        assert!(ResidueType::Ser.is_packable());
    }

    #[test]
    fn polar_h_period_values() {
        use std::f32::consts::{PI, TAU};

        assert_eq!(ResidueType::Ser.polar_h_period(), TAU);
        assert_eq!(ResidueType::Thr.polar_h_period(), TAU);
        assert_eq!(ResidueType::Cys.polar_h_period(), TAU);
        assert_eq!(ResidueType::Ash.polar_h_period(), TAU);
        assert_eq!(ResidueType::Glh.polar_h_period(), TAU);
        assert_eq!(ResidueType::Lyn.polar_h_period(), TAU);
        assert_eq!(ResidueType::Tyr.polar_h_period(), PI);
        assert_eq!(ResidueType::Lys.polar_h_period(), TAU / 3.0);
        assert_eq!(ResidueType::Val.polar_h_period(), 0.0);
        assert_eq!(ResidueType::Gly.polar_h_period(), 0.0);
    }
}
