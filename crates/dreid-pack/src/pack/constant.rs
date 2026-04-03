use crate::model::residue::ResidueType;

/// Van der Waals interaction cutoff radius (Å).
pub const VDW_CUTOFF: f32 = 6.0;
/// Van der Waals interaction cutoff distance squared (Å²).
pub const VDW_CUTOFF_SQ: f32 = VDW_CUTOFF * VDW_CUTOFF;

/// Hydrogen bond D···A interaction cutoff radius (Å).
pub const HBOND_CUTOFF: f32 = 4.5;
/// Hydrogen bond D···A interaction cutoff distance squared (Å²).
pub const HBOND_CUTOFF_SQ: f32 = HBOND_CUTOFF * HBOND_CUTOFF;

/// Coulomb interaction cutoff radius (Å).
pub const COULOMB_CUTOFF: f32 = 8.0;
/// Coulomb interaction cutoff distance squared (Å²).
pub const COULOMB_CUTOFF_SQ: f32 = COULOMB_CUTOFF * COULOMB_CUTOFF;

/// DREIDING Coulomb constant (kcal·Å·mol⁻¹·e⁻²).
pub const COULOMB_CONST: f32 = 332.0637;

/// DREIDING hydrogen bond cosine exponent.
pub const HBOND_N: usize = 4;

/// Maximum Dunbrack log-probability ratio cap (kcal/mol).
pub const ROTAMER_BIAS_CAP: f32 = 8.0;

// Invariant: `VDW_CUTOFF ≤ COULOMB_CUTOFF`.
const _: () = assert!(
    VDW_CUTOFF <= COULOMB_CUTOFF,
    "cutoff invariant violated: VDW_CUTOFF ≤ COULOMB_CUTOFF"
);

/// Returns the maximum non-bonded interaction cutoff radius (Å) across all
/// active potentials.
pub fn max_interaction_cutoff(electrostatics: Option<f32>) -> f32 {
    let main = match electrostatics {
        None => VDW_CUTOFF,
        Some(_) => COULOMB_CUTOFF,
    };
    main.max(HBOND_CUTOFF)
}

/// Returns the rotamer preference energy weight for the given [`ResidueType`].
pub fn rotamer_weight(rt: ResidueType) -> f32 {
    use ResidueType as T;
    match rt {
        T::Cys | T::Cym | T::Cyx => 16.5,
        T::Asp | T::Ash => 6.0,
        T::Glu | T::Glh => 3.0,
        T::Phe => 4.5,
        T::Hid | T::Hie | T::Hip => 9.0,
        T::Ile => 3.0,
        T::Lys | T::Lyn => 6.0,
        T::Leu => 6.0,
        T::Met => 4.5,
        T::Asn => 6.0,
        T::Pro => 4.5,
        T::Gln => 7.5,
        T::Arg | T::Arn => 4.5,
        T::Ser => 4.5,
        T::Thr => 6.0,
        T::Val => 6.0,
        T::Trp => 10.5,
        T::Tyr | T::Tym => 4.5,
        T::Gly | T::Ala => 0.0,
    }
}
