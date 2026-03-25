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
pub const HBOND_N: usize = 2;

/// Returns the maximum non-bonded interaction cutoff radius (Å) across all
/// active potentials.
pub fn max_interaction_cutoff(electrostatics: Option<f32>) -> f32 {
    let coulomb = electrostatics.map(|_| COULOMB_CUTOFF).unwrap_or(0.0);
    VDW_CUTOFF.max(HBOND_CUTOFF).max(coulomb)
}
