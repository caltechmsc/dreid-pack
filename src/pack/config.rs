/// Configuration for [`pack()`](super::pack).
#[derive(Debug, Clone)]
pub struct PackConfig {
    /// Coulomb electrostatics with a distance-dependent dielectric model.
    ///
    /// `None` disables electrostatics; `Some(D)` enables the ε(r) = D·r model
    /// using dielectric coefficient D, where D = 1.0 is the standard unscaled
    /// form.
    ///
    /// Default `None`.
    pub electrostatics: Option<f32>,

    /// Sample additional polar-hydrogen orientations during conformer generation.
    ///
    /// When `true`, each polar-hydrogen torsion is sampled at multiple
    /// orientations determined by the hybridization of the parent heavy atom.
    ///
    /// Default `true`.
    pub sample_polar_h: bool,

    /// Include the input side-chain conformation as an additional candidate.
    ///
    /// When `true`, the side-chain coordinates already present in the input
    /// structure are included as an additional candidate conformation.
    ///
    /// Default `false`.
    pub include_input_conformation: bool,

    /// Minimum Dunbrack rotamer probability required to include a rotamer.
    ///
    /// Rotamers whose backbone-dependent probability falls below this threshold
    /// are discarded before packing begins. Set to `0.0` to include all rotamers
    /// in the library.
    ///
    /// Default `0.0`.
    pub rotamer_prob_cutoff: f32,
}

impl Default for PackConfig {
    fn default() -> Self {
        Self {
            electrostatics: None,
            sample_polar_h: true,
            include_input_conformation: false,
            rotamer_prob_cutoff: 0.0,
        }
    }
}
