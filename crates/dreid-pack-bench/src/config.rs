pub use dreid_pack::PackConfig;
pub use dreid_pack::io::ReadConfig;

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub read: ReadConfig,
    pub pack: PackConfig,
    pub bfactor_percentile: f32,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            read: ReadConfig::default(),
            pack: PackConfig::default(),
            bfactor_percentile: 0.75,
        }
    }
}
