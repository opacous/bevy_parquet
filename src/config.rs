use bevy::prelude::*;
use bevy::ecs::component::ComponentId;
use parquet::file::properties::WriterProperties;

/// Configuration for the ParquetPlugin
#[derive(Clone, Resource)]
pub struct ParquetConfig {
    /// Path where the parquet file will be written
    pub output_path: String,
    pub file_name: Option<String>,
    /// Optional manual component clusters
    pub component_clusters: Option<Vec<Vec<(String, ComponentId)>>>,
    /// Parquet writer properties
    pub writer_properties: WriterProperties,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            output_path: "./".to_string(),
            file_name: None,
            component_clusters: None,
            writer_properties: WriterProperties::builder().build(),
        }
    }
}
