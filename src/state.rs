use bevy::prelude::*;
use bevy::ecs::component::ComponentId;
use std::collections::HashMap;

/// Resource holding the current serialization state
#[derive(Default, Resource)]
pub struct ParquetState {
    /// Maps TypeIds to their column indices
    pub(crate) type_to_column: HashMap<ComponentId, usize>,
    /// Detected or manually specified component clusters
    pub(crate) component_clusters: Vec<Vec<(String, ComponentId)>>,
}
