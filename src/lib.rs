#![feature(trait_upcasting)]

use arrow::array::{ArrayRef, Float32Builder, Int32Builder, RecordBatch, StringArray, StructArray};
use arrow::datatypes::{DataType, Field};
use bevy::ecs::component::ComponentId;
use bevy::prelude::*;
use bevy::reflect::serde::{ReflectSerializer, TypedReflectDeserializer};
use bevy::reflect::{ReflectKind, ReflectRef, TypeData, TypeRegistration, TypeRegistry};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

mod persistence_tracking;

#[derive(Error, Debug)]
pub enum ParquetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parquet write error: {0}")]
    ParquetWrite(String),
    #[error("Component serialization error: {0}")]
    Serialization(String),
}

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

/// Resource holding the current serialization state
#[derive(Default, Resource)]
pub struct ParquetState {
    /// Maps TypeIds to their column indices
    type_to_column: HashMap<ComponentId, usize>,
    /// Detected or manually specified component clusters
    component_clusters: Vec<Vec<(String, ComponentId)>>,
}

pub struct ParquetPlugin;

impl Plugin for ParquetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParquetConfig>()
            .init_resource::<ParquetState>();
    }
}

mod serialization;
use serialization::*;

/// Trigger manual serialization of the world state to parquet
pub fn serialize_world(world: &mut World) -> Result<(), ParquetError> {
    let binding = &world.resource::<AppTypeRegistry>().clone();
    let type_registry = &binding.read();

    let config = world.resource::<ParquetConfig>().clone();
    // Get component clusters (either from config or detect automatically)
    let clusters = if let Some(ref manual_clusters) = config.component_clusters {
        manual_clusters.clone()
    } else {
        detect_component_clusters(world)
    };
    println!("Detected Clusters: {:?}", clusters);
    let mut state = world.resource_mut::<ParquetState>();

    // TODO: This is rather stupid way to deal with ReflectedTypes that are NOT meant to be deserialized into parquet
    //       All this does is that it filters out clusters that does not have the PhantomPersistTag (which is...
    //       part of the Persistence module). Eventually do this better, but automated clustering will always
    //       have this proplem - the correct answer is to use the manual batching ComponentID specification.
    let clusters = clusters
        .into_iter()
        .filter_map(|cluster| {
            if cluster
                .iter()
                .any(|(name, _id)| name.contains("PhantomPersistTag"))
            {
                Some(cluster)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    state.component_clusters = clusters.clone();

    // Process each cluster as a row group
    for (i, cluster) in clusters.into_iter().enumerate() {
        println!("Processing cluster {:?}", cluster);
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let schema = create_arrow_schema(&cluster, world, type_registry);

        let mut writer = {
            // For subsequent clusters, open the file in append mode
            let file = std::fs::File::create(format!(
                "{}_{}.parquet",
                config.output_path,
                config.file_name.as_ref().unwrap_or({
                    let mut field_fold_name =
                        cluster.iter().fold("".to_string(), |acc, (name, _id)| {
                            let retv = name.to_string().split("::").last().unwrap().to_string();
                            println!("Filed Name: {}", retv);
                            acc + &retv + "_"
                        });

                    if field_fold_name.len() > 20 {
                        field_fold_name = field_fold_name.split_at(20).0.to_string();
                    }

                    &field_fold_name.clone()
                })
            ))
            .map_err(ParquetError::Io)?;
            ArrowWriter::try_new(
                file,
                Arc::new(schema),
                Some(config.writer_properties.clone()),
            )
        }
        .map_err(|e| ParquetError::ParquetWrite(e.to_string()))?;

        // Collect all entities that have these components
        let mut entities: Vec<Entity> = Vec::new();
        for entity in world.iter_entities() {
            if cluster
                .iter()
                .all(|component_pair| world.get_by_id(entity.id(), component_pair.1).is_some())
            {
                entities.push(entity.id());
            }
        }

        // Create arrays for each component
        let mut arrays = Vec::new();

        for type_id in cluster {
            let array = component_to_arrow_array(world, &entities, type_id.clone(), type_registry)?;
            arrays.push((type_id.0.clone(), array));
        }

        // Create RecordBatch and write it
        let record_batch = RecordBatch::try_from_iter(arrays.into_iter())
            .map_err(|e| ParquetError::ParquetWrite(e.to_string()))?;

        println!("Writing record batch");
        println!("{:?}", record_batch);

        writer
            .write(&record_batch)
            .map_err(|e| ParquetError::ParquetWrite(e.to_string()))?;

        writer
            .close()
            .map_err(|e| ParquetError::ParquetWrite(e.to_string()))?;
    }

    Ok(())
}

// TODO: Eventually to be able to take ExportType hint from the component and
fn component_to_arrow_array(
    world: &World,
    entities: &[Entity],
    component_info: (String, ComponentId),
    type_registry: &TypeRegistry,
) -> Result<ArrayRef, ParquetError> {
    let (name, cid) = component_info;
    let registry = type_registry.read();
    let component_info = world
        .components()
        .get_info(cid)
        .ok_or_else(|| ParquetError::Serialization("Component not registered".into()))?;
    let type_id = component_info
        .type_id()
        .ok_or_else(|| ParquetError::Serialization("Missing type ID".into()))?;
    let type_reg = registry
        .get(type_id)
        .ok_or_else(|| ParquetError::Serialization("Type not in registry".into()))?;

    // Initialize appropriate array builder based on type
    if let Some(vec3_reflect) = type_reg.downcast::<Vec3>() {
        // Handle Vec3 as struct array
        let mut x_builder = Float32Builder::new();
        let mut y_builder = Float32Builder::new();
        let mut z_builder = Float32Builder::new();

        for entity in entities {
            if let Some(vec3) = world.get::<Vec3>(*entity) {
                x_builder.append_value(vec3.x);
                y_builder.append_value(vec3.y);
                z_builder.append_value(vec3.z);
            } else {
                x_builder.append_null();
                y_builder.append_null();
                z_builder.append_null();
            }
        }

        Ok(Arc::new(StructArray::new(
            vec![
                Field::new("x", DataType::Float32, false),
                Field::new("y", DataType::Float32, false),
                Field::new("z", DataType::Float32, false),
            ]
            .into(),
            vec![
                Arc::new(x_builder.finish()),
                Arc::new(y_builder.finish()),
                Arc::new(z_builder.finish()),
            ],
            None,
        )))
    } else if let Some(float_reflect) = type_reg.downcast::<f32>() {
        let mut builder = Float32Builder::new();
        for entity in entities {
            if let Some(value) = world.get::<f32>(*entity) {
                builder.append_value(*value);
            } else {
                builder.append_null();
            }
        }
        Ok(Arc::new(builder.finish()))
    } else if let Some(int_reflect) = type_reg.downcast::<i32>() {
        let mut builder = Int32Builder::new();
        for entity in entities {
            if let Some(value) = world.get::<i32>(*entity) {
                builder.append_value(*value);
            } else {
                builder.append_null();
            }
        }
        Ok(Arc::new(builder.finish()))
    } else {
        // Fallback to string serialization for unsupported types
        let mut values = Vec::new();
        for entity in entities {
            let value = world
                .get_by_id(*entity, cid)
                .map(|c| {
                    let reflect =
                        ReflectComponent::from_world(world)
                            .reflect(c)
                            .ok_or_else(|| {
                                ParquetError::Serialization("Component reflection failed".into())
                            })?;
                    Ok::<_, ParquetError>(format!("{:?}", reflect))
                })
                .transpose()?;
            values.push(value);
        }
        Ok(Arc::new(StringArray::from(values)))
    }
}

pub trait Hope {
    fn hope(self);
    fn complain(self) -> Self;
    fn complain_msg(self, msg: &str) -> Self;
    fn relief_msg(self, msg: &str) -> Self;
}

impl<T, E: Debug> Hope for Result<T, E> {
    fn hope(self) {
        match self {
            Ok(..) => {}
            Err(e) => tracing::error!("failure: {:?}", e),
        }
    }

    fn complain(self) -> Self {
        match self {
            Ok(x) => Ok(x),
            Err(e) => {
                tracing::error!("failure: {:?}", e);
                Err(e)
            }
        }
    }
    fn complain_msg(self, msg: &str) -> Self {
        match self {
            Err(e) => {
                tracing::error!("{}: {:?}", msg, e);
                Err(e)
            }
            Ok(x) => Ok(x),
        }
    }

    fn relief_msg(self, msg: &str) -> Self {
        match self {
            Err(e) => Err(e),
            Ok(x) => {
                tracing::info!("{}", msg);
                Ok(x)
            }
        }
    }
}

impl<T> Hope for Option<T> {
    fn hope(self) {
        match self {
            Some(..) => {}
            None => tracing::error!("empty option"),
        }
    }

    fn complain(self) -> Self {
        match self {
            None => {
                tracing::error!("empty option");
                None
            }
            Some(x) => Some(x),
        }
    }

    fn complain_msg(self, msg: &str) -> Self {
        match self {
            Some(x) => Some(x),
            None => {
                tracing::error!("{}", msg);
                None
            }
        }
    }

    fn relief_msg(self, msg: &str) -> Self {
        match self {
            Some(x) => {
                tracing::info!("{}", msg);
                Some(x)
            }
            None => None,
        }
    }
}

impl Hope for bool {
    fn hope(self) {
        match self {
            true => {}
            false => tracing::error!("empty option"),
        }
    }

    fn complain(self) -> Self {
        match self {
            false => {
                tracing::error!("empty option");
                false
            }
            true => true,
        }
    }

    fn complain_msg(self, msg: &str) -> Self {
        match self {
            true => true,
            false => {
                tracing::error!("{}", msg);
                false
            }
        }
    }

    fn relief_msg(self, msg: &str) -> Self {
        match self {
            true => {
                tracing::info!("{}", msg);
                true
            }
            false => false,
        }
    }
}

pub trait Report {
    fn report_msg(self, msg: &str) -> Self;
}

impl<T: Debug> Report for T {
    fn report_msg(self, msg: &str) -> Self {
        tracing::debug!("{}: {:?}", msg, &self);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_initialization() {
        let mut app = App::new();
        app.add_plugins(ParquetPlugin);

        assert!(app.world().contains_resource::<ParquetConfig>());
        assert!(app.world().contains_resource::<ParquetState>());
    }
}
