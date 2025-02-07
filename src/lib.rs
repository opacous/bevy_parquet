#![feature(trait_upcasting)]

mod config;
mod state;
mod writer;
mod persistence_tracking;

use arrow::array::{ArrayRef, Float32Builder, Int32Builder, RecordBatch, StringArray, StructArray};
use arrow::datatypes::{DataType, Field};
use bevy::ecs::component::ComponentId;
use bevy::prelude::*;
use bevy::reflect::{ReflectRef, TypeRegistry};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fmt::Debug;
use std::sync::Arc;
use thiserror::Error;

pub use config::ParquetConfig;
pub use state::ParquetState;

#[derive(Error, Debug)]
pub enum ParquetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parquet write error: {0}")]
    ParquetWrite(String),
    #[error("Component serialization error: {0}")]
    Serialization(String),
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

    // Log cluster analysis
    println!("\n[Cluster Analysis] Detected {} component clusters:", clusters.len());
    for (i, cluster) in clusters.iter().enumerate() {
        println!("Cluster {}: {:?}", i, cluster.iter().map(|(n,_)| n).collect::<Vec<_>>());
    }

    // Process each cluster as a row group
    for (i, cluster) in clusters.into_iter().enumerate() {
        println!("Processing cluster {:?}", cluster);
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let schema = create_arrow_schema(&cluster, world, type_registry);
        println!("Created schema: {:#?}", schema);

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
    println!("\n[Serialization] Processing component: {}", component_info.0);
    let mut values = Vec::with_capacity(entities.len());
    let components = world.components();

    for &entity in entities {
        println!("[Entity] Processing entity {:?}", entity);
        if let Some(entity_ref) = world.get_entity(entity) {
            let reflect = world
                .components()
                .get_info(component_info.1)
                .complain_msg("unable to get info for component")
                .and_then(|info| info.type_id().complain_msg("missing type info"))
                .and_then(|id| {
                    println!("Getting type ID {id:?}");
                    type_registry
                        .get(id)
                        .complain_msg("missing type in registry")
                })
                .and_then(|reg| {
                    println!("Getting type registration");
                    println!("{:?}", reg);
                    let retv = reg.data::<ReflectComponent>();
                    println!("reflected does {:?} have something", retv.is_some());
                    retv
                })
                .and_then(|reflect| {
                    println!(
                        "{:?} @ entity with ID: {}",
                        component_info.0,
                        entity_ref.id()
                    );
                    reflect
                        .reflect(entity_ref)
                        .complain_msg("unable to reflect")
                });

            if let Some(reflect) = reflect {
                let reflect_discrete = reflect.reflect_ref();
                let output_field = match reflect_discrete {
                    // TODO: Assuming all structs are
                    ReflectRef::Struct(inner) => {
                        let output_field_reflect = inner.field("output").unwrap_or(inner);
                        // let reflect_serializer =
                        //     ReflectSerializer::new(output_field_reflect, type_registry);
                        // serde_json::to_string(&reflect_serializer)
                        (format!("{:?}", output_field_reflect).to_string())
                    }
                    ReflectRef::Value(inner) => format!("{:?}", inner).to_string(),
                    ReflectRef::TupleStruct(inner) => {
                        (format!("{:?}", inner.field(0).unwrap()).to_string())
                    }
                    ReflectRef::Tuple(inner) => format!("{:?}", inner.field(0).unwrap()),
                    ReflectRef::List(inner) => {
                        format!(
                            "[{:?}]",
                            inner
                                .iter()
                                .fold("".to_string(), |acc, x| acc + &format!("{:?}, ", x))
                        )
                    }
                    ReflectRef::Array(inner) => format!(
                        "[{:?}]",
                        inner
                            .iter()
                            .fold("".to_string(), |acc, x| acc + &format!("{:?}, ", x))
                    ),
                    _ => format!("{:?}", reflect),
                };

                println!("field here looks like: {}", output_field);
                values.push(output_field);
                // let reflected_json =
                //     serde_json::from_str::<serde_json::Value>(&json).unwrap();
                // match reflected_json.clone() {
                //     // TODO: Tie this back to export type
                //     serde_json::Value::Number(map) => {
                //         values.push(Some(map.to_string()));
                //     }
                //     serde_json::Value::String(map) => {
                //         values.push(Some(map));
                //     }
                //     _ => {
                //         values.push(Some(json));
                //     }
                // }
            }
        }
    }

    Ok(Arc::new(StringArray::from(values)) as ArrayRef)
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
