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
    use tracing::{debug_span, error, info, warn, instrument};

    // Create a tracing span for the entire component processing
    let _span = debug_span!(
        "component_to_arrow_array",
        component = %component_info.0,
        component_id = ?component_info.1
    ).entered();

    info!("Starting component serialization");
    let mut values = Vec::with_capacity(entities.len());
    let components = world.components();

    // Process each entity with tracing instrumentation
    #[instrument(skip_all, fields(entity = ?entity))]
    fn process_entity(
        entity: Entity,
        world: &World,
        component_info: &(String, ComponentId),
        type_registry: &TypeRegistry,
        values: &mut Vec<String>,
    ) {
        info!("Processing entity");
        
        let entity_ref = match world.get_entity(entity) {
            Some(e) => e,
            None => {
                warn!("Entity not found in world");
                return;
            }
        };

        debug!(
            component_count = entity_ref.archetype().components().count(),
            "Entity component count"
        );

        let reflect = world
            .components()
            .get_info(component_info.1)
            .map_err(|e| {
                error!(
                    error = ?e,
                    "Failed to get component info for {}",
                    component_info.0
                );
                e
            })
            .and_then(|info| {
                info!(
                    component_name = info.name(),
                    component_type_id = ?info.type_id(),
                    "Component info retrieved"
                );
                info.type_id().ok_or_else(|| {
                    error!("Missing type ID for component {}", component_info.0);
                    "Missing type ID"
                })
            })
            .and_then(|id| {
                debug!(type_id = ?id, "Looking up type registration");
                type_registry.get(id).ok_or_else(|| {
                    error!(
                        "Type ID {} not found in registry for component {}",
                        id, component_info.0
                    );
                    "Missing type registration"
                })
            })
            .and_then(|reg| {
                debug!(
                    type_registration = ?reg.type_info(),
                    "Retrieved type registration"
                );
                reg.data::<ReflectComponent>().ok_or_else(|| {
                    error!("No ReflectComponent data for {}", component_info.0);
                    "Missing ReflectComponent"
                })
            })
            .and_then(|reflect| {
                debug!("Reflecting component instance");
                reflect.reflect(entity_ref).ok_or_else(|| {
                    error!("Failed to reflect component on entity {:?}", entity);
                    "Reflection failed"
                })
            });

        match reflect {
            Ok(reflect) => {
                let reflect_discrete = reflect.reflect_ref();
                debug!(reflection_type = ?reflect_discrete, "Reflected component");

                let output_field = match reflect_discrete {
                    ReflectRef::Struct(inner) => {
                        debug!(
                            fields = ?inner.iter_fields().map(|f| f.name()).collect::<Vec<_>>(),
                            "Processing struct fields"
                        );
                        match inner.field("output") {
                            Some(output) => format!("{:?}", output),
                            None => {
                                warn!("No output field found, using full struct");
                                format!("{:?}", inner)
                            }
                        }
                    }
                    ReflectRef::Value(inner) => {
                        debug!(value_type = ?inner.get_represented_type_info(), "Processing value type");
                        format!("{:?}", inner)
                    }
                    ReflectRef::TupleStruct(inner) => {
                        debug!(
                            field_count = inner.field_len(),
                            "Processing tuple struct"
                        );
                        format!("{:?}", inner.field(0).unwrap())
                    }
                    ReflectRef::Tuple(inner) => {
                        debug!(
                            field_count = inner.field_len(),
                            "Processing tuple"
                        );
                        format!("{:?}", inner.field(0).unwrap())
                    }
                    ReflectRef::List(inner) => {
                        debug!(
                            item_count = inner.len(),
                            "Processing list"
                        );
                        format!(
                            "[{}]",
                            inner.iter()
                                .map(|x| format!("{:?}", x))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                    ReflectRef::Array(inner) => {
                        debug!(
                            length = inner.len(),
                            "Processing array"
                        );
                        format!(
                            "[{}]",
                            inner.iter()
                                .map(|x| format!("{:?}", x))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                    _ => {
                        warn!("Unhandled reflection type");
                        format!("{:?}", reflect)
                    }
                };

                info!(output_length = output_field.len(), "Serialized value");
                debug!(output_value = %output_field);
                values.push(output_field);
            }
            Err(e) => {
                warn!("Skipping entity due to error: {}", e);
            }
        }
    }

    info!(entity_count = entities.len(), "Processing entities");
    for &entity in entities {
        process_entity(entity, world, &component_info, type_registry, &mut values);
    }

    info!(
        values_count = values.len(),
        "Completed component serialization"
    );
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
