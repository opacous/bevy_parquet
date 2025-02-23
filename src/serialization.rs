use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use bevy::ecs::component::ComponentId;
use bevy::prelude::*;
use bevy::reflect::{GetTypeRegistration, TypeRegistry};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Detects natural component clusters in the world
pub(crate) fn detect_component_clusters(world: &World) -> Vec<Vec<(String, ComponentId)>> {
    let mut entity_components: HashMap<Entity, HashSet<(String, ComponentId)>> = HashMap::new();
    let mut components_we_care_about = vec![];

    // First pass: collect all components per things reflected in TypeRegistry into a list we care about
    world
        .get_resource::<AppTypeRegistry>()
        .unwrap()
        .read()
        .iter()
        .for_each(|type_registration| {
            let type_id = type_registration.type_info().type_id();

            components_we_care_about.push(type_id);
        });

    // 1.5 pass: collect all components per entity
    for entity in world.iter_entities() {
        let mut components = HashSet::new();
        entity.archetype().components().for_each(|component_id| {
            if let Some(component_info) = world.components().get_info(component_id) {
                if components_we_care_about.contains(&component_info.type_id().unwrap()) {
                    components.insert((component_info.name().to_string(), component_info.id()));
                }
            }
        });

        // Skip if this entoity has no component we care about
        if components.is_empty() {
            continue;
        }
        entity_components.insert(entity.id(), components);
    }

    // Second pass: cluster similar component sets
    let mut clusters: Vec<HashSet<(String, ComponentId)>> = Vec::new();
    let mut processed_entities = HashSet::new();

    for (entity, components) in entity_components.iter() {
        if processed_entities.contains(entity) {
            continue;
        }

        let mut cluster = components.clone();
        processed_entities.insert(*entity);

        // Find all entities with similar component sets
        for (other_entity, other_components) in entity_components.iter() {
            if processed_entities.contains(other_entity) {
                continue;
            }

            let intersection: HashSet<_> =
                cluster.intersection(other_components).cloned().collect();
            let union: HashSet<_> = cluster.union(other_components).cloned().collect();

            // If sets are similar enough (>80% overlap), merge them
            if intersection.len() as f32 / union.len() as f32 > 0.8 {
                cluster = intersection;
                processed_entities.insert(*other_entity);
            }
        }

        if !cluster.is_empty() {
            clusters.push(cluster);
        }
    }

    // Convert HashSets to Vecs
    clusters
        .into_iter()
        .map(|set| set.into_iter().collect())
        .collect()
}

/// Creates an Arrow schema for a given set of components
pub(crate) fn create_arrow_schema(components: &[(String, ComponentId)]) -> Schema {
    let mut fields = Vec::new();

    // Add fields for each component
    // NOTE: Name here lokks something like "bevy::ecs::component::ComponentId"
    //       which is not what we want. This is jank but we are going to split on "::"
    //       to get the name of the component.

    for (name, type_id) in components {
        fields.push(Field::new(name.split("::").last().unwrap(), DataType::Utf8, true));
    }

    Schema::new(fields)
}

/// Creates UUID array for entities
pub(crate) fn create_uuid_array(entities: &[Entity]) -> ArrayRef {
    let values: Vec<Option<String>> = entities
        .iter()
        .map(|e| Some(e.to_bits().to_string()))
        .collect();

    Arc::new(StringArray::from(values)) as ArrayRef
}
