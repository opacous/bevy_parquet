use arrow::array::Array;
use arrow::datatypes::Fields;
use {
    arrow::{
        array::{ArrayRef, StringArray},
        datatypes::{DataType, Field, Schema},
    },
    bevy::{
        ecs::component::ComponentId,
        math::Vec3,
        prelude::*,
        reflect::{TypeInfo, TypeRegistry},
    },
    std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    },
};

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
pub(crate) fn create_arrow_schema(
    components: &[(String, ComponentId)],
    world: &World,
    type_registry: &TypeRegistry,
) -> Schema {
    let mut fields = Vec::new();
    let registry = type_registry;

    for (component_name, component_id) in components {
        // Get type information from ComponentId
        let type_info = world
            .components()
            .get_info(*component_id)
            .expect("Component not registered");
        let type_id = type_info.type_id().expect("Missing type ID");
        let type_reg = registry.get(type_id).expect("Type not in registry");

        // Map to Arrow type
        let data_type = if type_reg.data::<ReflectComponent>().is_some() {
            match type_reg.type_info() {
                TypeInfo::Struct(s) => {
                    // Directly handle Vec3 without reflection
                    if s.type_path() == "bevy::math::Vec3" {
                        DataType::Struct(Fields::from(vec![
                            Field::new("x", DataType::Float32, false),
                            Field::new("y", DataType::Float32, false),
                            Field::new("z", DataType::Float32, false),
                        ]))
                    } else {
                        // Existing reflection-based handling for other structs
                        DataType::Struct(
                            (0..s.field_len())
                                .map(|i| s.field_at(i).unwrap())
                                .map(|field| {
                                    Field::new(
                                        field.name(),
                                        map_reflect_kind_to_arrow(field.ty().as_value().unwrap()),
                                        true,
                                    )
                                })
                                .collect(),
                        )
                    }
                }
                TypeInfo::Value(v) => match v.type_path() {
                    "f32" => DataType::Float32,
                    "f64" => DataType::Float64,
                    "i32" => DataType::Int32,
                    "i64" => DataType::Int64,
                    "u32" => DataType::UInt32,
                    "u64" => DataType::UInt64,
                    "bool" => DataType::Boolean,
                    "entity" => DataType::UInt64, // Store Entity as UInt64
                    _ => DataType::Utf8,          // Fallback for unknown value types
                },
                TypeInfo::Enum(_) => DataType::Utf8, // Store enums as strings
                TypeInfo::List(l) => DataType::List(Arc::new(Field::new(
                    "item",
                    map_reflect_kind_to_arrow(
                        registry.get(l.item_type_id())
                                .unwrap()
                                .data::<ReflectType>()
                                .unwrap()
                    ),
                    true,
                ))),
                TypeInfo::Array(a) => DataType::FixedSizeList(
                    Arc::new(Field::new(
                        "item",
                        map_reflect_kind_to_arrow(
                            registry.get(a.item_type_id())
                                    .unwrap()
                                    .data::<ReflectType>()
                                    .unwrap()
                        ),
                        true,
                    )),
                    a.array_len() as i32,
                ),
                _ => DataType::Utf8, // Fallback for Tuple/Map/TupleStruct
            }
        } else {
            DataType::Utf8
        };

        let short_name = component_name.split("::").last().unwrap();
        fields.push(Field::new(short_name, data_type, true));
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

fn map_reflect_kind_to_arrow(reflect: &dyn Reflect) -> DataType {
    let type_path = reflect.get_represented_type_info()
        .map(|info| info.type_path())
        .unwrap_or_else(|| reflect.reflect_type_path());

    match type_path {
        "f32" => DataType::Float32,
        "bevy::math::Vec3" => DataType::Struct(Fields::from(vec![
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
            Field::new("z", DataType::Float32, false),
        ])),
        _ => {
            if reflect.is::<Entity>() {
                DataType::UInt64
            } else {
                DataType::Utf8
            }
        }
    }
}
