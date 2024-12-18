use bevy::prelude::*;
use bevy::reflect::TypeRegistry;
use bevy_parquet::{serialize_world, ParquetConfig, ParquetPlugin};

// Example components
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
struct Velocity {
    x: f32,
    y: f32,
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
struct Health {
    value: i32,
}

// inconsequential component that is sometimes bundled with Health
#[derive(Component)]
struct DontSerialize {
    inner: bool,
}

fn main() {
    let mut app = App::new();

    // Register types
    let type_registry = app.world().resource::<AppTypeRegistry>();
    {
        let mut registry = type_registry.write();
        registry.register::<Position>();
        registry.register::<Velocity>();
        registry.register::<Health>();
    }

    // Add plugin and configure
    app.add_plugins(ParquetPlugin)
        .insert_resource(ParquetConfig {
            output_path: "./".to_string(),
            ..Default::default()
        })
        .add_systems(Startup, spawn_things)
        .add_systems(PostStartup, serialize);

    app.run();

    // // Serialize world() state
    // if let Err(e) = serialize_world(app.world_mut()) {
    //     eprintln!("Failed to serialize world: {}", e);
    // }
}

fn serialize(world: &mut World) {
    if let Err(e) = serialize_world(world) {
        eprintln!("Failed to serialize world: {}", e);
    }
}

fn exit_after_timeout(time: Res<Time>, mut exit: EventWriter<AppExit>) {
    if time.elapsed_seconds() > 10.0 {
        exit.send(AppExit::Success);
    }
}

// Spawn some entities with different component combinations
fn spawn_things(mut commands: Commands) {
    commands.spawn((Position { x: 0.0, y: 0.0 }, Velocity { x: 1.0, y: 1.0 }));
    commands.spawn((Position { x: 5.0, y: 5.0 }, Velocity { x: -1.0, y: 0.0 }));
    commands.spawn((Health { value: 100 }, DontSerialize { inner: true }));
    commands.spawn((Health { value: 50 },));
}
