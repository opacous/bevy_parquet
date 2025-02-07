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
    output: i32,
}

// inconsequential component that is sometimes bundled with Health
#[derive(Component)]
struct DontSerialize {
    inner: bool,
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
struct PhantomPersistTag;

fn main() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);

    // Register types
    let type_registry = app.world().resource::<AppTypeRegistry>();
    {
        let mut registry = type_registry.write();
        registry.register::<Position>();
        registry.register::<Velocity>();
        registry.register::<Health>();
        registry.register::<PhantomPersistTag>();
    }

    // Add plugin and configure
    app.add_plugins(ParquetPlugin)
        .insert_resource(ParquetConfig {
            output_path: "./".to_string(),
            ..Default::default()
        })
        .add_systems(Startup, spawn_things)
        .add_systems(PostStartup, serialize)
        .add_systems(Update, exit_after_timeout)
        ;

    // Serialize world() state
    if let Err(e) = serialize_world(app.world_mut()) {
        eprintln!("Failed to serialize world: {}", e);
    }

    app.run();

}

fn serialize(world: &mut World) {
    println!("Serializing world");
    if let Err(e) = serialize_world(world) {
        eprintln!("Failed to serialize world: {}", e);
    }
}

fn exit_after_timeout(time: Res<Time>, mut exit: EventWriter<AppExit>) {
    if time.elapsed_seconds() > 3.0 {
        exit.send(AppExit::Success);
    }
}

// Spawn some entities with different component combinations
fn spawn_things(mut commands: Commands) {
    commands.spawn((Health { output: 100 }, DontSerialize { inner: true }, PhantomPersistTag));
    commands.spawn((Health { output: 50 }, PhantomPersistTag));
}
