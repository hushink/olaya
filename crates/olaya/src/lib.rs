//! The olaya facade: an in-process candle engine, a model catalog, the hub,
//! and registration of the built-in families from `olaya-candle`.
//!
//! The family implementations live in [`olaya_candle::families`]. Third-party
//! families register through [`olaya_candle::FamilyFactory`] and
//! [`EngineBuilder::family`]; they never depend on this crate.

#![forbid(unsafe_code)]

pub mod catalog;
pub mod engine;
pub mod export;
pub mod families;
pub mod hub;
pub mod registry;
#[cfg(feature = "remote")]
pub mod remote;
pub mod router;
pub mod store;

#[cfg(feature = "serve")]
pub mod serve;

pub use catalog::Catalog;
pub use engine::{Engine, EngineBuilder, ModelHandle};
pub use export::{ExportReport, export_registry};
pub use families::BuiltinFamilies;
pub use hub::{DEFAULT_HF_ENDPOINT, Hub, HubConfig};
pub use registry::{FsRegistry, StoreRegistry, default_registry_dir, load_manifests};
#[cfg(feature = "remote")]
pub use remote::{
    CATALOG_CACHE_FILE, CatalogSource, DEFAULT_REGISTRY_URL, REGISTRY_URL_ENV, RemoteRegistry,
};
pub use router::route_state;
pub use store::Store;

pub use olaya_core::{ManifestRegistry, ManifestStore, RegistryError, StoreError, StoredModel};

#[cfg(feature = "serve")]
pub use serve::ServeConfig;

pub use candle_core;
pub use olaya_candle;
pub use olaya_candle::{Backbone, BackboneOutput, Family, FamilyFactory, LoadContext, ModelError};
pub use olaya_core;
pub use olaya_core::*;
