//! HTTP server, enabled by the `serve` feature.
//!
//! `/v1/*` is the strict TypeSafe contract. `/api/*` is the Ollaya-style model
//! management surface used for runtime pull, load, and unload.

mod batch;
pub mod error;
mod router;
pub mod state;

pub use batch::BatchConfig;
pub use router::{router, serve};
pub use state::ServeConfig;
