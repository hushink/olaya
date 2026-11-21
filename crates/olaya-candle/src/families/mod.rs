//! Built-in model families, feature-gated.
//!
//! The facade registers these factories; embedders can also use them directly
//! with `Engine::builder().family(...)` or their own loader.

#[cfg(any(
    feature = "decision",
    feature = "gliclass",
    feature = "kev",
    feature = "laya",
    feature = "von"
))]
mod batch;

#[cfg(feature = "clef")]
pub mod clef;
#[cfg(feature = "decima")]
pub mod decima;
#[cfg(feature = "decision")]
pub mod decision;
#[cfg(feature = "gliclass")]
pub mod gliclass;
#[cfg(feature = "kev")]
pub mod kev;
#[cfg(feature = "laya")]
pub mod laya;
#[cfg(feature = "nli")]
pub mod nli;
#[cfg(feature = "qwen3guard")]
pub mod qwen3guard;
#[cfg(feature = "von")]
pub mod von;
