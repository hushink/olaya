//! Built-in family registration.
//!
//! The implementations live in `olaya-candle::families`; the facade only
//! registers their factories. `olaya-candle` can be used without this crate.

#[cfg(any(
    feature = "laya",
    feature = "nli",
    feature = "gliclass",
    feature = "von",
    feature = "decima",
    feature = "qwen3guard",
    feature = "kev",
    feature = "decision",
    feature = "clef"
))]
use std::sync::Arc;

use crate::engine::EngineBuilder;

/// Which built-in families to register.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinFamilies;

impl BuiltinFamilies {
    pub fn all() -> Self {
        Self
    }

    pub fn register(self, builder: EngineBuilder) -> EngineBuilder {
        let builder = builder;

        #[cfg(feature = "laya")]
        let builder = builder.family(Arc::new(olaya_candle::families::laya::Factory));
        #[cfg(feature = "nli")]
        let builder = builder.family(Arc::new(olaya_candle::families::nli::Factory));
        #[cfg(feature = "gliclass")]
        let builder = builder.family(Arc::new(olaya_candle::families::gliclass::Factory));
        #[cfg(feature = "von")]
        let builder = builder.family(Arc::new(olaya_candle::families::von::Factory));
        #[cfg(feature = "decima")]
        let builder = builder.family(Arc::new(olaya_candle::families::decima::Factory));
        #[cfg(feature = "qwen3guard")]
        let builder = builder.family(Arc::new(olaya_candle::families::qwen3guard::Factory));
        #[cfg(feature = "kev")]
        let builder = builder.family(Arc::new(olaya_candle::families::kev::Factory));
        #[cfg(feature = "decision")]
        let builder = builder.family(Arc::new(olaya_candle::families::decision::Factory));
        #[cfg(feature = "clef")]
        let builder = builder.family(Arc::new(olaya_candle::families::clef::Factory));

        builder
    }
}
