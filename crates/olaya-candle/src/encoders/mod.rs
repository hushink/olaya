//! Shared encoder architectures.
//!
//! Encoders take token ids and an attention mask and return hidden states.
//! They know nothing about System One questions, rendering, or calibration;
//! families compose them with their own heads and readouts.

#[cfg(feature = "modernbert")]
pub mod modernbert;

#[cfg(feature = "deberta")]
pub mod deberta;

#[cfg(feature = "bert")]
pub mod bert;

#[cfg(feature = "qwen3")]
pub mod qwen3;

#[cfg(feature = "qwen35")]
pub mod qwen35;
