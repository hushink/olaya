//! Device selection.

use candle_core::Device;

use crate::ModelError;

/// Resolve a device preference: `auto`, `cpu`, `metal`, `cuda`, or `cuda:<index>`.
///
/// `auto` picks CUDA when this build has the `cuda` feature and a usable GPU,
/// then Metal, and falls back to CPU.
pub fn resolve(preference: &str) -> Result<Device, ModelError> {
    let value = preference.trim().to_ascii_lowercase();
    match value.as_str() {
        "auto" => Ok(preferred()),
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(ModelError::Candle)
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(ModelError::unsupported(
                    "metal support is not compiled in; enable the `metal` feature",
                ))
            }
        }
        "cuda" | "cuda:0" => cuda(0),
        other if other.starts_with("cuda:") => {
            let index = other["cuda:".len()..]
                .parse::<usize>()
                .map_err(|_| ModelError::config(format!("invalid CUDA device {other:?}")))?;
            cuda(index)
        }
        other => Err(ModelError::config(format!(
            "unknown device {other:?}; expected `cpu`, `metal`, `cuda`, or `cuda:<index>`"
        ))),
    }
}

/// The first usable accelerator compiled into this build, falling back to CPU.
pub fn preferred() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            return device;
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
    }
    Device::Cpu
}

/// The accelerators compiled into this build, for config templates and diagnostics.
pub const fn compiled_accelerators() -> &'static str {
    match (cfg!(feature = "cuda"), cfg!(feature = "metal")) {
        (true, true) => "cuda, metal",
        (true, false) => "cuda",
        (false, true) => "metal",
        (false, false) => "none",
    }
}

#[cfg(feature = "cuda")]
fn cuda(index: usize) -> Result<Device, ModelError> {
    Device::new_cuda(index).map_err(ModelError::Candle)
}

#[cfg(not(feature = "cuda"))]
fn cuda(_index: usize) -> Result<Device, ModelError> {
    Err(ModelError::unsupported(
        "CUDA support is not compiled in; enable the `cuda` feature",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_resolves_to_a_usable_device() {
        let device = resolve("auto").expect("auto resolves");
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        assert!(device.is_cpu());
    }

    #[test]
    fn compiled_accelerators_are_reported() {
        assert!(!compiled_accelerators().is_empty());
    }
}
