//! Device selection.

use candle_core::Device;

use crate::ModelError;

/// Resolve a device preference: `cpu`, `metal`, `cuda`, or `cuda:<index>`.
pub fn resolve(preference: &str) -> Result<Device, ModelError> {
    let value = preference.trim().to_ascii_lowercase();
    match value.as_str() {
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
