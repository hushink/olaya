//! Read PyTorch `torch.save` checkpoints without executing pickle code.
//!
//! A `torch.save` file is a ZIP archive whose `data.pkl` member describes a
//! state dict. Tensor entries are persistent storage references and
//! `torch._utils._rebuild_tensor_v2` calls; the storage bytes themselves are the
//! `data/<key>` members. This module interprets that small, fixed pickle subset
//! and builds candle tensors straight from the stored bytes.
//!
//! It is intentionally not a general pickle implementation: unsupported
//! opcodes and objects produce a configuration error instead of being
//! executed.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use zip::ZipArchive;

use crate::ModelError;

/// Load every tensor of a `torch.save` state dict as candle tensors.
///
/// Tensor names are the state-dict keys (for example
/// `encoder.embeddings.tok_embeddings.weight`).
pub fn load_state_dict(
    path: &Path,
    device: &Device,
) -> Result<HashMap<String, Tensor>, ModelError> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))?;

    let pickle_name = find_pickle(&archive)?;
    let prefix = pickle_name
        .strip_suffix("data.pkl")
        .unwrap_or_default()
        .trim_end_matches('/');
    let mut pickle = Vec::new();
    archive
        .by_name(&pickle_name)
        .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))?
        .read_to_end(&mut pickle)?;

    let root = Unpickler::new(&pickle).run()?;
    let mut references = HashMap::new();
    collect_tensors("", &root, &mut references)?;

    let mut tensors = HashMap::with_capacity(references.len());
    for (name, reference) in references {
        let tensor = read_tensor(&mut archive, prefix, &reference, device)
            .map_err(|error| ModelError::config(format!("{name}: {error}")))?;
        tensors.insert(name, tensor);
    }
    if tensors.is_empty() {
        return Err(ModelError::config(format!(
            "{}: no tensors in the checkpoint",
            path.display()
        )));
    }
    Ok(tensors)
}

fn find_pickle<R: Read + std::io::Seek>(archive: &ZipArchive<R>) -> Result<String, ModelError> {
    let mut found = None;
    for name in archive.file_names() {
        let name = name.map_err(|error| ModelError::config(format!("zip entry: {error}")))?;
        if name == "data.pkl" || name.ends_with("/data.pkl") {
            if found.is_some() {
                return Err(ModelError::config("torch zip has several data.pkl entries"));
            }
            found = Some(name.into_owned());
        }
    }
    found.ok_or_else(|| ModelError::config("torch zip has no data.pkl entry"))
}

#[derive(Clone, Debug)]
struct StorageRef {
    key: String,
    dtype: DType,
    numel: usize,
}

#[derive(Clone, Debug)]
struct TensorRef {
    storage: StorageRef,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<isize>,
}

#[derive(Clone, Debug)]
enum Value {
    None,
    Int(i64),
    /// Scalar floats appear in optimizer metadata; the state dicts we read
    /// carry their numbers in tensors instead.
    Float(#[allow(dead_code)] f64),
    String(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage(StorageRef),
    Tensor(TensorRef),
}

fn collect_tensors(
    prefix: &str,
    value: &Value,
    out: &mut HashMap<String, TensorRef>,
) -> Result<(), ModelError> {
    match value {
        Value::Dict(items) => {
            for (key, value) in items {
                let Value::String(key) = key else {
                    continue;
                };
                let name = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                collect_tensors(&name, value, out)?;
            }
            Ok(())
        }
        Value::Tensor(tensor) => {
            if out.insert(prefix.to_owned(), tensor.clone()).is_some() {
                return Err(ModelError::config(format!(
                    "duplicate tensor {prefix:?} in the checkpoint"
                )));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn read_tensor<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    reference: &TensorRef,
    device: &Device,
) -> Result<Tensor, ModelError> {
    let element = reference.storage.dtype.size_in_bytes();
    let storage_bytes = reference
        .storage
        .numel
        .checked_mul(element)
        .ok_or_else(|| ModelError::config("storage size overflows usize"))?;
    let element_name = if prefix.is_empty() {
        format!("data/{}", reference.storage.key)
    } else {
        format!("{prefix}/data/{}", reference.storage.key)
    };
    let mut entry = archive.by_name(&element_name).map_err(|error| {
        ModelError::config(format!("missing storage entry {element_name:?}: {error}"))
    })?;
    let mut bytes = Vec::with_capacity(storage_bytes);
    entry.read_to_end(&mut bytes)?;
    if bytes.len() != storage_bytes {
        return Err(ModelError::config(format!(
            "storage {element_name:?} has {} bytes, expected {storage_bytes}",
            bytes.len()
        )));
    }

    let count = reference
        .shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| ModelError::config("tensor shape overflows usize"))?;
    if reference.shape.len() != reference.stride.len() {
        return Err(ModelError::config("tensor shape/stride rank mismatch"));
    }

    if contiguous(&reference.shape, &reference.stride) {
        let start = reference
            .offset
            .checked_mul(element)
            .ok_or_else(|| ModelError::config("tensor offset overflows usize"))?;
        let length = count
            .checked_mul(element)
            .ok_or_else(|| ModelError::config("tensor size overflows usize"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| ModelError::config("tensor range overflows usize"))?;
        if end > bytes.len() {
            return Err(ModelError::config("tensor exceeds its storage"));
        }
        Ok(Tensor::from_raw_buffer(
            &bytes[start..end],
            reference.storage.dtype,
            &reference.shape,
            device,
        )?)
    } else {
        let mut contiguous = vec![0u8; count * element];
        let mut index = vec![0usize; reference.shape.len()];
        for linear in 0..count {
            let mut at = reference.offset as isize;
            for (dim, position) in index.iter().enumerate() {
                at += *position as isize * reference.stride[dim];
            }
            if at < 0 || at as usize >= reference.storage.numel {
                return Err(ModelError::config("strided tensor exceeds its storage"));
            }
            let source = at as usize * element;
            let target = linear * element;
            contiguous[target..target + element].copy_from_slice(&bytes[source..source + element]);
            for dim in (0..index.len()).rev() {
                index[dim] += 1;
                if index[dim] < reference.shape[dim] {
                    break;
                }
                index[dim] = 0;
            }
        }
        Ok(Tensor::from_raw_buffer(
            &contiguous,
            reference.storage.dtype,
            &reference.shape,
            device,
        )?)
    }
}

fn contiguous(shape: &[usize], stride: &[isize]) -> bool {
    let mut expected = 1isize;
    for (&dim, &stride) in shape.iter().zip(stride).rev() {
        if dim > 1 && stride != expected {
            return false;
        }
        expected *= dim as isize;
    }
    true
}

struct Unpickler<'a> {
    data: &'a [u8],
    position: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<usize, Value>,
}

impl<'a> Unpickler<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn run(mut self) -> Result<Value, ModelError> {
        loop {
            let opcode = self.read_u8()?;
            match opcode {
                0x80 => {
                    // PROTO
                    self.read_u8()?;
                }
                0x95 => {
                    // FRAME
                    self.read_u64()?;
                }
                b'.' => {
                    return self
                        .stack
                        .pop()
                        .ok_or_else(|| self.error("STOP with an empty stack"));
                }
                b'(' => self.marks.push(self.stack.len()),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b't' => {
                    let mark = self.pop_mark()?;
                    let items = self.stack.split_off(mark);
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => self.tuple(1)?,
                0x86 => self.tuple(2)?,
                0x87 => self.tuple(3)?,
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let Some(Value::Dict(mut items)) = self.stack.pop() else {
                        return Err(self.error("SETITEM without a dict"));
                    };
                    items.push((key, value));
                    self.stack.push(Value::Dict(items));
                }
                b'u' => {
                    let mark = self.pop_mark()?;
                    let items = self.stack.split_off(mark);
                    let Some(Value::Dict(mut dict)) = self.stack.pop() else {
                        return Err(self.error("SETITEMS without a dict"));
                    };
                    let mut items = items.into_iter();
                    while let (Some(key), Some(value)) = (items.next(), items.next()) {
                        dict.push((key, value));
                    }
                    self.stack.push(Value::Dict(dict));
                }
                b']' => self.stack.push(Value::List(Vec::new())),
                b'a' => {
                    let value = self.pop()?;
                    let Some(Value::List(mut items)) = self.stack.pop() else {
                        return Err(self.error("APPEND without a list"));
                    };
                    items.push(value);
                    self.stack.push(Value::List(items));
                }
                b'e' => {
                    let mark = self.pop_mark()?;
                    let items = self.stack.split_off(mark);
                    let Some(Value::List(mut list)) = self.stack.pop() else {
                        return Err(self.error("APPENDS without a list"));
                    };
                    list.extend(items);
                    self.stack.push(Value::List(list));
                }
                b'q' => {
                    let index = self.read_u8()? as usize;
                    self.memo(index)?;
                }
                b'r' => {
                    let index = self.read_u32()? as usize;
                    self.memo(index)?;
                }
                b'h' => {
                    let index = self.read_u8()? as usize;
                    self.get(index)?;
                }
                b'j' => {
                    let index = self.read_u32()? as usize;
                    self.get(index)?;
                }
                b'X' => {
                    let length = self.read_u32()? as usize;
                    let text = self.read_string(length)?;
                    self.stack.push(Value::String(text));
                }
                0x8c => {
                    let length = self.read_u8()? as usize;
                    let text = self.read_string(length)?;
                    self.stack.push(Value::String(text));
                }
                0x8d => {
                    let length = self.read_u64()? as usize;
                    let text = self.read_string(length)?;
                    self.stack.push(Value::String(text));
                }
                b'G' => {
                    // BINFLOAT stores a big-endian IEEE-754 double.
                    let bytes: [u8; 8] = self.read_bytes(8)?.try_into().expect("length");
                    self.stack.push(Value::Float(f64::from_be_bytes(bytes)));
                }
                b'J' => {
                    let value = self.read_i32()? as i64;
                    self.stack.push(Value::Int(value));
                }
                b'K' => {
                    let value = self.read_u8()? as i64;
                    self.stack.push(Value::Int(value));
                }
                b'M' => {
                    let value = self.read_u16()? as i64;
                    self.stack.push(Value::Int(value));
                }
                0x8a => {
                    let length = self.read_u8()? as usize;
                    let value = self.read_signed_le(length)?;
                    self.stack.push(Value::Int(value));
                }
                0x8b => {
                    let length = self.read_u32()? as usize;
                    let value = self.read_signed_le(length)?;
                    self.stack.push(Value::Int(value));
                }
                0x88 => self.stack.push(Value::Int(1)),
                0x89 => self.stack.push(Value::Int(0)),
                b'N' => self.stack.push(Value::None),
                b'c' => {
                    let module = self.read_c_string()?;
                    let name = self.read_c_string()?;
                    self.stack.push(Value::Global(module, name));
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (Value::String(name), Value::String(module)) = (name, module) else {
                        return Err(self.error("STACK_GLOBAL without two strings"));
                    };
                    self.stack.push(Value::Global(module, name));
                }
                b'Q' => {
                    let id = self.pop()?;
                    self.stack.push(self.persistent_id(id)?);
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let value = self.reduce(callable, args)?;
                    self.stack.push(value);
                }
                b'b' => {
                    // BUILD: the state is irrelevant for the dicts we read.
                    let _state = self.pop()?;
                    let value = self.pop()?;
                    self.stack.push(value);
                }
                other => {
                    return Err(self.error(&format!("unsupported pickle opcode 0x{other:02x}")));
                }
            }
        }
    }

    fn tuple(&mut self, count: usize) -> Result<(), ModelError> {
        if self.stack.len() < count {
            return Err(self.error("not enough values for a tuple"));
        }
        let items = self.stack.split_off(self.stack.len() - count);
        self.stack.push(Value::Tuple(items));
        Ok(())
    }

    fn persistent_id(&self, value: Value) -> Result<Value, ModelError> {
        let Value::Tuple(items) = value else {
            return Err(self.error("persistent id is not a tuple"));
        };
        if items.len() < 5 {
            return Err(self.error("persistent id has fewer than five fields"));
        }
        let Value::Global(_, storage_name) = &items[1] else {
            return Err(self.error("persistent id has no storage type"));
        };
        let Value::String(key) = &items[2] else {
            return Err(self.error("persistent id has no storage key"));
        };
        let Value::Int(numel) = &items[4] else {
            return Err(self.error("persistent id has no element count"));
        };
        let numel = usize::try_from(*numel)
            .map_err(|_| self.error("persistent id has a negative element count"))?;
        Ok(Value::Storage(StorageRef {
            key: key.clone(),
            dtype: storage_dtype(storage_name)
                .ok_or_else(|| self.error(&format!("unsupported storage {storage_name:?}")))?,
            numel,
        }))
    }

    fn reduce(&self, callable: Value, args: Value) -> Result<Value, ModelError> {
        let Value::Global(module, name) = callable else {
            return Err(self.error("REDUCE without a global callable"));
        };
        let args = match args {
            Value::Tuple(args) => args,
            other => vec![other],
        };
        match (module.as_str(), name.as_str()) {
            ("collections", "OrderedDict") => Ok(Value::Dict(Vec::new())),
            ("torch._utils", "_rebuild_tensor_v2") => self.rebuild_tensor(&args),
            ("torch._utils", "_rebuild_parameter") => args
                .into_iter()
                .next()
                .ok_or_else(|| self.error("_rebuild_parameter without a tensor")),
            ("torch._utils", "_rebuild_parameter_with_state") => args
                .into_iter()
                .next()
                .ok_or_else(|| self.error("_rebuild_parameter_with_state without a tensor")),
            _ => Err(self.error(&format!("unsupported pickle global {module}.{name}"))),
        }
    }

    fn rebuild_tensor(&self, args: &[Value]) -> Result<Value, ModelError> {
        if args.len() < 4 {
            return Err(self.error("_rebuild_tensor_v2 has fewer than four arguments"));
        }
        let Value::Storage(storage) = &args[0] else {
            return Err(self.error("_rebuild_tensor_v2 storage is not a storage"));
        };
        let Value::Int(offset) = &args[1] else {
            return Err(self.error("_rebuild_tensor_v2 offset is not an int"));
        };
        let offset = usize::try_from(*offset)
            .map_err(|_| self.error("_rebuild_tensor_v2 has a negative offset"))?;
        let Value::Tuple(shape) = &args[2] else {
            return Err(self.error("_rebuild_tensor_v2 shape is not a tuple"));
        };
        let Value::Tuple(stride) = &args[3] else {
            return Err(self.error("_rebuild_tensor_v2 stride is not a tuple"));
        };
        let shape = shape
            .iter()
            .map(|value| self.usize(value))
            .collect::<Result<Vec<_>, _>>()?;
        let stride = stride
            .iter()
            .map(|value| self.isize(value))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value::Tensor(TensorRef {
            storage: storage.clone(),
            offset,
            shape,
            stride,
        }))
    }

    fn usize(&self, value: &Value) -> Result<usize, ModelError> {
        match value {
            Value::Int(value) => {
                usize::try_from(*value).map_err(|_| self.error("negative tensor dimension"))
            }
            _ => Err(self.error("tensor dimension is not an int")),
        }
    }

    fn isize(&self, value: &Value) -> Result<isize, ModelError> {
        match value {
            Value::Int(value) => {
                isize::try_from(*value).map_err(|_| self.error("tensor stride is out of range"))
            }
            _ => Err(self.error("tensor stride is not an int")),
        }
    }

    fn pop(&mut self) -> Result<Value, ModelError> {
        self.stack
            .pop()
            .ok_or_else(|| self.error("pickle stack underflow"))
    }

    fn pop_mark(&mut self) -> Result<usize, ModelError> {
        self.marks
            .pop()
            .ok_or_else(|| self.error("pickle mark underflow"))
    }

    fn memo(&mut self, index: usize) -> Result<(), ModelError> {
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| self.error("memo without a value"))?;
        self.memo.insert(index, value);
        Ok(())
    }

    fn get(&mut self, index: usize) -> Result<(), ModelError> {
        let value = self
            .memo
            .get(&index)
            .cloned()
            .ok_or_else(|| self.error(&format!("missing pickle memo {index}")))?;
        self.stack.push(value);
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, ModelError> {
        let bytes: [u8; 2] = self.read_bytes(2)?.try_into().expect("length");
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32, ModelError> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().expect("length");
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, ModelError> {
        let bytes: [u8; 8] = self.read_bytes(8)?.try_into().expect("length");
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32, ModelError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_signed_le(&mut self, length: usize) -> Result<i64, ModelError> {
        if length > 8 {
            return Err(self.error("long pickle integer is wider than eight bytes"));
        }
        let bytes = self.read_bytes(length)?;
        if bytes.is_empty() {
            return Ok(0);
        }
        let mut value = 0u64;
        for (index, byte) in bytes.iter().enumerate() {
            value |= u64::from(*byte) << (8 * index);
        }
        let shift = 64 - 8 * length;
        Ok(((value << shift) as i64) >> shift)
    }

    fn read_bytes(&mut self, length: usize) -> Result<&'a [u8], ModelError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| self.error("pickle offset overflows usize"))?;
        if end > self.data.len() {
            return Err(self.error("truncated pickle"));
        }
        let bytes = &self.data[self.position..end];
        self.position = end;
        Ok(bytes)
    }

    fn read_string(&mut self, length: usize) -> Result<String, ModelError> {
        let bytes = self.read_bytes(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| self.error("pickle string is not UTF-8"))
    }

    fn read_c_string(&mut self) -> Result<String, ModelError> {
        let start = self.position;
        while self.position < self.data.len() && self.data[self.position] != b'\n' {
            self.position += 1;
        }
        if self.position >= self.data.len() {
            return Err(self.error("unterminated pickle global"));
        }
        let text = self.data[start..self.position].to_vec();
        self.position += 1;
        String::from_utf8(text).map_err(|_| self.error("pickle global is not UTF-8"))
    }

    fn error(&self, message: &str) -> ModelError {
        ModelError::config(format!("torch pickle at byte {}: {message}", self.position))
    }
}

fn storage_dtype(name: &str) -> Option<DType> {
    let name = name.rsplit('.').next().unwrap_or(name);
    match name {
        "FloatStorage" => Some(DType::F32),
        "HalfStorage" => Some(DType::F16),
        "BFloat16Storage" => Some(DType::BF16),
        "LongStorage" => Some(DType::I64),
        "IntStorage" => Some(DType::I32),
        "ShortStorage" => Some(DType::I16),
        "ByteStorage" | "CharStorage" | "BoolStorage" => Some(DType::U8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_torch_storage_types() {
        assert_eq!(storage_dtype("FloatStorage"), Some(DType::F32));
        assert_eq!(storage_dtype("torch.BFloat16Storage"), Some(DType::BF16));
        assert_eq!(storage_dtype("LongStorage"), Some(DType::I64));
        assert_eq!(storage_dtype("MysteryStorage"), None);
    }

    #[test]
    fn contiguous_layouts_match_row_major() {
        assert!(contiguous(&[2, 3], &[3, 1]));
        assert!(!contiguous(&[2, 3], &[1, 2]));
        assert!(contiguous(&[1, 3], &[7, 1]));
        assert!(contiguous(&[], &[]));
    }

    #[test]
    fn reads_small_pickle_values() {
        // PROTO 2, BINUNICODE "ok", STOP.
        let mut pickle = vec![0x80, 2, b'X', 2, 0, 0, 0];
        pickle.extend_from_slice(b"ok");
        pickle.push(b'.');
        let value = Unpickler::new(&pickle).run().expect("pickle");
        assert!(matches!(value, Value::String(text) if text == "ok"));
    }

    #[test]
    #[ignore = "requires ~/.olaya/models/von/option_marker.pt"]
    fn loads_the_von_state_dict() {
        let path = std::env::var_os("OLAYA_VON_WEIGHTS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let home = std::env::var_os("USERPROFILE")
                    .or_else(|| std::env::var_os("HOME"))
                    .expect("home directory");
                std::path::PathBuf::from(home).join(".olaya/models/von/option_marker.pt")
            });
        let tensors = load_state_dict(&path, &Device::Cpu).expect("von state dict");
        assert_eq!(tensors.len(), 178);
        assert_eq!(
            tensors["encoder.embeddings.tok_embeddings.weight"].dims(),
            &[50368, 1024]
        );
        assert_eq!(tensors["scorer.out_proj.weight"].dims(), &[1, 512]);
        assert_eq!(tensors["scorer.out_proj.bias"].dims(), &[1]);
    }
}
