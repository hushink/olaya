#![cfg(feature = "qwen35")]

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use olaya_candle::encoders::qwen35::{Qwen35Config, Qwen35Encoder, Qwen35StreamingEncoder};
use serde_json::Value;

fn model_dir() -> PathBuf {
    std::env::var("QWEN35_BASE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35-tiny")
        })
}

fn max_difference(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn qwen35_matches_transformers_fp32() {
    let dir = model_dir();
    let config = Qwen35Config::from_file(&dir.join("config.json")).expect("config");
    let device = Device::Cpu;
    let tensors =
        candle_core::safetensors::load(dir.join("model.safetensors"), &device).expect("weights");
    let encoder = Qwen35Encoder::load(&config, tensors, &device).expect("encoder");
    let reference: Vec<Value> =
        serde_json::from_slice(&std::fs::read(dir.join("reference.json")).expect("reference"))
            .expect("reference json");

    for (index, record) in reference.iter().enumerate() {
        let ids: Vec<u32> = record["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|value| value.as_u64().expect("id") as u32)
            .collect();
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &device).expect("input");
        let hidden = encoder.forward(&input).expect("forward");
        let last = hidden
            .i((0, ids.len() - 1))
            .expect("last hidden")
            .to_dtype(DType::F32)
            .expect("f32")
            .to_vec1::<f32>()
            .expect("values");
        let expected: Vec<f32> = record["last_hidden"]
            .as_array()
            .expect("last_hidden")
            .iter()
            .map(|value| value.as_f64().expect("value") as f32)
            .collect();
        let hidden_diff = max_difference(&last, &expected);

        let candidates: Vec<u32> = record["cand_ids"]
            .as_array()
            .expect("cand_ids")
            .iter()
            .map(|value| value.as_u64().expect("id") as u32)
            .collect();
        let last = Tensor::from_vec(last, (1, expected.len()), &device).expect("hidden");
        let logits = encoder
            .candidate_logits(&last, &candidates)
            .expect("candidate logits")
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("logits");
        let expected_logits: Vec<f32> = record["cand_logits"]
            .as_array()
            .expect("cand_logits")
            .iter()
            .map(|value| value.as_f64().expect("logit") as f32)
            .collect();
        let logit_diff = max_difference(&logits, &expected_logits);
        println!(
            "record {index} ({} tokens): max |Δhidden| = {hidden_diff:e}, max |Δlogit| = {logit_diff:e}",
            ids.len()
        );
        assert!(
            hidden_diff < 1e-3,
            "record {index}: hidden diff {hidden_diff}"
        );
        assert!(logit_diff < 1e-2, "record {index}: logit diff {logit_diff}");
    }
}

#[test]
fn streaming_encoder_matches_the_loaded_encoder() {
    let dir = model_dir();
    let config = Qwen35Config::from_file(&dir.join("config.json")).expect("config");
    let device = Device::Cpu;
    let tensors =
        candle_core::safetensors::load(dir.join("model.safetensors"), &device).expect("weights");
    let loaded = Qwen35Encoder::load(&config, tensors, &device).expect("encoder");
    let streamed = Qwen35StreamingEncoder::load(&config, &[dir.join("model.safetensors")], &device)
        .expect("streaming encoder");
    let reference: Vec<Value> =
        serde_json::from_slice(&std::fs::read(dir.join("reference.json")).expect("reference"))
            .expect("reference json");
    for (index, record) in reference.iter().enumerate() {
        let ids: Vec<u32> = record["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|value| value.as_u64().expect("id") as u32)
            .collect();
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &device).expect("input");
        let expected = loaded.forward(&input).expect("loaded forward");
        let got = streamed.forward(&input).expect("streamed forward");
        let diff = (expected - got)
            .expect("diff")
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");
        println!("record {index}: streaming max |Δhidden| = {diff:e}");
        assert!(diff < 1e-6, "record {index}: streaming diff {diff}");
    }
}
