//! Clef (`clef-joint-schema-v1`): a Qwen3.5 backbone with a joint schema head.
//!
//! One packed row answers every question at once:
//!
//! ```text
//! <|im_start|>system ... <|im_end|>
//! <|im_start|>user
//! STATE: <state>
//!
//! SCHEMA FIELDS:
//! FIELD 1
//! ID: ... TYPE: ... INSTRUCTION: ...
//! ALLOWED OPTIONS:
//! OPTION 1: {"description":...,"option_id":...}
//! END FIELD
//! <|im_end|>
//! <|im_start|>assistant
//! <think></think>
//! JOINT SCHEMA DECISIONS:
//! ```
//!
//! The joint head reads the final hidden states at each question/option span,
//! routes evidence from the whole row, and scores every option jointly. A
//! per-question softmax gives the answer probabilities; there is no generation
//! and no calibration temperature.
//!
//! The backbone is memory-mapped and streamed one layer at a time: the 9B
//! checkpoint does not fit in fp32 RAM.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::encoders::qwen35::{Qwen35Config, Qwen35StreamingEncoder};
use crate::{Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear, linear_no_bias,
};
use indexmap::IndexMap;
use olaya_core::{
    Answer, Calibration, ChoiceAnswer, Content, DecisionError, NoulAnswer, Question, Questions,
    ScoreAnswer, State, SystemOneResponse, TokenEncoder, Usage, argmax, dumps_value_canonical,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

const LAYOUT: &str = "clef-joint-schema-v1";
const DEFAULT_MAX_LENGTH: usize = 16_384;
const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";
const HEAD_EPS: f64 = 1e-5;
const LN_100: f32 = 4.605_170_4;

#[derive(Clone, Debug, Deserialize)]
struct HeadConfig {
    hidden_size: usize,
    width: usize,
    routing_layers: usize,
    layers: usize,
    heads: usize,
    feedforward: usize,
}

impl HeadConfig {
    fn validate(&self, hidden: usize) -> Result<(), ModelError> {
        if self.hidden_size != hidden {
            return Err(ModelError::config(format!(
                "clef head expects hidden size {}, backbone has {hidden}",
                self.hidden_size
            )));
        }
        if self.width == 0
            || self.heads == 0
            || !self.width.is_multiple_of(self.heads)
            || self.feedforward == 0
        {
            return Err(ModelError::config(
                "clef joint head has an invalid width/heads/feedforward",
            ));
        }
        Ok(())
    }
}

fn content_value(content: &Content) -> Value {
    serde_json::to_value(content).expect("Content is always serializable")
}

/// `joint_schema_model.render`: strings pass through, everything else is
/// canonical JSON (`sort_keys=True`, `separators=(",", ":")`).
fn render(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => dumps_value_canonical(other),
    }
}

fn option_semantics(option_id: &str, description: &Value) -> Value {
    let mut object = Map::new();
    object.insert("option_id".to_owned(), Value::String(option_id.to_owned()));
    if !description.is_null() {
        object.insert("description".to_owned(), description.clone());
    }
    Value::Object(object)
}

struct ClefOption {
    id: String,
    description: Value,
}

fn question_options(question: &Question) -> Result<(u8, Vec<ClefOption>), DecisionError> {
    let options = match question {
        Question::Noul(noul) => {
            let described = |value: Option<&Content>, default: &str| match value {
                None => Value::String(default.to_owned()),
                Some(Content::Null) => Value::Null,
                Some(content) => content_value(content),
            };
            let criteria = noul.criteria.as_ref();
            vec![
                ClefOption {
                    id: "true".to_owned(),
                    description: described(criteria.and_then(|c| c.is_true.as_ref()), NOUL_TRUE),
                },
                ClefOption {
                    id: "false".to_owned(),
                    description: described(criteria.and_then(|c| c.is_false.as_ref()), NOUL_FALSE),
                },
            ]
        }
        Question::Choice(choice) => {
            if choice.criteria.is_empty() {
                return Err(DecisionError::invalid(
                    "choice needs at least one criterion",
                ));
            }
            let mut options = choice
                .criteria
                .iter()
                .map(|(id, description)| ClefOption {
                    id: id.clone(),
                    description: content_value(description),
                })
                .collect::<Vec<_>>();
            options.sort_by(|left, right| left.id.cmp(&right.id));
            options
        }
        Question::Score(score) => {
            if score.criteria.is_empty() {
                return Err(DecisionError::invalid("score needs at least one level"));
            }
            score
                .criteria
                .iter()
                .enumerate()
                .map(|(index, description)| ClefOption {
                    id: index.to_string(),
                    description: content_value(description),
                })
                .collect()
        }
        Question::Box(_) => {
            return Err(DecisionError::unsupported(
                "clef does not answer box questions",
            ));
        }
    };
    let qtype = match question {
        Question::Noul(_) => 0,
        Question::Choice(_) => 1,
        Question::Score(_) => 2,
        Question::Box(_) => unreachable!("box is rejected above"),
    };
    Ok((qtype, options))
}

fn instructions_value(question: &Question, question_id: &str) -> Value {
    match question.instructions() {
        None | Some(Content::Null) => Value::String(question_id.to_owned()),
        Some(content) => match content_value(content) {
            Value::String(text) if text.is_empty() => Value::String(question_id.to_owned()),
            value => value,
        },
    }
}

fn state_text(state: &Content) -> String {
    match state {
        Content::String(text) => text.clone(),
        other => dumps_value_canonical(&content_value(other)),
    }
}

#[derive(Clone, Debug)]
struct ClefQuestion {
    id: String,
    qtype: u8,
    question_span: (usize, usize),
    option_spans: Vec<(usize, usize)>,
    option_ids: Vec<String>,
}

struct ClefRow {
    ids: Vec<u32>,
    questions: Vec<ClefQuestion>,
}

fn push_tokens(
    ids: &mut Vec<u32>,
    tokenizer: &dyn TokenEncoder,
    text: &str,
) -> Result<(), DecisionError> {
    ids.extend(tokenizer.encode(text)?);
    Ok(())
}

fn build_row(
    tokenizer: &dyn TokenEncoder,
    state: &Content,
    questions: &Questions,
    max_length: usize,
) -> Result<ClefRow, DecisionError> {
    let mut schema = Vec::new();
    push_tokens(&mut schema, tokenizer, "\n\nSCHEMA FIELDS:\n")?;
    let mut encoded = Vec::with_capacity(questions.len());
    for (index, (question_id, question)) in questions.iter().enumerate() {
        let (qtype, options) = question_options(question)?;
        push_tokens(
            &mut schema,
            tokenizer,
            &format!(
                "\nFIELD {}\nID: {question_id}\nTYPE: {}\nINSTRUCTION: ",
                index + 1,
                question.type_name()
            ),
        )?;
        let question_start = schema.len();
        let instructions = instructions_value(question, question_id);
        push_tokens(&mut schema, tokenizer, &render(&instructions))?;
        let question_end = schema.len();
        push_tokens(&mut schema, tokenizer, "\nALLOWED OPTIONS:\n")?;

        let mut option_spans = Vec::with_capacity(options.len());
        let mut option_ids = Vec::with_capacity(options.len());
        for (option_index, option) in options.iter().enumerate() {
            push_tokens(
                &mut schema,
                tokenizer,
                &format!("OPTION {}: ", option_index + 1),
            )?;
            let option_start = schema.len();
            let semantics = option_semantics(&option.id, &option.description);
            push_tokens(&mut schema, tokenizer, &render(&semantics))?;
            option_spans.push((option_start, schema.len()));
            option_ids.push(option.id.clone());
            push_tokens(&mut schema, tokenizer, "\n")?;
        }
        push_tokens(&mut schema, tokenizer, "END FIELD\n")?;
        encoded.push(ClefQuestion {
            id: question_id.clone(),
            qtype,
            question_span: (question_start, question_end),
            option_spans,
            option_ids,
        });
    }

    let mut prefix = Vec::new();
    push_tokens(
        &mut prefix,
        tokenizer,
        &format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"),
    )?;
    let mut suffix = Vec::new();
    push_tokens(
        &mut suffix,
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let state_ids = tokenizer.encode(&state_text(state))?;
    let fixed = prefix.len() + schema.len() + suffix.len();
    if fixed > max_length {
        return Err(DecisionError::InputTooLong {
            tokens: fixed,
            limit: max_length,
        });
    }
    if state_ids.len() > max_length - fixed {
        return Err(DecisionError::StateTruncated);
    }

    let offset = prefix.len() + state_ids.len();
    let questions = encoded
        .into_iter()
        .map(|question| ClefQuestion {
            id: question.id,
            qtype: question.qtype,
            question_span: (
                question.question_span.0 + offset,
                question.question_span.1 + offset,
            ),
            option_spans: question
                .option_spans
                .into_iter()
                .map(|(start, end)| (start + offset, end + offset))
                .collect(),
            option_ids: question.option_ids,
        })
        .collect();

    let mut ids = Vec::with_capacity(prefix.len() + state_ids.len() + schema.len() + suffix.len());
    ids.extend(prefix);
    ids.extend(state_ids);
    ids.extend(schema);
    ids.extend(suffix);
    Ok(ClefRow { ids, questions })
}

/// One attention block with PyTorch's packed `in_proj_weight` layout.
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn load(width: usize, heads: usize, vb: VarBuilder) -> Result<Self, ModelError> {
        let weight = vb.get((3 * width, width), "in_proj_weight")?;
        let bias = vb.get(3 * width, "in_proj_bias")?;
        let split = |offset: usize| -> Result<(Tensor, Tensor), ModelError> {
            Ok((
                weight.narrow(0, offset, width)?.contiguous()?,
                bias.narrow(0, offset, width)?.contiguous()?,
            ))
        };
        let (query_weight, query_bias) = split(0)?;
        let (key_weight, key_bias) = split(width)?;
        let (value_weight, value_bias) = split(2 * width)?;
        Ok(Self {
            query: Linear::new(query_weight, Some(query_bias)),
            key: Linear::new(key_weight, Some(key_bias)),
            value: Linear::new(value_weight, Some(value_bias)),
            out: linear(width, width, vb.pp("out_proj"))?,
            heads,
            head_dim: width / heads,
        })
    }

    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, query_len, width) = query.dims3()?;
        let split = |tensor: Tensor| -> candle_core::Result<Tensor> {
            tensor
                .reshape((batch, tensor.dim(1)?, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let query = split(self.query.forward(query)?)?;
        let key = split(self.key.forward(key)?)?;
        let value = split(self.value.forward(value)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (query.matmul(&key.transpose(2, 3)?.contiguous()?)? * scale)?;
        let context = softmax_last_dim(&scores)?.matmul(&value)?;
        let context = context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, query_len, width))?;
        self.out.forward(&context)
    }
}

struct EvidenceLayer {
    query_norm: LayerNorm,
    memory_norm: LayerNorm,
    attention: MultiHeadAttention,
    feedforward_norm: LayerNorm,
    feedforward_in: Linear,
    feedforward_out: Linear,
}

impl EvidenceLayer {
    fn load(
        width: usize,
        heads: usize,
        feedforward: usize,
        vb: VarBuilder,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            query_norm: layer_norm(width, HEAD_EPS, vb.pp("query_norm"))?,
            memory_norm: layer_norm(width, HEAD_EPS, vb.pp("memory_norm"))?,
            attention: MultiHeadAttention::load(width, heads, vb.pp("attention"))?,
            feedforward_norm: layer_norm(width, HEAD_EPS, vb.pp("feedforward_norm"))?,
            feedforward_in: linear(width, feedforward, vb.pp("feedforward.0"))?,
            feedforward_out: linear(feedforward, width, vb.pp("feedforward.3"))?,
        })
    }

    fn forward(&self, queries: &Tensor, memory: &Tensor) -> candle_core::Result<Tensor> {
        let memory = self.memory_norm.forward(memory)?;
        let routed =
            self.attention
                .forward(&self.query_norm.forward(queries)?, &memory, &memory)?;
        let queries = (queries + routed)?;
        let feedforward = self
            .feedforward_in
            .forward(&self.feedforward_norm.forward(&queries)?)?
            .gelu_erf()?;
        let feedforward = self.feedforward_out.forward(&feedforward)?;
        queries + feedforward
    }
}

/// `torch.nn.TransformerDecoderLayer(norm_first=True, activation="gelu")`.
struct DecoderLayer {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl DecoderLayer {
    fn load(
        width: usize,
        heads: usize,
        feedforward: usize,
        vb: VarBuilder,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            self_attention: MultiHeadAttention::load(width, heads, vb.pp("self_attn"))?,
            cross_attention: MultiHeadAttention::load(width, heads, vb.pp("multihead_attn"))?,
            norm1: layer_norm(width, HEAD_EPS, vb.pp("norm1"))?,
            norm2: layer_norm(width, HEAD_EPS, vb.pp("norm2"))?,
            norm3: layer_norm(width, HEAD_EPS, vb.pp("norm3"))?,
            linear1: linear(width, feedforward, vb.pp("linear1"))?,
            linear2: linear(feedforward, width, vb.pp("linear2"))?,
        })
    }

    fn forward(&self, fields: &Tensor, memory: &Tensor) -> candle_core::Result<Tensor> {
        let normalized = self.norm1.forward(fields)?;
        let fields = (fields
            + self
                .self_attention
                .forward(&normalized, &normalized, &normalized)?)?;
        let computed =
            self.cross_attention
                .forward(&self.norm2.forward(&fields)?, memory, memory)?;
        let fields = (fields + computed)?;
        let feedforward = self
            .linear1
            .forward(&self.norm3.forward(&fields)?)?
            .gelu_erf()?;
        fields + self.linear2.forward(&feedforward)?
    }
}

struct JointSchemaHead {
    width: usize,
    hidden_norm: LayerNorm,
    memory_projection: Linear,
    question_projection: Linear,
    option_question_projection: Linear,
    global_projection: Linear,
    option_context_projection: Linear,
    option_lexical_projection: Linear,
    type_embedding: Embedding,
    evidence_layers: Vec<EvidenceLayer>,
    option_summary_norm: LayerNorm,
    layers: Vec<DecoderLayer>,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    residual_in: Linear,
    residual_out: Linear,
    prior_scale: f32,
    joint_scale: f32,
    residual_gate: f32,
}

fn clamp_scale(value: f32) -> f32 {
    value.min(LN_100).exp()
}

fn scalar(tensors: &HashMap<String, Tensor>, name: &str) -> Result<f32, ModelError> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| ModelError::config(format!("clef joint head has no {name}")))?;
    Ok(tensor.to_dtype(DType::F32)?.to_scalar::<f32>()?)
}

impl JointSchemaHead {
    fn load(path: &Path, config: &HeadConfig, device: &Device) -> Result<Self, ModelError> {
        let mut tensors = candle_core::safetensors::load(path, device)?;
        for tensor in tensors.values_mut() {
            if tensor.dtype() != DType::F32 {
                *tensor = tensor.to_dtype(DType::F32)?;
            }
        }
        let prior_scale = clamp_scale(scalar(&tensors, "prior_logit_scale")?);
        let joint_scale = clamp_scale(scalar(&tensors, "joint_logit_scale")?);
        let residual_gate = 1.0 / (1.0 + (-scalar(&tensors, "residual_gate")?).exp());
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        let width = config.width;
        let mut evidence_layers = Vec::with_capacity(config.routing_layers);
        for index in 0..config.routing_layers {
            evidence_layers.push(EvidenceLayer::load(
                width,
                config.heads,
                config.feedforward,
                vb.pp(format!("evidence_layers.{index}")),
            )?);
        }
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            layers.push(DecoderLayer::load(
                width,
                config.heads,
                config.feedforward,
                vb.pp(format!("layers.{index}")),
            )?);
        }
        Ok(Self {
            width,
            hidden_norm: layer_norm(config.hidden_size, HEAD_EPS, vb.pp("hidden_norm"))?,
            memory_projection: linear_no_bias(
                config.hidden_size,
                width,
                vb.pp("memory_projection"),
            )?,
            question_projection: linear_no_bias(
                config.hidden_size,
                width,
                vb.pp("question_projection"),
            )?,
            option_question_projection: linear_no_bias(
                config.hidden_size,
                width,
                vb.pp("option_question_projection"),
            )?,
            global_projection: linear_no_bias(
                config.hidden_size,
                width,
                vb.pp("global_projection"),
            )?,
            option_context_projection: linear_no_bias(
                config.hidden_size,
                width,
                vb.pp("option_context_projection"),
            )?,
            option_lexical_projection: linear_no_bias(
                config.hidden_size,
                width,
                vb.pp("option_lexical_projection"),
            )?,
            type_embedding: embedding(3, width, vb.pp("type_embedding"))?,
            evidence_layers,
            option_summary_norm: layer_norm(width, HEAD_EPS, vb.pp("option_summary_norm"))?,
            layers,
            field_norm: layer_norm(width, HEAD_EPS, vb.pp("field_norm"))?,
            option_norm: layer_norm(width, HEAD_EPS, vb.pp("option_norm"))?,
            residual_in: linear(width * 4, width, vb.pp("residual_scorer.0"))?,
            residual_out: linear(width, 1, vb.pp("residual_scorer.3"))?,
            prior_scale,
            joint_scale,
            residual_gate,
        })
    }

    fn mean_span(values: &Tensor, span: (usize, usize)) -> candle_core::Result<Tensor> {
        let (start, end) = span;
        values.narrow(0, start, end - start)?.mean(0)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        ids: &[u32],
        row: &ClefRow,
        output_weight: &Tensor,
    ) -> Result<Vec<Vec<f64>>, ModelError> {
        let device = hidden.device();
        let sequence_length = ids.len();
        if hidden.dims().len() != 3 || hidden.dim(1)? != sequence_length {
            return Err(ModelError::config(
                "clef hidden states do not match the encoded row",
            ));
        }
        let normalized = self.hidden_norm.forward(hidden)?.i(0)?;
        let sequence = normalized.narrow(0, 0, sequence_length)?;
        let memory = self.memory_projection.forward(&sequence)?.unsqueeze(0)?;
        let global = sequence.narrow(0, sequence_length - 1, 1)?.squeeze(0)?;

        let question_vectors = Tensor::stack(
            &row.questions
                .iter()
                .map(|question| Self::mean_span(&sequence, question.question_span))
                .collect::<Result<Vec<_>, _>>()?,
            0,
        )?;
        let type_ids = Tensor::from_vec(
            row.questions
                .iter()
                .map(|question| question.qtype as u32)
                .collect::<Vec<_>>(),
            row.questions.len(),
            device,
        )?;

        let mut option_contexts = Vec::with_capacity(row.questions.len());
        let mut lexical_options = Vec::with_capacity(row.questions.len());
        let mut option_counts = Vec::with_capacity(row.questions.len());
        for question in &row.questions {
            let contexts = Tensor::stack(
                &question
                    .option_spans
                    .iter()
                    .map(|span| Self::mean_span(&sequence, *span))
                    .collect::<Result<Vec<_>, _>>()?,
                0,
            )?;
            let mut lexical = Vec::with_capacity(question.option_spans.len());
            for span in &question.option_spans {
                let token_ids = ids[span.0..span.1].to_vec();
                let indices = Tensor::from_vec(token_ids, span.1 - span.0, device)?;
                lexical.push(output_weight.index_select(&indices, 0)?.mean(0)?);
            }
            option_contexts.push(contexts);
            lexical_options.push(Tensor::stack(&lexical, 0)?);
            option_counts.push(question.option_spans.len());
        }

        let mut option_queries = Vec::with_capacity(row.questions.len());
        for (index, (context, lexical)) in option_contexts
            .iter()
            .zip(lexical_options.iter())
            .enumerate()
        {
            let count = option_counts[index];
            let question = self
                .option_question_projection
                .forward(&question_vectors.narrow(0, index, 1)?)?
                .broadcast_as((count, self.width))?;
            option_queries.push(
                self.option_context_projection
                    .forward(context)?
                    .add(&self.option_lexical_projection.forward(lexical)?)?
                    .add(&question)?,
            );
        }
        let mut routed = Tensor::cat(&option_queries, 0)?.unsqueeze(0)?;
        for layer in &self.evidence_layers {
            routed = layer.forward(&routed, &memory)?;
        }
        let routed = routed.i(0)?;
        let mut split_options = Vec::with_capacity(row.questions.len());
        let mut offset = 0;
        for count in &option_counts {
            split_options.push(routed.narrow(0, offset, *count)?);
            offset += count;
        }

        let base_fields = self.question_projection.forward(&question_vectors)?;
        let mut option_summaries = Vec::with_capacity(row.questions.len());
        for (index, options) in split_options.iter().enumerate() {
            let field = base_fields.narrow(0, index, 1)?.squeeze(0)?;
            let scores = options
                .matmul(&field.unsqueeze(1)?)?
                .squeeze(1)?
                .affine(1.0 / (options.dim(1)? as f64).sqrt(), 0.0)?;
            let weights = softmax_last_dim(&scores)?;
            option_summaries.push(weights.unsqueeze(1)?.broadcast_mul(options)?.sum(0)?);
        }
        let global_projected = self
            .global_projection
            .forward(&global.unsqueeze(0)?)?
            .squeeze(0)?;
        let fields = base_fields
            .add(
                &self
                    .option_summary_norm
                    .forward(&Tensor::stack(&option_summaries, 0)?)?,
            )?
            .add(
                &global_projected
                    .unsqueeze(0)?
                    .broadcast_as(base_fields.dims())?,
            )?
            .add(&self.type_embedding.forward(&type_ids)?)?;
        let mut fields = fields.unsqueeze(0)?;
        for layer in &self.layers {
            fields = layer.forward(&fields, &memory)?;
        }
        let fields = self.field_norm.forward(&fields.i(0)?)?;

        let mut answers = Vec::with_capacity(row.questions.len());
        for (index, question) in row.questions.iter().enumerate() {
            let field = fields.narrow(0, index, 1)?.squeeze(0)?;
            let lexical = lexical_options[index].clone();
            let routed = &split_options[index];
            let anchor = normalize_last(
                &question_vectors
                    .narrow(0, index, 1)?
                    .add(&global.unsqueeze(0)?)?,
            )?
            .squeeze(0)?;
            let lexical_anchor = normalize_last(&lexical)?;
            let prior = lexical_anchor
                .matmul(&anchor.unsqueeze(1)?)?
                .squeeze(1)?
                .affine(f64::from(self.prior_scale), 0.0)?;
            let options = self.option_norm.forward(routed)?;
            let count = question.option_spans.len();
            let repeated = field.unsqueeze(0)?.broadcast_as((count, options.dim(1)?))?;
            let cosine = cosine_similarity(&repeated, &options)?;
            let features = Tensor::cat(
                &[
                    &repeated,
                    &options,
                    &repeated.mul(&options)?,
                    &repeated.sub(&options)?.abs()?,
                ],
                1,
            )?;
            let residual = self
                .residual_in
                .forward(&features)?
                .gelu_erf()
                .and_then(|values| self.residual_out.forward(&values))?
                .squeeze(1)?;
            let joint = cosine
                .affine(f64::from(self.joint_scale), 0.0)?
                .add(&residual)?;
            let logits = prior.add(&joint.affine(f64::from(self.residual_gate), 0.0)?)?;
            let probabilities = softmax_last_dim(&logits.unsqueeze(0)?)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            answers.push(probabilities.into_iter().map(f64::from).collect());
        }
        Ok(answers)
    }
}

fn normalize_last(values: &Tensor) -> candle_core::Result<Tensor> {
    let norm = values
        .sqr()?
        .sum(candle_core::D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?;
    values.broadcast_div(&norm.unsqueeze(candle_core::D::Minus1)?)
}

fn cosine_similarity(left: &Tensor, right: &Tensor) -> candle_core::Result<Tensor> {
    let dot = left.mul(right)?.sum(candle_core::D::Minus1)?;
    let left_norm = left
        .sqr()?
        .sum(candle_core::D::Minus1)?
        .sqrt()?
        .clamp(1e-8, f64::INFINITY)?;
    let right_norm = right
        .sqr()?
        .sum(candle_core::D::Minus1)?
        .sqrt()?
        .clamp(1e-8, f64::INFINITY)?;
    dot.broadcast_div(&left_norm.mul(&right_norm)?)
}

fn answer_from_probabilities(
    question: &Question,
    option_ids: &[String],
    probabilities: &[f64],
) -> Result<Answer, DecisionError> {
    if option_ids.len() != probabilities.len()
        || probabilities.is_empty()
        || probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(DecisionError::invalid(
            "clef returned invalid option probabilities",
        ));
    }
    let best = argmax(probabilities);
    Ok(match question {
        Question::Noul(_) => {
            let index = option_ids
                .iter()
                .position(|id| id == "true")
                .ok_or_else(|| DecisionError::invalid("noul has no true option"))?;
            Answer::Noul(NoulAnswer {
                noul: probabilities[index].clamp(0.0, 1.0),
            })
        }
        Question::Choice(_) => Answer::Choice(ChoiceAnswer {
            choice: option_ids[best].clone(),
            confidence: probabilities[best],
            probabilities: option_ids
                .iter()
                .cloned()
                .zip(probabilities.iter().copied())
                .collect(),
        }),
        Question::Score(score) => Answer::Score(ScoreAnswer {
            score: probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum(),
            confidence: probabilities[best],
            legend: score
                .criteria
                .iter()
                .enumerate()
                .map(|(index, level)| (index.to_string(), level.clone()))
                .collect::<IndexMap<_, _>>(),
            probabilities: probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| (index.to_string(), *probability))
                .collect(),
        }),
        Question::Box(_) => {
            return Err(DecisionError::unsupported(
                "clef does not answer box questions",
            ));
        }
    })
}

struct ClefFamily {
    tokenizer: Arc<dyn TokenEncoder>,
    encoder: Qwen35StreamingEncoder,
    head: JointSchemaHead,
    calibration: Calibration,
    device: Device,
}

impl Family for ClefFamily {
    fn id(&self) -> &'static str {
        "clef"
    }

    fn capabilities(&self) -> olaya_core::Capabilities {
        olaya_core::Capabilities::classification()
    }

    fn tokenizer(&self) -> &dyn TokenEncoder {
        self.tokenizer.as_ref()
    }

    fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    fn predict(
        &self,
        state: &State,
        questions: &Questions,
    ) -> Result<SystemOneResponse, ModelError> {
        let row = build_row(
            self.tokenizer.as_ref(),
            state.as_content(),
            questions,
            DEFAULT_MAX_LENGTH,
        )?;
        let input = Tensor::from_vec(row.ids.clone(), (1, row.ids.len()), &self.device)?;
        let hidden = self.encoder.forward(&input)?;
        let probabilities = self.head.forward(
            &hidden,
            &row.ids,
            &row,
            &self.encoder.output_embedding_weight(),
        )?;
        let mut answers = IndexMap::with_capacity(questions.len());
        for (question, probabilities) in questions.iter().zip(probabilities.iter()) {
            answers.insert(
                question.0.clone(),
                answer_from_probabilities(
                    question.1,
                    &row.questions[answers.len()].option_ids,
                    probabilities,
                )?,
            );
        }
        Ok(SystemOneResponse::new(
            self.id(),
            answers,
            Usage::new(row.ids.len() as u64, 0),
        ))
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, ModelError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))
}

/// Loads a Clef release: streamed Qwen3.5 backbone + joint schema head.
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        "clef"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        if ctx.manifest.layout != LAYOUT {
            return Err(ModelError::config(format!(
                "clef serves layout {LAYOUT}, got {:?}",
                ctx.manifest.layout
            )));
        }
        if ctx.manifest.arch != "qwen35" {
            return Err(ModelError::config(format!(
                "clef serves the qwen35 architecture, got {:?}",
                ctx.manifest.arch
            )));
        }
        let config_file = ctx
            .manifest
            .file(olaya_core::FileRole::Config)
            .ok_or_else(|| ModelError::config("clef manifest has no config.json"))?;
        let config = Qwen35Config::from_file(&ctx.dir.join(&config_file.path))?;
        let head_config_file = ctx
            .manifest
            .files
            .iter()
            .find(|file| file.path.ends_with("joint_head_config.json"))
            .ok_or_else(|| ModelError::config("clef manifest has no joint_head_config.json"))?;
        let head_config: HeadConfig = read_json(&ctx.dir.join(&head_config_file.path))?;
        head_config.validate(config.hidden_size)?;
        let head_file = ctx
            .manifest
            .files
            .iter()
            .find(|file| file.path.ends_with("joint_head.safetensors"))
            .ok_or_else(|| ModelError::config("clef manifest has no joint_head.safetensors"))?;
        let mut backbone_paths = ctx
            .manifest
            .files
            .iter()
            .filter(|file| {
                file.role == olaya_core::FileRole::Weights
                    && !file.path.ends_with("joint_head.safetensors")
            })
            .map(|file| ctx.dir.join(&file.path))
            .collect::<Vec<_>>();
        backbone_paths.sort();
        if backbone_paths.is_empty() {
            return Err(ModelError::config("clef manifest has no backbone shards"));
        }
        // Transformer layers are visited in order; a prefix cache removes that
        // much disk I/O and fp32 widening on every request. CUDA/Metal default
        // to 0 because the layer weights would compete with activations.
        let cache_mb = std::env::var("OLAYA_CLEF_LAYER_CACHE_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| if ctx.device.is_cpu() { 4096 } else { 0 });
        let encoder = Qwen35StreamingEncoder::load_with_cache(
            &config,
            &backbone_paths,
            ctx.device,
            cache_mb.saturating_mul(1024 * 1024),
        )?;
        let head = JointSchemaHead::load(&ctx.dir.join(&head_file.path), &head_config, ctx.device)?;
        Ok(Arc::new(ClefFamily {
            tokenizer: ctx.tokenizer.clone(),
            encoder,
            head,
            calibration: Calibration::default(),
            device: ctx.device.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/clef-head-tiny")
    }

    fn matrix(value: &Value, rows: usize, columns: usize, label: &str) -> Tensor {
        let values = value
            .as_array()
            .unwrap_or_else(|| panic!("{label} rows"))
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap_or_else(|| panic!("{label} row"))
                    .iter()
                    .map(|item| item.as_f64().unwrap_or_else(|| panic!("{label} value")) as f32)
            })
            .collect::<Vec<_>>();
        assert_eq!(values.len(), rows * columns, "{label} shape");
        Tensor::from_vec(values, (rows, columns), &Device::Cpu).expect(label)
    }

    #[test]
    #[ignore = "requires ~/.olaya/models/clef-flash/tokenizer.json"]
    fn encoding_matches_the_author_reference() {
        let home = std::env::var("OLAYA_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(std::env::var("USERPROFILE").expect("home")).join(".olaya")
            });
        let tokenizer = crate::tokenizer::HfTokenizer::from_dir(home.join("models/clef-flash"))
            .expect("tokenizer");
        let records: Vec<Value> = serde_json::from_slice(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/clef-encoding.json"),
            )
            .expect("encoding fixture"),
        )
        .expect("encoding json");
        for record in records {
            let id = record["id"].as_str().expect("id");
            let state = State::from_value(record["state"].clone()).expect("state");
            let mut map = IndexMap::new();
            for (question_id, question) in record["questions"].as_object().expect("questions") {
                map.insert(
                    question_id.clone(),
                    serde_json::from_value::<Question>(question.clone()).expect("question"),
                );
            }
            let questions = Questions::from(map);
            let row = build_row(
                &tokenizer,
                state.as_content(),
                &questions,
                DEFAULT_MAX_LENGTH,
            )
            .expect("row");
            let expected_ids = record["input_ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|value| value.as_u64().expect("id") as u32)
                .collect::<Vec<_>>();
            assert_eq!(row.ids, expected_ids, "{id}: input ids");
            let expected = record["encoded_questions"].as_array().expect("questions");
            assert_eq!(row.questions.len(), expected.len(), "{id}");
            for (got, want) in row.questions.iter().zip(expected) {
                assert_eq!(got.id, want["id"].as_str().expect("id"), "{id}");
                assert_eq!(
                    got.qtype,
                    want["qtype"].as_u64().expect("qtype") as u8,
                    "{id}"
                );
                assert_eq!(
                    got.question_span,
                    (
                        want["question_span"][0].as_u64().expect("span") as usize,
                        want["question_span"][1].as_u64().expect("span") as usize
                    ),
                    "{id}/{}: question span",
                    got.id
                );
                let spans = want["option_spans"].as_array().expect("option spans");
                assert_eq!(got.option_spans.len(), spans.len(), "{id}/{}", got.id);
                for (span, want) in got.option_spans.iter().zip(spans) {
                    assert_eq!(
                        *span,
                        (
                            want[0].as_u64().expect("span") as usize,
                            want[1].as_u64().expect("span") as usize
                        ),
                        "{id}/{}: option span",
                        got.id
                    );
                }
                assert_eq!(
                    got.option_ids,
                    want["option_ids"]
                        .as_array()
                        .expect("option ids")
                        .iter()
                        .map(|value| value.as_str().expect("option id").to_owned())
                        .collect::<Vec<_>>(),
                    "{id}/{}: option ids",
                    got.id
                );
            }
        }
    }

    fn span(value: &Value, label: &str) -> (usize, usize) {
        let items = value.as_array().unwrap_or_else(|| panic!("{label}"));
        (
            items[0].as_u64().expect(label) as usize,
            items[1].as_u64().expect(label) as usize,
        )
    }

    fn check_head_reference(name: &str) {
        let dir = fixture_dir();
        let config: HeadConfig =
            read_json(&dir.join("joint_head_config.json")).expect("head config");
        let head =
            JointSchemaHead::load(&dir.join("joint_head.safetensors"), &config, &Device::Cpu)
                .expect("head");
        let reference: Value = read_json(&dir.join(name)).expect("reference json");
        let length = reference["ids"].as_array().expect("ids").len();
        let hidden = matrix(&reference["hidden"], length, config.hidden_size, "hidden")
            .unsqueeze(0)
            .expect("batch");
        let output = matrix(
            &reference["output_embedding"],
            reference["output_embedding"]
                .as_array()
                .expect("output")
                .len(),
            config.hidden_size,
            "output",
        );
        let ids = reference["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|value| value.as_u64().expect("id") as u32)
            .collect::<Vec<_>>();
        let questions = match reference.get("questions").and_then(Value::as_array) {
            Some(questions) => questions.clone(),
            None => vec![reference["question"].clone()],
        };
        let row = ClefRow {
            ids: ids.clone(),
            questions: questions
                .iter()
                .map(|question| ClefQuestion {
                    id: question["id"].as_str().expect("id").to_owned(),
                    qtype: question["qtype"].as_u64().expect("qtype") as u8,
                    question_span: span(&question["question_span"], "question_span"),
                    option_spans: question["option_spans"]
                        .as_array()
                        .expect("option_spans")
                        .iter()
                        .map(|value| span(value, "option span"))
                        .collect(),
                    option_ids: question["option_ids"]
                        .as_array()
                        .expect("option_ids")
                        .iter()
                        .map(|value| value.as_str().expect("option id").to_owned())
                        .collect(),
                })
                .collect(),
        };
        let got = head.forward(&hidden, &ids, &row, &output).expect("forward");
        let logits = match reference["logits"].as_array().expect("logits")[0].as_array() {
            Some(_) => reference["logits"]
                .as_array()
                .expect("logits")
                .iter()
                .map(|question| {
                    question
                        .as_array()
                        .expect("question logits")
                        .iter()
                        .map(|value| value.as_f64().expect("logit"))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            None => vec![
                reference["logits"]
                    .as_array()
                    .expect("logits")
                    .iter()
                    .map(|value| value.as_f64().expect("logit"))
                    .collect::<Vec<_>>(),
            ],
        };
        assert_eq!(got.len(), logits.len());
        for (question, (got, logits)) in got.iter().zip(logits.iter()).enumerate() {
            let expected = olaya_core::softmax(logits.iter().copied());
            for (index, (got, want)) in got.iter().zip(expected.iter()).enumerate() {
                let diff = (got - want).abs();
                println!("question {question} option {index}: Δ = {diff:e}");
                assert!(
                    diff < 1e-5,
                    "question {question} option {index}: {got} != {want}"
                );
            }
        }
    }

    #[test]
    fn joint_head_matches_the_author_reference() {
        check_head_reference("reference.json");
    }

    #[test]
    fn joint_head_handles_multiple_question_types() {
        check_head_reference("reference_multi.json");
    }
}
