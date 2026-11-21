//! Answers, responses, token usage, and the model list.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::Value;

use crate::content::Content;

/// An answer whose type matches the corresponding question.
///
/// Unknown answer kinds are preserved in [`Answer::Unknown`] so that a newer
/// server can add answer types without breaking older clients.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Box(BoxAnswer),
    Unknown(Value),
}

/// The kind of an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnswerType {
    Noul,
    Choice,
    Score,
    Box,
    Unknown,
}

impl AnswerType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Box => "box",
            Self::Unknown => "unknown",
        }
    }
}

/// The probability of a yes answer or a true statement, from 0 to 1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoulAnswer {
    pub noul: f64,
}

/// A selected choice, its confidence, and the probability of every choice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub confidence: f64,
    pub probabilities: IndexMap<String, f64>,
}

/// An expected score with its rubric and per-level probabilities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreAnswer {
    /// Probability-weighted average of the rubric levels.
    pub score: f64,
    pub confidence: f64,
    /// The rubric descriptions keyed by integer score level.
    pub legend: BTreeMap<u8, Content>,
    /// Probability of each score level.
    pub probabilities: BTreeMap<u8, f64>,
}

/// A located box with normalized edge coordinates and optional confidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxAnswer {
    /// The box's edges as fractions of the image height and width, from 0 to 1.
    #[serde(rename = "box")]
    pub coordinates: BoxCoordinates,
    /// `None` when the model does not provide confidence.
    pub confidence: Option<f64>,
}

/// Normalized box edges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxCoordinates {
    /// Top edge as a fraction of image height from the top (0-1).
    pub top: f64,
    /// Bottom edge as a fraction of image height from the top (0-1).
    pub bottom: f64,
    /// Left edge as a fraction of image width from the left (0-1).
    pub left: f64,
    /// Right edge as a fraction of image width from the left (0-1).
    pub right: f64,
}

/// Token counts for a request, when reported by the API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
}

/// A successful System One response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    /// The model that answered the questions; may differ from the requested
    /// alias.
    pub model: String,
    #[serde(default)]
    pub answers: IndexMap<String, Answer>,
    #[serde(default)]
    pub usage: Usage,
    /// The `x-typesafe-request-id` response header, when present.
    #[serde(skip)]
    pub request_id: Option<String>,
}

/// The models and aliases available to the authenticated account.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelMetadata>,
    /// The `x-typesafe-request-id` response header, when present.
    #[serde(skip)]
    pub request_id: Option<String>,
}

/// A model name or alias accepted by the `model` request field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub description: String,
    /// Release date formatted as `YYYY-MM-DD`.
    pub release_date: String,
}

impl Answer {
    pub fn answer_type(&self) -> AnswerType {
        match self {
            Self::Noul(_) => AnswerType::Noul,
            Self::Choice(_) => AnswerType::Choice,
            Self::Score(_) => AnswerType::Score,
            Self::Box(_) => AnswerType::Box,
            Self::Unknown(_) => AnswerType::Unknown,
        }
    }

    pub fn as_noul(&self) -> Option<&NoulAnswer> {
        match self {
            Self::Noul(answer) => Some(answer),
            _ => None,
        }
    }

    pub fn as_choice(&self) -> Option<&ChoiceAnswer> {
        match self {
            Self::Choice(answer) => Some(answer),
            _ => None,
        }
    }

    pub fn as_score(&self) -> Option<&ScoreAnswer> {
        match self {
            Self::Score(answer) => Some(answer),
            _ => None,
        }
    }

    pub fn as_box(&self) -> Option<&BoxAnswer> {
        match self {
            Self::Box(answer) => Some(answer),
            _ => None,
        }
    }

    pub fn as_unknown(&self) -> Option<&Value> {
        match self {
            Self::Unknown(value) => Some(value),
            _ => None,
        }
    }
}

impl ChoiceAnswer {
    /// The probability assigned to `choice`, if present.
    pub fn probability(&self, choice: &str) -> Option<f64> {
        self.probabilities.get(choice).copied()
    }
}

impl ScoreAnswer {
    /// The rubric description for a score level, if present.
    pub fn legend_for(&self, level: u8) -> Option<&Content> {
        self.legend.get(&level)
    }

    /// The probability of a score level, if present.
    pub fn probability(&self, level: u8) -> Option<f64> {
        self.probabilities.get(&level).copied()
    }
}

impl Usage {
    /// Input plus output tokens, when both are reported.
    pub fn total_tokens(&self) -> Option<u64> {
        Some(self.input_tokens? + self.output_tokens?)
    }
}

impl SystemOneResponse {
    /// The answer named `name`, if present.
    pub fn answer(&self, name: &str) -> Option<&Answer> {
        self.answers.get(name)
    }

    /// The yes/no answer named `name`, if it has that type.
    pub fn noul(&self, name: &str) -> Option<&NoulAnswer> {
        self.answer(name)?.as_noul()
    }

    /// The choice answer named `name`, if it has that type.
    pub fn choice(&self, name: &str) -> Option<&ChoiceAnswer> {
        self.answer(name)?.as_choice()
    }

    /// The score answer named `name`, if it has that type.
    pub fn score(&self, name: &str) -> Option<&ScoreAnswer> {
        self.answer(name)?.as_score()
    }

    /// The box answer named `name`, if it has that type.
    pub fn box_answer(&self, name: &str) -> Option<&BoxAnswer> {
        self.answer(name)?.as_box()
    }

    /// All yes/no answers keyed by question name.
    pub fn nouls(&self) -> impl Iterator<Item = (&str, &NoulAnswer)> + '_ {
        self.answers.iter().filter_map(|(name, answer)| {
            if let Answer::Noul(answer) = answer {
                Some((name.as_str(), answer))
            } else {
                None
            }
        })
    }

    /// All choice answers keyed by question name.
    pub fn choices(&self) -> impl Iterator<Item = (&str, &ChoiceAnswer)> + '_ {
        self.answers.iter().filter_map(|(name, answer)| {
            if let Answer::Choice(answer) = answer {
                Some((name.as_str(), answer))
            } else {
                None
            }
        })
    }

    /// All score answers keyed by question name.
    pub fn scores(&self) -> impl Iterator<Item = (&str, &ScoreAnswer)> + '_ {
        self.answers.iter().filter_map(|(name, answer)| {
            if let Answer::Score(answer) = answer {
                Some((name.as_str(), answer))
            } else {
                None
            }
        })
    }

    /// All box answers keyed by question name.
    pub fn boxes(&self) -> impl Iterator<Item = (&str, &BoxAnswer)> + '_ {
        self.answers.iter().filter_map(|(name, answer)| {
            if let Answer::Box(answer) = answer {
                Some((name.as_str(), answer))
            } else {
                None
            }
        })
    }

    /// Answers whose kind this SDK does not recognize.
    pub fn unknown_answers(&self) -> impl Iterator<Item = (&str, &Value)> + '_ {
        self.answers.iter().filter_map(|(name, answer)| {
            if let Answer::Unknown(value) = answer {
                Some((name.as_str(), value))
            } else {
                None
            }
        })
    }
}

impl ListModelsResponse {
    /// The model named `name` or exposed under that alias.
    pub fn get(&self, name: &str) -> Option<&ModelMetadata> {
        self.models.iter().find(|model| model.name == name)
    }

    /// The available model names.
    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.models.iter().map(|model| model.name.as_str())
    }
}

impl Serialize for Answer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "lowercase")]
        enum Known<'a> {
            Noul(&'a NoulAnswer),
            Choice(&'a ChoiceAnswer),
            Score(&'a ScoreAnswer),
            Box(&'a BoxAnswer),
        }

        #[derive(Serialize)]
        #[serde(untagged)]
        enum Any<'a> {
            Known(Known<'a>),
            Unknown(&'a Value),
        }

        let value = match self {
            Self::Noul(answer) => Any::Known(Known::Noul(answer)),
            Self::Choice(answer) => Any::Known(Known::Choice(answer)),
            Self::Score(answer) => Any::Known(Known::Score(answer)),
            Self::Box(answer) => Any::Known(Known::Box(answer)),
            Self::Unknown(value) => Any::Unknown(value),
        };
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Answer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| de::Error::missing_field("type"))?
            .to_owned();
        match kind.as_str() {
            "noul" => serde_json::from_value(value)
                .map(Self::Noul)
                .map_err(de::Error::custom),
            "choice" => serde_json::from_value(value)
                .map(Self::Choice)
                .map_err(de::Error::custom),
            "score" => serde_json::from_value(value)
                .map(Self::Score)
                .map_err(de::Error::custom),
            "box" => serde_json::from_value(value)
                .map(Self::Box)
                .map_err(de::Error::custom),
            _ => {
                tracing::warn!(answer_type = %kind, "preserving an unknown answer type");
                Ok(Self::Unknown(value))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: Value) -> SystemOneResponse {
        serde_json::from_value(value).expect("response")
    }

    #[test]
    fn parses_the_official_noul_response() {
        let response = parse(json!({
            "model": "jev-latest",
            "answers": {"billing": {"type": "noul", "noul": 0.98}},
            "usage": {"input_tokens": 120, "output_tokens": 12}
        }));
        assert_eq!(response.model, "jev-latest");
        assert_eq!(response.noul("billing").expect("noul").noul, 0.98);
        assert!(response.choices().next().is_none());
        assert_eq!(response.usage.input_tokens, Some(120));
        assert_eq!(response.usage.total_tokens(), Some(132));
    }

    #[test]
    fn parses_the_official_choice_response() {
        let response = parse(json!({
            "model": "jev-latest",
            "answers": {
                "tone": {
                    "type": "choice",
                    "choice": "calm",
                    "confidence": 0.7,
                    "probabilities": {"angry": 0.15, "calm": 0.8, "excited": 0.05}
                }
            },
            "usage": {"input_tokens": 120, "output_tokens": 12}
        }));
        let answer = response.choice("tone").expect("choice");
        assert_eq!(answer.choice, "calm");
        assert_eq!(answer.confidence, 0.7);
        assert_eq!(answer.probability("calm"), Some(0.8));
        assert_eq!(answer.probability("missing"), None);
    }

    #[test]
    fn parses_the_official_score_response_with_integer_keys() {
        let response = parse(json!({
            "model": "jev-latest",
            "answers": {
                "urgency": {
                    "type": "score",
                    "score": 1.7,
                    "confidence": 0.55,
                    "legend": {"0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today"},
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}
                }
            },
            "usage": {"input_tokens": 120, "output_tokens": 12}
        }));
        let answer = response.score("urgency").expect("score");
        assert_eq!(answer.score, 1.7);
        assert_eq!(answer.probability(2), Some(0.8));
        assert_eq!(
            answer.legend_for(1),
            Some(&Content::from("Needs attention this week"))
        );
        assert_eq!(
            answer.probabilities.keys().copied().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn parses_the_official_box_response() {
        let response = parse(json!({
            "model": "jev-preview",
            "answers": {
                "red_rectangle": {
                    "type": "box",
                    "box": {"top": 0.2, "bottom": 0.66, "left": 0.19, "right": 0.68},
                    "confidence": null
                }
            },
            "usage": {"input_tokens": 285, "output_tokens": 20}
        }));
        let answer = response.box_answer("red_rectangle").expect("box");
        assert_eq!(answer.coordinates.top, 0.2);
        assert_eq!(answer.coordinates.right, 0.68);
        assert_eq!(answer.confidence, None);
    }

    #[test]
    fn unknown_answer_kinds_are_preserved_and_round_trip() {
        let raw = json!({"type": "ranking", "items": ["a", "b"]});
        let response = parse(json!({
            "model": "jev-latest",
            "answers": {"rank": raw},
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
        assert_eq!(
            response.answer("rank").expect("answer").answer_type(),
            AnswerType::Unknown
        );
        let unknown = response.unknown_answers().next().expect("unknown");
        assert_eq!(unknown.0, "rank");
        assert_eq!(
            serde_json::to_value(response.answer("rank").expect("answer")).expect("json"),
            raw
        );
    }

    #[test]
    fn known_answers_round_trip_with_the_type_tag() {
        let response = parse(json!({
            "model": "jev-latest",
            "answers": {
                "billing": {"type": "noul", "noul": 0.98},
                "tone": {
                    "type": "choice",
                    "choice": "calm",
                    "confidence": 0.7,
                    "probabilities": {"calm": 1.0}
                }
            },
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }));
        let value = serde_json::to_value(&response).expect("json");
        assert_eq!(value["answers"]["billing"]["type"], "noul");
        assert_eq!(value["answers"]["tone"]["type"], "choice");
        assert!(value.get("request_id").is_none());
        assert_eq!(parse(value), response);
    }

    #[test]
    fn score_and_box_answers_round_trip() {
        let response = parse(json!({
            "model": "jev-latest",
            "answers": {
                "urgency": {
                    "type": "score",
                    "score": 1.7,
                    "confidence": 0.55,
                    "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}
                },
                "red_rectangle": {
                    "type": "box",
                    "box": {"top": 0.2, "bottom": 0.66, "left": 0.19, "right": 0.68},
                    "confidence": 0.9
                }
            },
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }));
        let value = serde_json::to_value(&response).expect("json");
        assert_eq!(value["answers"]["urgency"]["legend"]["2"], "Today");
        assert_eq!(value["answers"]["red_rectangle"]["box"]["bottom"], 0.66);
        assert_eq!(parse(value), response);
    }

    #[test]
    fn missing_usage_and_answers_are_tolerated() {
        let response = parse(json!({"model": "jev-latest"}));
        assert!(response.answers.is_empty());
        assert_eq!(response.usage.total_tokens(), None);
    }
}
