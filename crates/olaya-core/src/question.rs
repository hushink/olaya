use crate::content::{Content, deserialize_optional_content};
use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};

/// A question about the supplied state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul(NoulQuestion),
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
    Box(BoxQuestion),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Noul,
    Choice,
    Score,
    Box,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoulQuestion {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub instructions: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub is_true: Option<Content>,
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub is_false: Option<Content>,
}

impl<'de> Deserialize<'de> for NoulCriteria {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // The shared parser lower-cases noul keys; the last spelling wins.
        let criteria = IndexMap::<String, Content>::deserialize(deserializer)?;
        let mut is_true = None;
        let mut is_false = None;
        for (key, value) in criteria {
            match key.to_ascii_lowercase().as_str() {
                "true" => is_true = Some(value),
                "false" => is_false = Some(value),
                _ => {}
            }
        }
        Ok(Self { is_true, is_false })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestion {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub instructions: Option<Content>,
    pub criteria: IndexMap<String, Content>,
    /// Decima's extension: read the choice with the ordinal head.
    #[serde(default, skip_serializing_if = "is_false")]
    pub ordinal: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreQuestion {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub instructions: Option<Content>,
    pub criteria: Vec<Content>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxQuestion {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub instructions: Option<Content>,
}

impl Question {
    pub fn question_type(&self) -> QuestionType {
        match self {
            Self::Noul(_) => QuestionType::Noul,
            Self::Choice(_) => QuestionType::Choice,
            Self::Score(_) => QuestionType::Score,
            Self::Box(_) => QuestionType::Box,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Noul(_) => "noul",
            Self::Choice(_) => "choice",
            Self::Score(_) => "score",
            Self::Box(_) => "box",
        }
    }

    /// Whether this is a choice read by a model's ordinal head.
    pub fn is_ordinal(&self) -> bool {
        matches!(self, Self::Choice(choice) if choice.ordinal)
    }

    pub fn instructions(&self) -> Option<&Content> {
        match self {
            Self::Noul(question) => question.instructions.as_ref(),
            Self::Choice(question) => question.instructions.as_ref(),
            Self::Score(question) => question.instructions.as_ref(),
            Self::Box(question) => question.instructions.as_ref(),
        }
    }

    /// The instruction text a family renders. An absent or explicit `null`
    /// instruction means the question id; other non-string content renders as
    /// Python's `json.dumps` (the rule shared by every family).
    pub fn instructions_or_id(&self, question_id: &str) -> String {
        match self.instructions() {
            None | Some(Content::Null) => question_id.to_owned(),
            Some(Content::String(text)) => text.clone(),
            Some(content) => crate::pyjson::dumps_content(content),
        }
    }

    /// Number of options the model scores; `0` for a box question.
    pub fn num_options(&self) -> usize {
        match self {
            Self::Noul(_) => 2,
            Self::Choice(question) => question.criteria.len(),
            Self::Score(question) => question.criteria.len(),
            Self::Box(_) => 0,
        }
    }
}

impl NoulQuestion {
    pub fn new(instructions: impl Into<Content>) -> Self {
        Self {
            instructions: Some(instructions.into()),
            criteria: None,
        }
    }

    pub fn with_criteria(mut self, criteria: NoulCriteria) -> Self {
        self.criteria = Some(criteria);
        self
    }
}

impl NoulCriteria {
    pub fn new(is_true: impl Into<Content>, is_false: impl Into<Content>) -> Self {
        Self {
            is_true: Some(is_true.into()),
            is_false: Some(is_false.into()),
        }
    }
}

impl ChoiceQuestion {
    pub fn new(criteria: IndexMap<String, Content>) -> Self {
        Self {
            instructions: None,
            criteria,
            ordinal: false,
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<Content>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Read this choice with the model's ordinal head (Decima extension).
    pub fn with_ordinal(mut self, ordinal: bool) -> Self {
        self.ordinal = ordinal;
        self
    }
}

impl ScoreQuestion {
    pub fn new(criteria: Vec<Content>) -> Self {
        Self {
            instructions: None,
            criteria,
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<Content>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

impl BoxQuestion {
    pub fn new(instructions: impl Into<Content>) -> Self {
        Self {
            instructions: Some(instructions.into()),
        }
    }
}
