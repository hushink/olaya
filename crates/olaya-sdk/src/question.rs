//! System One question types: noul (yes/no), choice, score, and box.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::content::Content;

/// A question about the supplied state.
///
/// The wire representation is discriminated by the `type` field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul(NoulQuestion),
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
    Box(BoxQuestion),
}

/// The kind of a question.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Noul,
    Choice,
    Score,
    Box,
}

impl QuestionType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Box => "box",
        }
    }
}

/// A yes/no question or statement, answered with the probability of yes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NoulQuestion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

/// Optional descriptions of what counts as a yes or no answer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub is_true: Option<Content>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub is_false: Option<Content>,
}

/// A question that selects one of the named alternatives in `criteria`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Content>,
    /// Choice names and optional descriptions. `None` means an undescribed
    /// choice (serialized as `null`).
    pub criteria: IndexMap<String, Option<Content>>,
}

/// A question that assigns a score using an ordered rubric.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScoreQuestion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Content>,
    /// Ordered descriptions, one per score level starting at zero.
    pub criteria: Vec<Content>,
}

/// A question that locates what its instructions describe in an image.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoxQuestion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
        self.question_type().name()
    }

    pub fn instructions(&self) -> Option<&Content> {
        match self {
            Self::Noul(question) => question.instructions.as_ref(),
            Self::Choice(question) => question.instructions.as_ref(),
            Self::Score(question) => question.instructions.as_ref(),
            Self::Box(question) => question.instructions.as_ref(),
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
    /// A yes/no question with instructions.
    pub fn new(instructions: impl Into<Content>) -> Self {
        Self {
            instructions: Some(instructions.into()),
            criteria: None,
        }
    }

    /// A yes/no question described only by its true/false criteria.
    pub fn from_criteria(criteria: NoulCriteria) -> Self {
        Self {
            instructions: None,
            criteria: Some(criteria),
        }
    }

    pub fn with_criteria(mut self, criteria: NoulCriteria) -> Self {
        self.criteria = Some(criteria);
        self
    }

    pub fn with_instructions(mut self, instructions: impl Into<Content>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

impl NoulCriteria {
    /// Describe both the yes and no outcomes.
    pub fn new(is_true: impl Into<Content>, is_false: impl Into<Content>) -> Self {
        Self {
            is_true: Some(is_true.into()),
            is_false: Some(is_false.into()),
        }
    }
}

impl ChoiceQuestion {
    /// A choice question over named alternatives, without instructions.
    ///
    /// ```
    /// use olaya_sdk::{ChoiceQuestion, Content};
    ///
    /// let question = ChoiceQuestion::new([
    ///     ("calm", Some(Content::from("A neutral or polite message"))),
    ///     ("angry", None),
    /// ]);
    /// assert_eq!(question.criteria.len(), 2);
    /// ```
    pub fn new<I, S>(criteria: I) -> Self
    where
        I: IntoIterator<Item = (S, Option<Content>)>,
        S: Into<String>,
    {
        Self {
            instructions: None,
            criteria: criteria
                .into_iter()
                .map(|(label, description)| (label.into(), description))
                .collect(),
        }
    }

    /// A choice question over undescribed labels.
    pub fn labels<I, S>(labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            instructions: None,
            criteria: labels
                .into_iter()
                .map(|label| (label.into(), None))
                .collect(),
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<Content>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

impl ScoreQuestion {
    /// A score question over ordered rubric levels, without instructions.
    ///
    /// ```
    /// use olaya_sdk::ScoreQuestion;
    ///
    /// let question = ScoreQuestion::new(["can wait", "this week", "today"]);
    /// assert_eq!(question.criteria.len(), 3);
    /// ```
    pub fn new<I, C>(criteria: I) -> Self
    where
        I: IntoIterator<Item = C>,
        C: Into<Content>,
    {
        Self {
            instructions: None,
            criteria: criteria.into_iter().map(Into::into).collect(),
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

impl From<NoulQuestion> for Question {
    fn from(value: NoulQuestion) -> Self {
        Self::Noul(value)
    }
}

impl From<ChoiceQuestion> for Question {
    fn from(value: ChoiceQuestion) -> Self {
        Self::Choice(value)
    }
}

impl From<ScoreQuestion> for Question {
    fn from(value: ScoreQuestion) -> Self {
        Self::Score(value)
    }
}

impl From<BoxQuestion> for Question {
    fn from(value: BoxQuestion) -> Self {
        Self::Box(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serializes_the_official_noul_example() {
        let question = Question::from(NoulQuestion::new("Is this message about billing?"));
        assert_eq!(
            serde_json::to_value(question).expect("json"),
            json!({"type": "noul", "instructions": "Is this message about billing?"})
        );
    }

    #[test]
    fn serializes_the_official_choice_example() {
        let question = Question::from(ChoiceQuestion::new([
            ("angry", Some(Content::from("An upset or hostile message"))),
            ("calm", Some(Content::from("A neutral or polite message"))),
            (
                "excited",
                Some(Content::from("An enthusiastic or eager message")),
            ),
        ]));
        assert_eq!(
            serde_json::to_value(question).expect("json"),
            json!({
                "type": "choice",
                "criteria": {
                    "angry": "An upset or hostile message",
                    "calm": "A neutral or polite message",
                    "excited": "An enthusiastic or eager message"
                }
            })
        );
    }

    #[test]
    fn serializes_undescribed_choices_as_null() {
        let question = Question::from(ChoiceQuestion::labels(["calm", "angry"]));
        assert_eq!(
            serde_json::to_value(question).expect("json"),
            json!({"type": "choice", "criteria": {"calm": null, "angry": null}})
        );
    }

    #[test]
    fn serializes_the_official_score_example() {
        let question = Question::from(ScoreQuestion::new([
            "Can wait",
            "Needs attention this week",
            "Needs attention today",
        ]));
        assert_eq!(
            serde_json::to_value(question).expect("json"),
            json!({
                "type": "score",
                "criteria": [
                    "Can wait",
                    "Needs attention this week",
                    "Needs attention today"
                ]
            })
        );
    }

    #[test]
    fn serializes_noul_criteria() {
        let question = Question::from(NoulQuestion::from_criteria(NoulCriteria::new(
            "Unsolicited advertising",
            "A legitimate conversation",
        )));
        assert_eq!(
            serde_json::to_value(question).expect("json"),
            json!({
                "type": "noul",
                "criteria": {
                    "true": "Unsolicited advertising",
                    "false": "A legitimate conversation"
                }
            })
        );
    }

    #[test]
    fn round_trips_box_questions() {
        let question = Question::from(BoxQuestion::new("The dog"));
        let value = serde_json::to_value(&question).expect("json");
        assert_eq!(value, json!({"type": "box", "instructions": "The dog"}));
        assert_eq!(
            serde_json::from_value::<Question>(value).expect("question"),
            question
        );
    }
}
