//! The `Content` union and the request `State`.
//!
//! The TypeSafe schema accepts a string, a JSON object, an array, or `null` in
//! content fields. Top-level numbers and booleans are rejected. `State` is the
//! same union without `null`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use thiserror::Error;

/// A TypeSafe content value: string, object, array, or null.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    String(String),
    Object(Map<String, Value>),
    Array(Vec<Value>),
    Null,
}

/// A top-level value that is not valid TypeSafe content.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ContentError {
    /// Content must be a string, object, array, or null.
    #[error("content must be a string, object, array, or null")]
    InvalidType,
}

impl Content {
    /// An explicit JSON `null`.
    pub fn null() -> Self {
        Self::Null
    }

    pub fn string(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    pub fn object(value: Map<String, Value>) -> Self {
        Self::Object(value)
    }

    pub fn array(value: Vec<Value>) -> Self {
        Self::Array(value)
    }

    /// Validate and convert a JSON value.
    pub fn from_value(value: Value) -> Result<Self, ContentError> {
        match value {
            Value::Null => Ok(Self::Null),
            Value::String(value) => Ok(Self::String(value)),
            Value::Object(value) => Ok(Self::Object(value)),
            Value::Array(value) => Ok(Self::Array(value)),
            Value::Number(_) | Value::Bool(_) => Err(ContentError::InvalidType),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Map<String, Value>> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    /// Convert this content into a JSON value.
    pub fn into_value(self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::String(value) => Value::String(value),
            Self::Object(value) => Value::Object(value),
            Self::Array(value) => Value::Array(value),
        }
    }
}

impl From<&str> for Content {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<String> for Content {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<Map<String, Value>> for Content {
    fn from(value: Map<String, Value>) -> Self {
        Self::Object(value)
    }
}

impl From<Vec<Value>> for Content {
    fn from(value: Vec<Value>) -> Self {
        Self::Array(value)
    }
}

impl TryFrom<Value> for Content {
    type Error = ContentError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Self::from_value(value)
    }
}

/// The content all questions in a request refer to.
///
/// The schema allows a string, object, or array; `null`, numbers, and booleans
/// are rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct State(Content);

/// A value that is not valid `State`.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum StateError {
    /// State must be a string, object, or array.
    #[error("state must be a string, object, or array")]
    InvalidType,
    #[error(transparent)]
    Content(#[from] ContentError),
}

impl State {
    pub fn text(value: impl Into<String>) -> Self {
        Self(Content::String(value.into()))
    }

    pub fn object(value: Map<String, Value>) -> Self {
        Self(Content::Object(value))
    }

    pub fn array(value: Vec<Value>) -> Self {
        Self(Content::Array(value))
    }

    /// Validate a JSON value as state.
    pub fn from_value(value: Value) -> Result<Self, StateError> {
        match value {
            Value::Null | Value::Number(_) | Value::Bool(_) => Err(StateError::InvalidType),
            value => Ok(Self(Content::from_value(value)?)),
        }
    }

    /// Validate content as state.
    pub fn from_content(content: Content) -> Result<Self, StateError> {
        if content.is_null() {
            Err(StateError::InvalidType)
        } else {
            Ok(Self(content))
        }
    }

    pub fn as_content(&self) -> &Content {
        &self.0
    }

    pub fn into_content(self) -> Content {
        self.0
    }

    pub fn as_str(&self) -> Option<&str> {
        self.0.as_str()
    }

    pub fn as_object(&self) -> Option<&Map<String, Value>> {
        self.0.as_object()
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        self.0.as_array()
    }
}

impl From<&str> for State {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}

impl From<String> for State {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

impl From<Map<String, Value>> for State {
    fn from(value: Map<String, Value>) -> Self {
        Self::object(value)
    }
}

impl From<Vec<Value>> for State {
    fn from(value: Vec<Value>) -> Self {
        Self::array(value)
    }
}

impl TryFrom<Value> for State {
    type Error = StateError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Self::from_value(value)
    }
}

impl TryFrom<Content> for State {
    type Error = StateError;

    fn try_from(value: Content) -> Result<Self, Self::Error> {
        Self::from_content(value)
    }
}

impl Serialize for State {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for State {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_matches_the_official_union() {
        assert_eq!(
            serde_json::to_value(Content::from("hello")).expect("json"),
            json!("hello")
        );
        assert_eq!(
            serde_json::from_value::<Content>(json!({"task": "find", "weight": 2}))
                .expect("object"),
            Content::Object(
                serde_json::from_value(json!({"task": "find", "weight": 2})).expect("map")
            )
        );
        assert_eq!(
            serde_json::from_value::<Content>(json!([1, true, null])).expect("array"),
            Content::Array(vec![json!(1), json!(true), json!(null)])
        );
        assert_eq!(
            serde_json::from_value::<Content>(Value::Null).expect("null"),
            Content::Null
        );
    }

    #[test]
    fn top_level_scalars_are_rejected() {
        assert!(serde_json::from_value::<Content>(json!(3)).is_err());
        assert!(serde_json::from_value::<Content>(json!(true)).is_err());
    }

    #[test]
    fn state_rejects_null_and_scalars() {
        assert!(State::from_value(Value::Null).is_err());
        assert!(State::from_value(json!(1)).is_err());
        assert!(State::from_value(json!(false)).is_err());
        assert!(State::from_value(json!({"message": "hi"})).is_ok());
        assert!(State::from_value(json!(["a", "b"])).is_ok());
    }

    #[test]
    fn state_round_trips() {
        let state = State::object(
            serde_json::from_value(json!({"message": "Please help.", "priority": 2})).expect("map"),
        );
        let value = serde_json::to_value(&state).expect("json");
        assert_eq!(value, json!({"message": "Please help.", "priority": 2}));
        assert_eq!(
            serde_json::from_value::<State>(value).expect("state"),
            state
        );
    }
}
