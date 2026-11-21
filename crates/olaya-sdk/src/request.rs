//! System One request construction and wire preparation.

use std::sync::Arc;

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::content::State;
use crate::error::Error;
use crate::image::Image;
use crate::question::Question;

/// Questions keyed by the names used to identify their answers.
pub type Questions = IndexMap<String, Question>;

/// Assets keyed by the reference used for them in state or questions.
pub type Assets = IndexMap<String, Asset>;

/// The maximum number of choices in one choice question.
pub const MAX_CHOICE_OPTIONS: usize = 255;

/// The maximum number of levels in one score question.
pub const MAX_SCORE_LEVELS: usize = 10;

/// An image attached to a System One request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Asset {
    /// Sent as raw bytes in a multipart part.
    Image(Image),
    /// Sent inline as base64 in the JSON body.
    Inline(Image),
}

impl Asset {
    /// Send the image as a multipart part (efficient for large images).
    pub fn image(image: Image) -> Self {
        Self::Image(image)
    }

    /// Send the image inline as base64 JSON.
    pub fn inline(image: Image) -> Self {
        Self::Inline(image)
    }

    /// A multipart image from encoded bytes.
    pub fn image_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self, Error> {
        Ok(Self::Image(Image::from_bytes(bytes)?))
    }

    /// A multipart image from a file.
    pub fn image_path(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        Ok(Self::Image(Image::open(path)?))
    }

    /// An inline image from base64-encoded bytes.
    pub fn inline_base64(data: impl AsRef<str>) -> Result<Self, Error> {
        Ok(Self::Inline(Image::from_base64(data)?))
    }

    /// An inline image from a file.
    pub fn inline_path(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        Ok(Self::Inline(Image::open(path)?))
    }

    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline(_))
    }

    pub fn image_ref(&self) -> &Image {
        match self {
            Self::Image(image) | Self::Inline(image) => image,
        }
    }
}

impl From<Image> for Asset {
    fn from(value: Image) -> Self {
        Self::Image(value)
    }
}

/// A System One request.
///
/// `model` is optional; when it is `None` the client's default model is used.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemOneRequest {
    pub model: Option<String>,
    pub state: State,
    pub questions: Questions,
    pub assets: Assets,
    /// Extra top-level request fields, shallow-merged after the typed fields.
    /// Colliding keys override the typed fields.
    pub extra_body: Map<String, Value>,
}

impl SystemOneRequest {
    pub fn new(state: impl Into<State>, questions: Questions) -> Self {
        Self {
            model: None,
            state: state.into(),
            questions,
            assets: Assets::new(),
            extra_body: Map::new(),
        }
    }

    /// Set the model name or alias for this request.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Attach an asset keyed by the reference used in the request content.
    pub fn asset(mut self, name: impl Into<String>, asset: impl Into<Asset>) -> Self {
        self.assets.insert(name.into(), asset.into());
        self
    }

    /// Attach a multipart image.
    pub fn image(mut self, name: impl Into<String>, image: Image) -> Self {
        self.assets.insert(name.into(), Asset::Image(image));
        self
    }

    /// Attach an inline base64 image.
    pub fn inline_image(mut self, name: impl Into<String>, image: Image) -> Self {
        self.assets.insert(name.into(), Asset::Inline(image));
        self
    }

    /// Add an extra top-level request field.
    pub fn extra_body(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra_body.insert(key.into(), value.into());
        self
    }

    /// Validate constraints from the official schema that Rust types cannot
    /// express.
    pub fn validate(&self) -> Result<(), Error> {
        if let Some(model) = &self.model
            && model.trim().is_empty()
        {
            return Err(Error::InvalidRequest {
                message: "model must not be empty".to_owned(),
            });
        }
        if self.questions.is_empty() {
            return Err(Error::InvalidRequest {
                message: "questions must contain at least one named question".to_owned(),
            });
        }
        for (name, question) in &self.questions {
            match question {
                Question::Choice(question) => {
                    if question.criteria.is_empty() {
                        return Err(Error::InvalidRequest {
                            message: format!(
                                "questions.{name}.criteria must contain at least one choice"
                            ),
                        });
                    }
                    if question.criteria.len() > MAX_CHOICE_OPTIONS {
                        return Err(Error::InvalidRequest {
                            message: format!(
                                "questions.{name}.criteria has {} choices; the maximum is {MAX_CHOICE_OPTIONS}",
                                question.criteria.len()
                            ),
                        });
                    }
                }
                Question::Score(question) => {
                    if question.criteria.is_empty() {
                        return Err(Error::InvalidRequest {
                            message: format!(
                                "questions.{name}.criteria must contain at least one score level"
                            ),
                        });
                    }
                    if question.criteria.len() > MAX_SCORE_LEVELS {
                        return Err(Error::InvalidRequest {
                            message: format!(
                                "questions.{name}.criteria has {} levels; the maximum is {MAX_SCORE_LEVELS}",
                                question.criteria.len()
                            ),
                        });
                    }
                }
                Question::Noul(_) | Question::Box(_) => {}
            }
        }
        for name in self.assets.keys() {
            if name.is_empty() {
                return Err(Error::InvalidRequest {
                    message: "asset references must not be empty".to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Build the JSON body and the multipart image parts for `model`.
    pub(crate) fn prepare(&self, model: &str) -> Result<PreparedBody, Error> {
        let mut wire_assets = IndexMap::new();
        let mut images = Vec::new();
        for (name, asset) in &self.assets {
            match asset {
                Asset::Inline(image) => {
                    wire_assets.insert(
                        name.clone(),
                        WireAsset {
                            kind: "image",
                            format: Some(image.format().as_str()),
                            data_base64: Some(image.to_base64()),
                            part: None,
                        },
                    );
                }
                Asset::Image(image) => {
                    let part = format!("image-{}", images.len());
                    wire_assets.insert(
                        name.clone(),
                        WireAsset {
                            kind: "image",
                            format: None,
                            data_base64: None,
                            part: Some(part.clone()),
                        },
                    );
                    images.push(MultipartImage {
                        part,
                        filename: format!("image-{}.{}", images.len(), image.format().extension()),
                        content_type: image.format().content_type(),
                        bytes: image.shared_bytes(),
                    });
                }
            }
        }

        let wire = WireRequest {
            model,
            state: &self.state,
            questions: &self.questions,
            assets: wire_assets,
        };
        let mut json = serde_json::to_value(&wire).map_err(|error| Error::Config {
            message: format!("serializing the request body: {error}"),
        })?;
        if let Some(object) = json.as_object_mut() {
            for (key, value) in &self.extra_body {
                object.insert(key.clone(), value.clone());
            }
        }
        Ok(PreparedBody { json, images })
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    state: &'a State,
    questions: &'a Questions,
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    assets: IndexMap<String, WireAsset>,
}

#[derive(Serialize)]
struct WireAsset {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    part: Option<String>,
}

/// A wire-ready body shared by the async and blocking clients.
pub(crate) struct PreparedBody {
    pub(crate) json: Value,
    pub(crate) images: Vec<MultipartImage>,
}

pub(crate) struct MultipartImage {
    pub(crate) part: String,
    pub(crate) filename: String,
    pub(crate) content_type: &'static str,
    pub(crate) bytes: Arc<[u8]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::Content;
    use crate::question::{ChoiceQuestion, NoulQuestion, ScoreQuestion};
    use serde_json::json;

    fn questions() -> Questions {
        let mut questions = Questions::new();
        questions.insert(
            "billing".to_owned(),
            Question::from(NoulQuestion::new("Is this message about billing?")),
        );
        questions.insert(
            "tone".to_owned(),
            Question::from(ChoiceQuestion::labels(["calm", "angry"])),
        );
        questions.insert(
            "urgency".to_owned(),
            Question::from(ScoreQuestion::new(["low", "high"])),
        );
        questions
    }

    #[test]
    fn prepares_the_official_request_shape() {
        let request = SystemOneRequest::new("I was charged twice.", questions());
        request.validate().expect("valid");
        let body = request.prepare("jev-latest").expect("prepared");
        assert_eq!(body.json["model"], "jev-latest");
        assert_eq!(body.json["state"], "I was charged twice.");
        assert_eq!(body.json["questions"]["billing"]["type"], "noul");
        assert_eq!(
            body.json["questions"]["tone"]["criteria"]["calm"],
            json!(null)
        );
        assert!(body.json.get("assets").is_none());
        assert!(body.images.is_empty());
    }

    #[test]
    fn extra_body_is_shallow_merged_last() {
        let request = SystemOneRequest::new("text", questions())
            .model("explicit")
            .extra_body("beam_width", 4)
            .extra_body("model", "overridden");
        let body = request.prepare("resolved").expect("prepared");
        assert_eq!(body.json["beam_width"], 4);
        assert_eq!(body.json["model"], "overridden");
    }

    #[test]
    fn inline_assets_use_base64_json() {
        let image =
            Image::from_bytes(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]).expect("png");
        let request = SystemOneRequest::new("Look: <photo>", questions())
            .asset("<photo>", Asset::inline(image));
        let body = request.prepare("jev-preview").expect("prepared");
        assert_eq!(body.json["assets"]["<photo>"]["type"], "image");
        assert_eq!(body.json["assets"]["<photo>"]["format"], "png");
        assert!(body.json["assets"]["<photo>"]["data_base64"].is_string());
        assert!(body.images.is_empty());
    }

    #[test]
    fn file_assets_become_unique_multipart_parts() {
        let png =
            Image::from_bytes(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]).expect("png");
        let jpeg = Image::from_bytes(vec![0xFF, 0xD8, 0xFF, 0xE0]).expect("jpeg");
        let request = SystemOneRequest::new("Compare <a> and <b>", questions())
            .image("<a>", png)
            .image("<b>", jpeg);
        let body = request.prepare("jev-preview").expect("prepared");
        assert_eq!(body.json["assets"]["<a>"]["part"], "image-0");
        assert_eq!(body.json["assets"]["<b>"]["part"], "image-1");
        assert_eq!(body.images.len(), 2);
        assert_eq!(body.images[0].part, "image-0");
        assert_eq!(body.images[0].content_type, "image/png");
        assert_eq!(body.images[1].part, "image-1");
        assert_eq!(body.images[1].content_type, "image/jpeg");
    }

    #[test]
    fn rejects_empty_questions_and_empty_criteria() {
        let request = SystemOneRequest::new("text", Questions::new());
        assert!(request.validate().is_err());

        let mut questions = Questions::new();
        questions.insert(
            "tone".to_owned(),
            Question::from(ChoiceQuestion::new(Vec::<(String, Option<Content>)>::new())),
        );
        let request = SystemOneRequest::new("text", questions);
        assert!(request.validate().is_err());

        let mut questions = Questions::new();
        questions.insert(
            "urgency".to_owned(),
            Question::from(ScoreQuestion::new(Vec::<Content>::new())),
        );
        let request = SystemOneRequest::new("text", questions);
        assert!(request.validate().is_err());
    }

    #[test]
    fn rejects_too_many_options() {
        let mut questions = Questions::new();
        questions.insert(
            "tone".to_owned(),
            Question::from(ChoiceQuestion::labels(
                (0..=MAX_CHOICE_OPTIONS).map(|index| format!("choice-{index}")),
            )),
        );
        assert!(SystemOneRequest::new("text", questions).validate().is_err());
    }

    #[test]
    fn rejects_empty_model_and_asset_reference() {
        let request = SystemOneRequest::new("text", questions()).model("  ");
        assert!(request.validate().is_err());
        let request = SystemOneRequest::new("text", questions()).asset(
            "",
            Asset::image_bytes(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]).expect("png"),
        );
        assert!(request.validate().is_err());
    }
}
