//! Image assets for System One requests.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD};

use crate::error::Error;

/// An image format accepted by the TypeSafe API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    Png,
    Jpeg,
}

impl ImageFormat {
    /// The wire name used by inline assets.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
        }
    }

    /// The MIME type used for multipart parts.
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
        }
    }

    /// The filename extension used for multipart parts.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
        }
    }

    /// Detect PNG or JPEG from the leading bytes.
    pub fn detect(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
            Ok(Self::Png)
        } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Ok(Self::Jpeg)
        } else {
            Err(Error::Image {
                message: "unsupported image data; expected a PNG or JPEG image".to_owned(),
            })
        }
    }
}

/// Image bytes ready to attach to a System One request.
///
/// Construct with [`Image::open`], [`Image::from_bytes`], or
/// [`Image::from_base64`]. The format is detected from the bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    bytes: Arc<[u8]>,
    format: ImageFormat,
}

impl Image {
    /// Read an image file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes)
    }

    /// An image from raw encoded bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self, Error> {
        let bytes: Vec<u8> = bytes.into();
        if bytes.is_empty() {
            return Err(Error::Image {
                message: "image data is empty".to_owned(),
            });
        }
        let format = ImageFormat::detect(&bytes)?;
        Ok(Self {
            bytes: Arc::from(bytes),
            format,
        })
    }

    /// An image from base64-encoded bytes. ASCII whitespace is ignored.
    pub fn from_base64(data: impl AsRef<str>) -> Result<Self, Error> {
        let cleaned: String = data
            .as_ref()
            .chars()
            .filter(|character| !character.is_ascii_whitespace())
            .collect();
        let bytes = STANDARD
            .decode(&cleaned)
            .or_else(|_| STANDARD_NO_PAD.decode(&cleaned))
            .map_err(|error| Error::Image {
                message: format!("invalid base64 image data: {error}"),
            })?;
        Self::from_bytes(bytes)
    }

    /// The raw image bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn shared_bytes(&self) -> Arc<[u8]> {
        self.bytes.clone()
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Base64-encode the bytes for an inline JSON asset.
    pub fn to_base64(&self) -> String {
        STANDARD.encode(&self.bytes)
    }

    /// Copy the image bytes into a vector.
    pub fn to_vec(&self) -> Vec<u8> {
        self.bytes.to_vec()
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Image")
            .field("format", &self.format)
            .field("len", &self.bytes.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3];
    const JPEG: &[u8] = &[0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3];

    #[test]
    fn detects_png_and_jpeg() {
        assert_eq!(
            Image::from_bytes(PNG).expect("png").format(),
            ImageFormat::Png
        );
        assert_eq!(
            Image::from_bytes(JPEG).expect("jpeg").format(),
            ImageFormat::Jpeg
        );
    }

    #[test]
    fn rejects_empty_and_unknown_images() {
        assert!(Image::from_bytes(Vec::new()).is_err());
        assert!(Image::from_bytes(b"not an image").is_err());
    }

    #[test]
    fn base64_round_trips_with_whitespace() {
        let image = Image::from_bytes(PNG).expect("png");
        let encoded = image.to_base64();
        let wrapped = format!("  {}\n\n{}  ", &encoded[..8], &encoded[8..]);
        let decoded = Image::from_base64(wrapped).expect("base64");
        assert_eq!(decoded, image);
    }

    #[test]
    fn base64_accepts_missing_padding() {
        let image = Image::from_bytes(JPEG).expect("jpeg");
        let encoded = image.to_base64();
        let unpadded = encoded.trim_end_matches('=');
        assert_eq!(Image::from_base64(unpadded).expect("base64"), image);
    }

    #[test]
    fn rejects_invalid_base64() {
        assert!(Image::from_base64("!!!not base64!!!").is_err());
    }
}
