//! Lightweight, local OCR for searchable text embedded in images.

use image::GenericImageView;
use ocrs::{ImageSource, OcrEngine as OcrsEngine, OcrEngineParams};
use rten::Model;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Manager};

pub const OCR_ENGINE_ID: &str = "ocrs-0.12";
const MAX_OCR_EDGE: u32 = 2560;
const MAX_OCR_TEXT_BYTES: usize = 64 * 1024;

pub enum OcrOutcome {
    Text(String),
    Empty,
    Unsupported,
}

pub struct OcrEngine {
    engine: Option<OcrsEngine>,
}

impl OcrEngine {
    pub fn new() -> Self {
        Self { engine: None }
    }

    fn model_dir(app: &AppHandle) -> Result<PathBuf, String> {
        #[cfg(debug_assertions)]
        {
            let dev_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/models");
            if dev_path.exists() {
                return Ok(dev_path);
            }
        }

        app.path()
            .resolve("models", tauri::path::BaseDirectory::Resource)
            .map_err(|error| format!("Failed to resolve OCR model directory: {error}"))
    }

    fn load_models(&mut self, app: &AppHandle) -> Result<(), String> {
        if self.engine.is_some() {
            return Ok(());
        }

        let model_dir = Self::model_dir(app)?;
        let detection_path = model_dir.join(crate::t_common::OCR_DETECTION_MODEL);
        let recognition_path = model_dir.join(crate::t_common::OCR_RECOGNITION_MODEL);
        let detection_model = Model::load_file(&detection_path).map_err(|error| {
            format!(
                "Failed to load OCR detection model from {}: {error}",
                detection_path.display()
            )
        })?;
        let recognition_model = Model::load_file(&recognition_path).map_err(|error| {
            format!(
                "Failed to load OCR recognition model from {}: {error}",
                recognition_path.display()
            )
        })?;
        self.engine = Some(
            OcrsEngine::new(OcrEngineParams {
                detection_model: Some(detection_model),
                recognition_model: Some(recognition_model),
                ..Default::default()
            })
            .map_err(|error| format!("Failed to initialize OCR engine: {error}"))?,
        );
        Ok(())
    }

    pub fn recognize(&mut self, app: &AppHandle, file_path: &str) -> Result<OcrOutcome, String> {
        if !supports_path(file_path) {
            return Ok(OcrOutcome::Unsupported);
        }
        self.load_models(app)?;

        let image = image::open(file_path)
            .map_err(|error| format!("Failed to decode image for OCR: {error}"))?;
        let (width, height) = image.dimensions();
        let image = if width > MAX_OCR_EDGE || height > MAX_OCR_EDGE {
            image.thumbnail(MAX_OCR_EDGE, MAX_OCR_EDGE)
        } else {
            image
        };
        let rgb = image.into_rgb8();
        let source = ImageSource::from_bytes(rgb.as_raw(), rgb.dimensions())
            .map_err(|error| format!("Failed to prepare OCR image source: {error}"))?;
        let engine = self.engine.as_ref().expect("OCR engine loaded");
        let input = engine
            .prepare_input(source)
            .map_err(|error| format!("Failed to prepare OCR input: {error}"))?;
        let text = engine
            .get_text(&input)
            .map_err(|error| format!("OCR inference failed: {error}"))?;
        let text = clean_text(&text);
        Ok(match text {
            Some(text) => OcrOutcome::Text(text),
            None => OcrOutcome::Empty,
        })
    }
}

pub struct OcrState(pub Mutex<OcrEngine>);

pub fn supports_path(file_path: &str) -> bool {
    Path::new(file_path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "jfif" | "png" | "gif" | "bmp" | "tif" | "tiff" | "webp" | "avif"
            )
        })
}

fn clean_text(text: &str) -> Option<String> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| line.chars().count() > 1)
        .collect::<Vec<_>>();
    lines.dedup();
    let text = lines.join("\n");
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut end = text.len().min(MAX_OCR_TEXT_BYTES);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    Some(text[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::{clean_text, supports_path};

    #[test]
    fn filters_noise_and_duplicate_lines() {
        assert_eq!(
            clean_text("x\n meme text \nmeme text\n"),
            Some("meme text".to_string())
        );
    }

    #[test]
    fn limits_ocr_to_directly_decodable_images() {
        assert!(supports_path("meme.PNG"));
        assert!(supports_path("photo.webp"));
        assert!(!supports_path("photo.heic"));
        assert!(!supports_path("clip.webm"));
    }
}
