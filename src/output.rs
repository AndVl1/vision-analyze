//! Render inference responses to the configured output format.

use crate::client::InferenceResponse;
use crate::config::OutputFormat;

/// Render an [`InferenceResponse`] to a plain string suitable for `println!`.
///
/// - [`OutputFormat::Text`] — returns the raw text from the model.
/// - [`OutputFormat::Json`] — returns a single-line JSON object with all
///   response fields.
///
/// # Examples
///
/// ```
/// use vision_analyze::client::InferenceResponse;
/// use vision_analyze::config::OutputFormat;
/// use vision_analyze::output::render;
///
/// let resp = InferenceResponse {
///     text: "a button".into(),
///     tokens: 3,
///     time_ms: 42,
///     model: "qwen2".into(),
/// };
/// let out = render(&resp, OutputFormat::Text);
/// assert_eq!(out, "a button");
/// ```
pub fn render(resp: &InferenceResponse, format: OutputFormat) -> String {
    match format {
        OutputFormat::Text => resp.text.clone(),
        OutputFormat::Json => serde_json::json!({
            "result": resp.text,
            "tokens": resp.tokens,
            "time_ms": resp.time_ms,
            "model": resp.model,
        })
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> InferenceResponse {
        InferenceResponse {
            text: "Hello, world".into(),
            tokens: 5,
            time_ms: 100,
            model: "test-model".into(),
        }
    }

    #[test]
    fn text_format_returns_raw_text() {
        assert_eq!(render(&sample(), OutputFormat::Text), "Hello, world");
    }

    #[test]
    fn json_format_contains_all_fields() {
        let out = render(&sample(), OutputFormat::Json);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["result"], "Hello, world");
        assert_eq!(v["tokens"], 5);
        assert_eq!(v["time_ms"], 100);
        assert_eq!(v["model"], "test-model");
    }

    #[test]
    fn json_format_is_single_line() {
        let out = render(&sample(), OutputFormat::Json);
        assert!(!out.contains('\n'));
    }
}
