/// Integration tests for `output::render`.
///
/// Covers Text and Json formats including the edge case where the model text
/// itself is a valid JSON string — it must be stored as a string field in the
/// output object, not re-serialized as a nested value.
use vision_analyze::client::InferenceResponse;
use vision_analyze::config::OutputFormat;
use vision_analyze::output::render;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_response(text: &str) -> InferenceResponse {
    InferenceResponse {
        text: text.into(),
        tokens: 10,
        time_ms: 50,
        model: "test-model".into(),
    }
}

// ── Text format ───────────────────────────────────────────────────────────────

#[test]
fn text_format_returns_raw_text() {
    let resp = make_response("hello world");
    assert_eq!(render(&resp, OutputFormat::Text), "hello world");
}

#[test]
fn text_format_preserves_whitespace_and_newlines() {
    let resp = make_response("line1\nline2\t  end");
    assert_eq!(render(&resp, OutputFormat::Text), "line1\nline2\t  end");
}

#[test]
fn text_format_with_empty_string() {
    let resp = make_response("");
    assert_eq!(render(&resp, OutputFormat::Text), "");
}

// ── Json format ───────────────────────────────────────────────────────────────

#[test]
fn json_format_contains_result_field() {
    let resp = make_response("hi");
    let out = render(&resp, OutputFormat::Json);
    let v: serde_json::Value = serde_json::from_str(&out).expect("output must be valid JSON");
    assert_eq!(v["result"], "hi");
}

#[test]
fn json_format_contains_tokens_field() {
    let resp = make_response("hi");
    let out = render(&resp, OutputFormat::Json);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["tokens"], 10);
}

#[test]
fn json_format_contains_time_ms_field() {
    let resp = make_response("hi");
    let out = render(&resp, OutputFormat::Json);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["time_ms"], 50);
}

#[test]
fn json_format_contains_model_field() {
    let resp = make_response("hi");
    let out = render(&resp, OutputFormat::Json);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["model"], "test-model");
}

/// Edge case: when the model text is itself valid JSON (e.g. `{"x":1}`),
/// the `result` field in the output JSON must still be a *string*, not a
/// nested JSON object.  The render function must not parse the text.
#[test]
fn json_format_stores_json_text_as_string_not_nested_object() {
    let json_text = r#"{"x":1,"y":"hello"}"#;
    let resp = make_response(json_text);
    let out = render(&resp, OutputFormat::Json);

    let v: serde_json::Value = serde_json::from_str(&out).expect("output must be valid JSON");
    // `result` must be a string, not an object.
    assert!(
        v["result"].is_string(),
        "result field must be a string even when model output is JSON, got: {}",
        v["result"]
    );
    assert_eq!(v["result"].as_str().unwrap(), json_text);
}

#[test]
fn json_format_stores_array_text_as_string() {
    // Model returns a JSON array — must still be stored as a string.
    let array_text = r#"[1, 2, 3]"#;
    let resp = make_response(array_text);
    let out = render(&resp, OutputFormat::Json);

    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(v["result"].is_string());
    assert_eq!(v["result"].as_str().unwrap(), array_text);
}

#[test]
fn json_format_is_single_line() {
    let resp = make_response("some text");
    let out = render(&resp, OutputFormat::Json);
    assert!(!out.contains('\n'), "JSON output must be a single line");
}

// ── Tokens and time_ms precision ─────────────────────────────────────────────

#[test]
fn json_format_zero_tokens_and_zero_time_ms() {
    let resp = InferenceResponse {
        text: "ok".into(),
        tokens: 0,
        time_ms: 0,
        model: "m".into(),
    };
    let out = render(&resp, OutputFormat::Json);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["tokens"], 0);
    assert_eq!(v["time_ms"], 0);
}

#[test]
fn json_format_large_token_count() {
    let resp = InferenceResponse {
        text: "long response".into(),
        tokens: u32::MAX,
        time_ms: u64::MAX,
        model: "big-model".into(),
    };
    let out = render(&resp, OutputFormat::Json);
    // Must still be valid JSON without overflow.
    serde_json::from_str::<serde_json::Value>(&out)
        .expect("render must produce valid JSON even for u32::MAX tokens");
}
