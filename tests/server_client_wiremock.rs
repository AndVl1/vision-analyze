/// Integration tests for `ServerClient` using a wiremock HTTP server.
///
/// No real llama-server is needed — every HTTP call is intercepted by wiremock.
use std::time::Duration;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use vision_analyze::client::server::ServerClient;
use vision_analyze::client::InferenceRequest;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Minimal 1×1 transparent PNG.
const MIN_PNG_1X1: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
    0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
    0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
    0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
    0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

fn make_inference_request() -> InferenceRequest {
    InferenceRequest {
        prompt: "describe".into(),
        image: MIN_PNG_1X1.to_vec(),
        image_b: None,
        max_tokens: 64,
        temperature: 0.0,
        timeout: Duration::from_secs(5),
    }
}

fn valid_openai_response_body() -> serde_json::Value {
    serde_json::json!({
        "model": "qwen2-vl-2b",
        "choices": [{"message": {"content": "hello world"}}],
        "usage": {"completion_tokens": 5}
    })
}

// ── probe ─────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn probe_returns_true_when_health_endpoint_is_200() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let result = ServerClient::probe(&server.uri(), Duration::from_secs(2)).await;
    assert!(result, "probe should return true for 200 /health");
}

#[tokio::test(flavor = "multi_thread")]
async fn probe_returns_false_when_server_not_running() {
    // Point at a port that is definitely not listening.
    let result = ServerClient::probe("http://127.0.0.1:19998", Duration::from_millis(200)).await;
    assert!(!result, "probe should return false for unreachable server");
}

// ── health ────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn health_ok_when_server_returns_200() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let client = ServerClient::new(server.uri());
    client.health().await.expect("health should succeed");
}

#[tokio::test(flavor = "multi_thread")]
async fn health_err_when_server_returns_500() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let client = ServerClient::new(server.uri());
    let result = client.health().await;
    assert!(result.is_err(), "health should fail for 500");
}

// ── infer ─────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn infer_returns_text_tokens_and_model_on_success() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(valid_openai_response_body()),
        )
        .mount(&server)
        .await;

    let client = ServerClient::new(server.uri());
    let resp = client
        .infer(make_inference_request())
        .await
        .expect("infer should succeed");

    assert_eq!(resp.text, "hello world");
    assert_eq!(resp.tokens, 5);
    assert_eq!(resp.model, "qwen2-vl-2b");
}

#[tokio::test(flavor = "multi_thread")]
async fn infer_returns_err_analysis_on_500() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&server)
        .await;

    let client = ServerClient::new(server.uri());
    let result = client.infer(make_inference_request()).await;

    assert!(result.is_err(), "infer should return Err on 500");
    let err = result.unwrap_err();
    // Should be an Analysis error (exit code 1, not 2).
    assert_eq!(
        err.exit_code(),
        1,
        "Analysis error should map to exit code 1"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn infer_extracts_openai_error_message_from_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {"message": "model not loaded", "type": "invalid_request_error"}
            })),
        )
        .mount(&server)
        .await;

    let client = ServerClient::new(server.uri());
    let err = client
        .infer(make_inference_request())
        .await
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("model not loaded"),
        "error message should be extracted from OpenAI error body, got: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn infer_handles_missing_usage_gracefully() {
    let server = MockServer::start().await;

    // Response without `usage` field — tokens should default to 0.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "model": "test-model",
            "choices": [{"message": {"content": "result text"}}]
        })))
        .mount(&server)
        .await;

    let client = ServerClient::new(server.uri());
    let resp = client
        .infer(make_inference_request())
        .await
        .expect("infer without usage should succeed");

    assert_eq!(resp.text, "result text");
    assert_eq!(resp.tokens, 0, "missing usage should default to 0 tokens");
}
