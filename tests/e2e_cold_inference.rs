/// End-to-end tests that launch the real `vision-analyze` binary against a
/// wiremock HTTP server (cold inference path, no daemon involved).
///
/// The daemon warm-path is bypassed by pointing `VISION_SOCKET_PATH` at a
/// tempdir socket that does not exist — `try_client()` will fail immediately
/// and the code falls through to the cold path.
use std::io::Write;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── constants ─────────────────────────────────────────────────────────────────

/// Minimal 1×1 transparent PNG.
const MIN_PNG_1X1: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
    0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
    0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
    0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
    0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

// ── helpers ───────────────────────────────────────────────────────────────────

/// Create a PNG file in `dir` and return its path as a String.
fn write_png(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("test.png");
    let mut f = std::fs::File::create(&path).expect("create png");
    f.write_all(MIN_PNG_1X1).expect("write png");
    path.to_str().unwrap().to_string()
}

/// Start a `vision-analyze` Command wired to the mock server with the daemon
/// paths pointing to non-existent files so the warm path is bypassed.
async fn vision_cmd_with_server(
    server: &MockServer,
    dir: &tempfile::TempDir,
) -> assert_cmd::Command {
    let mut cmd = Command::cargo_bin("vision-analyze").expect("binary");
    cmd.env("VISION_LLAMA_SERVER", server.uri())
        .env("VISION_LLAMA_BIN", "/nonexistent/llama-cli")
        .env(
            "VISION_SOCKET_PATH",
            dir.path().join("no.sock").to_str().unwrap(),
        )
        .env(
            "VISION_PID_PATH",
            dir.path().join("no.pid").to_str().unwrap(),
        )
        // Force server backend to avoid auto-detect trying the cli.
        .env("RUST_LOG", "off")
        .arg("--backend")
        .arg("server");
    cmd
}

fn valid_openai_body() -> serde_json::Value {
    serde_json::json!({
        "model": "qwen2-vl-2b",
        "choices": [{"message": {"content": "hello world"}}],
        "usage": {"completion_tokens": 5}
    })
}

// ── cold inference, text output ───────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn cold_inference_text_output_contains_model_response() {
    let server = MockServer::start().await;
    let dir = tempdir().expect("tempdir");

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(valid_openai_body()))
        .mount(&server)
        .await;

    let png = write_png(&dir);
    let mut cmd = vision_cmd_with_server(&server, &dir).await;

    cmd.args([&png, "describe"])
        .timeout(std::time::Duration::from_secs(15))
        .assert()
        .success()
        .stdout(predicate::str::contains("hello world"));
}

// ── cold inference, json output ───────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn cold_inference_json_output_is_valid_json() {
    let server = MockServer::start().await;
    let dir = tempdir().expect("tempdir");

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(valid_openai_body()))
        .mount(&server)
        .await;

    let png = write_png(&dir);
    let mut cmd = vision_cmd_with_server(&server, &dir).await;

    let output = cmd
        .args([&png, "describe", "--format", "json"])
        .timeout(std::time::Duration::from_secs(15))
        .output()
        .expect("run command");

    assert!(output.status.success(), "exit code should be 0");

    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let v: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("stdout must be valid JSON");

    assert_eq!(v["result"], "hello world", "result field must match");
    assert!(v["tokens"].is_number(), "tokens field must be present");
    assert!(v["time_ms"].is_number(), "time_ms field must be present");
    assert!(v["model"].is_string(), "model field must be present");
}

// ── json output with explicit flag ───────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn cold_inference_json_result_field_equals_model_text() {
    let server = MockServer::start().await;
    let dir = tempdir().expect("tempdir");

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(valid_openai_body()))
        .mount(&server)
        .await;

    let png = write_png(&dir);
    let mut cmd = vision_cmd_with_server(&server, &dir).await;

    let output = cmd
        .args([&png, "describe", "--format", "json"])
        .timeout(std::time::Duration::from_secs(15))
        .output()
        .expect("run");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["result"].as_str().unwrap(), "hello world");
}

// ── backend unavailable → exit 2 ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn cold_inference_server_500_exits_nonzero() {
    let server = MockServer::start().await;
    let dir = tempdir().expect("tempdir");

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    // The completions endpoint fails — inference should return an error.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&server)
        .await;

    let png = write_png(&dir);
    let mut cmd = vision_cmd_with_server(&server, &dir).await;

    cmd.args([&png, "describe"])
        .timeout(std::time::Duration::from_secs(15))
        .assert()
        .failure();
}

// ── no image argument → exit 1 ────────────────────────────────────────────────

#[test]
fn missing_image_argument_exits_nonzero() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("vision-analyze")
        .unwrap()
        .env("VISION_LLAMA_SERVER", "http://127.0.0.1:19999")
        .env("VISION_LLAMA_BIN", "/nonexistent")
        .env(
            "VISION_SOCKET_PATH",
            dir.path().join("no.sock").to_str().unwrap(),
        )
        .env(
            "VISION_PID_PATH",
            dir.path().join("no.pid").to_str().unwrap(),
        )
        .env("RUST_LOG", "off")
        // Pass a prompt but no image — should fail with InvalidArgs.
        .arg("-- describe something")
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        .failure();
}
