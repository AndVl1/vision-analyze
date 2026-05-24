/// Integration tests for the Unix-socket daemon protocol (`socket.rs`).
///
/// No real daemon is spawned.  Instead each test creates a `UnixListener` on a
/// tempfile path, spins up a server task that reads one request and writes one
/// response, then exercises `socket::read_message` / `socket::write_message` on
/// the client side.
use tempfile::tempdir;
use tokio::net::{UnixListener, UnixStream};

use vision_analyze::socket::{read_message, write_message, SocketRequest, SocketResponse};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Minimal 1×1 transparent PNG as a base64 string for the Infer variant.
fn mini_png_b64() -> String {
    use base64::Engine;
    let bytes: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
        0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
        0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78,
        0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

// ── Health round-trip ─────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn health_request_response_roundtrip() {
    let dir = tempdir().expect("tempdir");
    let sock_path = dir.path().join("test.sock");

    let listener = UnixListener::bind(&sock_path).expect("bind");

    // Server task: read Health request, write Health response.
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let (reader, mut writer) = stream.into_split();
        let mut buf = tokio::io::BufReader::new(reader);

        let req: SocketRequest = {
            use tokio::io::AsyncBufReadExt;
            let mut line = String::new();
            buf.read_line(&mut line).await.expect("read line");
            serde_json::from_str(line.trim_end()).expect("parse request")
        };

        assert!(matches!(req, SocketRequest::Health));

        let resp = SocketResponse::Health {
            status: "ok".into(),
            backend: "mock".into(),
        };
        write_message(&mut writer, &resp).await.expect("write response");
    });

    // Client side.
    let mut stream = UnixStream::connect(&sock_path).await.expect("connect");
    write_message(&mut stream, &SocketRequest::Health)
        .await
        .expect("write request");

    let resp: SocketResponse = read_message(&mut stream).await.expect("read response");

    match resp {
        SocketResponse::Health { status, backend } => {
            assert_eq!(status, "ok");
            assert_eq!(backend, "mock");
        }
        other => panic!("expected Health response, got: {other:?}"),
    }

    server.await.expect("server task panicked");
}

// ── Stop round-trip ───────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn stop_request_response_roundtrip() {
    let dir = tempdir().expect("tempdir");
    let sock_path = dir.path().join("stop.sock");

    let listener = UnixListener::bind(&sock_path).expect("bind");

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let (reader, mut writer) = stream.into_split();
        let mut buf = tokio::io::BufReader::new(reader);

        let req: SocketRequest = {
            use tokio::io::AsyncBufReadExt;
            let mut line = String::new();
            buf.read_line(&mut line).await.expect("read line");
            serde_json::from_str(line.trim_end()).expect("parse request")
        };

        assert!(matches!(req, SocketRequest::Stop));

        write_message(&mut writer, &SocketResponse::Stopped)
            .await
            .expect("write Stopped");
    });

    let mut stream = UnixStream::connect(&sock_path).await.expect("connect");
    write_message(&mut stream, &SocketRequest::Stop)
        .await
        .expect("write Stop");

    let resp: SocketResponse = read_message(&mut stream).await.expect("read Stopped");
    assert!(
        matches!(resp, SocketResponse::Stopped),
        "expected Stopped, got: {resp:?}"
    );

    server.await.expect("server task panicked");
}

// ── Infer round-trip ──────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn infer_request_response_roundtrip() {
    let dir = tempdir().expect("tempdir");
    let sock_path = dir.path().join("infer.sock");

    let listener = UnixListener::bind(&sock_path).expect("bind");

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let (reader, mut writer) = stream.into_split();
        let mut buf = tokio::io::BufReader::new(reader);

        let req: SocketRequest = {
            use tokio::io::AsyncBufReadExt;
            let mut line = String::new();
            buf.read_line(&mut line).await.expect("read line");
            serde_json::from_str(line.trim_end()).expect("parse Infer request")
        };

        match req {
            SocketRequest::Infer { prompt, .. } => {
                assert_eq!(prompt, "describe the image");
            }
            other => panic!("expected Infer, got {other:?}"),
        }

        let ok_resp = SocketResponse::Ok {
            text: "a cat".into(),
            tokens: 3,
            time_ms: 42,
            model: "test-model".into(),
        };
        write_message(&mut writer, &ok_resp)
            .await
            .expect("write Ok response");
    });

    let image_b64 = mini_png_b64();
    let req = SocketRequest::Infer {
        prompt: "describe the image".into(),
        image: image_b64,
        image_b: None,
        max_tokens: 64,
        temperature: 0.0,
        timeout_ms: 5000,
        format: "text".into(),
        preset: None,
    };

    let mut stream = UnixStream::connect(&sock_path).await.expect("connect");
    write_message(&mut stream, &req)
        .await
        .expect("write Infer request");

    let resp: SocketResponse = read_message(&mut stream).await.expect("read Ok response");

    match resp {
        SocketResponse::Ok {
            text,
            tokens,
            time_ms: _,
            model,
        } => {
            assert_eq!(text, "a cat");
            assert_eq!(tokens, 3);
            assert_eq!(model, "test-model");
        }
        other => panic!("expected Ok response, got: {other:?}"),
    }

    server.await.expect("server task panicked");
}

// ── Error response round-trip ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn error_response_roundtrip() {
    let dir = tempdir().expect("tempdir");
    let sock_path = dir.path().join("err.sock");

    let listener = UnixListener::bind(&sock_path).expect("bind");

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let (reader, mut writer) = stream.into_split();
        let mut buf = tokio::io::BufReader::new(reader);

        // Consume the request (Health).
        let _: SocketRequest = {
            use tokio::io::AsyncBufReadExt;
            let mut line = String::new();
            buf.read_line(&mut line).await.expect("read line");
            serde_json::from_str(line.trim_end()).expect("parse request")
        };

        let err_resp = SocketResponse::Error {
            error: "model unavailable".into(),
            code: "backend_unavailable".into(),
        };
        write_message(&mut writer, &err_resp)
            .await
            .expect("write Error response");
    });

    let mut stream = UnixStream::connect(&sock_path).await.expect("connect");
    write_message(&mut stream, &SocketRequest::Health)
        .await
        .expect("write Health");

    let resp: SocketResponse = read_message(&mut stream).await.expect("read Error response");

    match resp {
        SocketResponse::Error { error, code } => {
            assert_eq!(error, "model unavailable");
            assert_eq!(code, "backend_unavailable");
        }
        other => panic!("expected Error response, got: {other:?}"),
    }

    server.await.expect("server task panicked");
}

// ── serde tag correctness ─────────────────────────────────────────────────────

#[test]
fn socket_request_infer_tag_is_snake_case() {
    let req = SocketRequest::Infer {
        prompt: "x".into(),
        image: "y".into(),
        image_b: None,
        max_tokens: 1,
        temperature: 0.0,
        timeout_ms: 1000,
        format: "text".into(),
        preset: None,
    };
    let s = serde_json::to_string(&req).unwrap();
    assert!(s.contains("\"op\":\"infer\""), "op tag must be 'infer', got: {s}");
}

#[test]
fn socket_response_ok_tag_is_snake_case() {
    let resp = SocketResponse::Ok {
        text: "t".into(),
        tokens: 1,
        time_ms: 1,
        model: "m".into(),
    };
    let s = serde_json::to_string(&resp).unwrap();
    assert!(s.contains("\"type\":\"ok\""), "type tag must be 'ok', got: {s}");
}
