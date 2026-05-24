//! Line-delimited JSON protocol for the warm-daemon unix socket.
//!
//! Each message is a single JSON line terminated by `\n`.
//! Maximum line length is 64 MiB to prevent unbounded allocations.

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

use crate::error::{Result, VisionError};

/// Maximum allowed frame size in bytes (64 MiB).
///
/// Used by both `socket.rs` and `daemon.rs` — single source of truth.
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Request sent from a CLI client to the daemon over the unix socket.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum SocketRequest {
    Infer {
        prompt: String,
        /// Primary image, base64-encoded.
        image: String,
        /// Optional second image, base64-encoded.
        image_b: Option<String>,
        max_tokens: u32,
        temperature: f32,
        timeout_ms: u64,
        format: String,
        preset: Option<String>,
    },
    Health,
    Stop,
}

/// Response sent from the daemon back to the CLI client.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SocketResponse {
    Ok {
        text: String,
        tokens: u32,
        time_ms: u64,
        model: String,
    },
    Health {
        status: String,
        backend: String,
    },
    Stopped,
    Error {
        error: String,
        code: String,
    },
}

/// Read one newline-terminated JSON message from `reader`.
///
/// Returns `VisionError::PayloadTooLarge` if the line exceeds [`MAX_FRAME_BYTES`],
/// or `VisionError::Daemon` on connection close / parse error.
///
/// # DoS protection
///
/// Uses `.take(MAX_FRAME_BYTES + 1)` to cap the actual read: `read_line` would
/// otherwise pre-allocate up to the entire line before we can check the length.
pub async fn read_message<R, T>(reader: &mut R) -> Result<T>
where
    R: tokio::io::AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let buf_reader = BufReader::new(reader);
    let mut line = String::new();
    // Cap actual I/O: read at most MAX_FRAME_BYTES + 1 bytes.  If the caller
    // sends exactly MAX_FRAME_BYTES of payload without a newline the +1 cap
    // will be hit and we'll detect the oversize on the length check below.
    let n = buf_reader
        .take((MAX_FRAME_BYTES as u64) + 1)
        .read_line(&mut line)
        .await
        .map_err(VisionError::Io)?;

    if n == 0 {
        return Err(VisionError::Daemon("connection closed by peer".into()));
    }
    if line.len() > MAX_FRAME_BYTES {
        return Err(VisionError::PayloadTooLarge(line.len(), MAX_FRAME_BYTES));
    }

    serde_json::from_str(line.trim_end()).map_err(VisionError::Json)
}

/// Write a message as a single JSON line (`{...}\n`) to `writer`.
pub async fn write_message<W, T>(writer: &mut W, msg: &T) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
    T: Serialize,
{
    let mut line = serde_json::to_string(msg)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::BufReader;

    #[tokio::test]
    async fn roundtrip_infer_request() {
        let req = SocketRequest::Infer {
            prompt: "describe".into(),
            image: "abc123".into(),
            image_b: None,
            max_tokens: 512,
            temperature: 0.0,
            timeout_ms: 5000,
            format: "text".into(),
            preset: None,
        };

        let mut buf: Vec<u8> = Vec::new();
        write_message(&mut buf, &req).await.unwrap();

        assert!(buf.ends_with(b"\n"));

        let mut reader = BufReader::new(buf.as_slice());
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let decoded: SocketRequest = serde_json::from_str(line.trim_end()).unwrap();
        assert!(matches!(decoded, SocketRequest::Infer { .. }));
    }

    #[tokio::test]
    async fn roundtrip_health_response() {
        let resp = SocketResponse::Health {
            status: "ok".into(),
            backend: "cli".into(),
        };
        let mut buf: Vec<u8> = Vec::new();
        write_message(&mut buf, &resp).await.unwrap();

        let mut reader = BufReader::new(buf.as_slice());
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let decoded: SocketResponse = serde_json::from_str(line.trim_end()).unwrap();
        assert!(matches!(decoded, SocketResponse::Health { .. }));
    }

    #[test]
    fn socket_request_stop_tag() {
        let s = serde_json::to_string(&SocketRequest::Stop).unwrap();
        assert!(s.contains("\"op\":\"stop\""));
    }

    #[test]
    fn socket_response_stopped_tag() {
        let s = serde_json::to_string(&SocketResponse::Stopped).unwrap();
        assert!(s.contains("\"type\":\"stopped\""));
    }

    /// C4: read_message rejects payloads that exceed MAX_FRAME_BYTES.
    ///
    /// We craft a fake line that is exactly MAX_FRAME_BYTES + 1 bytes to verify
    /// that PayloadTooLarge is returned rather than a huge allocation.
    /// NOTE: We only test the *detection* path here (line.len() > MAX check)
    /// by using a valid but oversized line, not actually allocating 64 MiB in
    /// tests.  The .take() cap prevents that in production; here we simulate
    /// the post-read check using a line that exceeds the threshold.
    #[tokio::test]
    async fn read_message_rejects_oversized_frame() {
        use crate::error::VisionError;

        // Build a line that is just over the limit.  Use 1 extra byte beyond the cap.
        // We insert it as a JSON string with enough padding that the total JSON
        // payload is MAX_FRAME_BYTES + 1.
        // Because the take() cap prevents the actual read, we directly test that
        // the length check fires by constructing a BufReader with a pre-built oversized line.
        let oversized: Vec<u8> = {
            // Fill with spaces then append \n; content doesn't matter — JSON parse
            // won't be reached because the size check fires first.
            let mut v = vec![b'x'; MAX_FRAME_BYTES + 1];
            v.push(b'\n');
            v
        };
        let result: crate::error::Result<SocketRequest> =
            read_message(&mut oversized.as_slice()).await;
        assert!(
            matches!(result, Err(VisionError::PayloadTooLarge(_, _))),
            "expected PayloadTooLarge, got: {result:?}"
        );
    }

    /// C4: read_message accepts a small valid frame normally.
    #[tokio::test]
    async fn read_message_accepts_small_frame() {
        let req = SocketRequest::Health;
        let mut buf: Vec<u8> = Vec::new();
        write_message(&mut buf, &req).await.unwrap();
        let decoded: crate::error::Result<SocketRequest> = read_message(&mut buf.as_slice()).await;
        assert!(decoded.is_ok());
    }
}
