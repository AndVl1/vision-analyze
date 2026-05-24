//! Line-delimited JSON protocol for the warm-daemon unix socket.
//!
//! Each message is a single JSON line terminated by `\n`.
//! Maximum line length is 64 MiB to prevent unbounded allocations.

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::error::{Result, VisionError};

/// Maximum allowed line size in bytes (64 MiB).
const MAX_LINE_BYTES: usize = 64 * 1024 * 1024;

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
/// Returns `VisionError::Daemon` if the line exceeds 64 MiB or cannot be
/// deserialized.
pub async fn read_message<R, T>(reader: &mut R) -> Result<T>
where
    R: tokio::io::AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    let n = buf_reader
        .read_line(&mut line)
        .await
        .map_err(VisionError::Io)?;

    if n == 0 {
        return Err(VisionError::Daemon("connection closed by peer".into()));
    }
    if line.len() > MAX_LINE_BYTES {
        return Err(VisionError::Daemon(format!(
            "message too large: {} bytes (limit {} MiB)",
            line.len(),
            MAX_LINE_BYTES / 1024 / 1024
        )));
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
}
