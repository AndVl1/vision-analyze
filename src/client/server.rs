//! HTTP client for an externally running `llama-server` (OpenAI-compatible API).

use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::{encode_b64, InferenceRequest, InferenceResponse};
use crate::error::{Result, VisionError};

#[derive(Debug, Clone)]
pub struct ServerClient {
    base_url: String,
    http: Client,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: &'static str,
    content: Vec<MessagePart>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum MessagePart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    model: Option<String>,
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    #[serde(default)]
    completion_tokens: u32,
}

impl ServerClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        while url.ends_with('/') {
            url.pop();
        }
        let http = Client::builder()
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        Self {
            base_url: url,
            http,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Quick health probe — does NOT consume the request timeout, uses its own short window.
    pub async fn probe(base_url: &str, timeout: Duration) -> bool {
        let http = match Client::builder().timeout(timeout).build() {
            Ok(c) => c,
            Err(_) => return false,
        };
        let url = format!("{}/health", base_url.trim_end_matches('/'));
        match http.get(&url).send().await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }

    pub async fn health(&self) -> Result<()> {
        let url = format!("{}/health", self.base_url);
        let r = self
            .http
            .get(&url)
            .timeout(Duration::from_millis(500))
            .send()
            .await
            .map_err(|e| VisionError::BackendUnavailable(format!("health: {e}")))?;
        if r.status().is_success() {
            Ok(())
        } else {
            Err(VisionError::BackendUnavailable(format!(
                "health http {}",
                r.status()
            )))
        }
    }

    pub async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse> {
        let started = Instant::now();
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut parts: Vec<MessagePart> = Vec::new();
        parts.push(MessagePart::Text {
            text: req.prompt.clone(),
        });
        parts.push(MessagePart::ImageUrl {
            image_url: ImageUrl {
                url: format!("data:image/png;base64,{}", encode_b64(&req.image)),
            },
        });
        if let Some(b) = req.image_b.as_ref() {
            parts.push(MessagePart::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:image/png;base64,{}", encode_b64(b)),
                },
            });
        }

        let body = ChatRequest {
            model: "vision".into(),
            messages: vec![ChatMessage {
                role: "user",
                content: parts,
            }],
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            stream: false,
        };

        let resp = self
            .http
            .post(&url)
            .timeout(req.timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| VisionError::Analysis(format!("server request: {e}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| VisionError::Analysis(format!("server read body: {e}")))?;

        if !status.is_success() {
            // Try to extract `error.message` from OpenAI-style error body.
            let snippet = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| text.chars().take(200).collect());
            return Err(VisionError::Analysis(format!(
                "server http {status}: {snippet}"
            )));
        }

        let parsed: ChatResponse = serde_json::from_str(&text).map_err(|e| {
            VisionError::Analysis(format!(
                "server parse: {e}; body: {}",
                text.chars().take(200).collect::<String>()
            ))
        })?;

        let content = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| VisionError::Analysis("server returned no choices".into()))?;

        Ok(InferenceResponse {
            text: content,
            tokens: parsed.usage.map(|u| u.completion_tokens).unwrap_or(0),
            time_ms: started.elapsed().as_millis() as u64,
            model: parsed.model.unwrap_or_else(|| "unknown".into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_trailing_slash_in_base_url() {
        let c = ServerClient::new("http://127.0.0.1:8080///");
        assert_eq!(c.base_url(), "http://127.0.0.1:8080");
    }

    #[test]
    fn chat_request_serializes_content_array() {
        let body = ChatRequest {
            model: "vision".into(),
            messages: vec![ChatMessage {
                role: "user",
                content: vec![
                    MessagePart::Text { text: "hi".into() },
                    MessagePart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,AAA".into(),
                        },
                    },
                ],
            }],
            temperature: 0.0,
            max_tokens: 16,
            stream: false,
        };
        let s = serde_json::to_string(&body).unwrap();
        assert!(s.contains("\"type\":\"text\""));
        assert!(s.contains("\"type\":\"image_url\""));
        assert!(s.contains("data:image/png;base64,AAA"));
    }
}
