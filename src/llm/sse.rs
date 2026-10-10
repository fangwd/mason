//! Incremental SSE framing with UTF-8, LF, CRLF, lone-CR, and multiline data support.
//! Incomplete framing at EOF is an error.

use crate::Error;
use crate::error::Result;
use crate::llm::error::LlmError;

/// A single SSE line is bounded; a longer one is a broken or hostile provider.
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;
/// One event's accumulated `data:` payload is bounded the same way.
const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;

/// Incremental SSE framing over a byte stream.
#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    /// Bytes not yet consumed into lines: at most one unfinished line,
    /// possibly ending in a CR that the next chunk may turn into CRLF.
    buf: Vec<u8>,
    /// The `data:` values collected for the event being dispatched.
    data: Vec<String>,
    /// Bytes held in `data`, bounded by [`MAX_EVENT_BYTES`].
    data_bytes: usize,
}

impl SseDecoder {
    /// Consumes a body chunk and returns every event it completed, in order.
    /// Each event is its `data` payload (multiple `data:` lines joined with
    /// `\n`).
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < self.buf.len() {
            let terminator = match self.buf[i] {
                b'\n' => 1,
                b'\r' => match self.buf.get(i + 1) {
                    // A CR at the end of the buffer may still become CRLF;
                    // wait for the next chunk to decide.
                    None => break,
                    Some(b'\n') => 2,
                    Some(_) => 1,
                },
                _ => 0,
            };
            if terminator == 0 {
                i += 1;
                continue;
            }
            if let Some(event) =
                process_line(&mut self.data, &mut self.data_bytes, &self.buf[start..i])?
            {
                events.push(event);
            }
            i += terminator;
            start = i;
        }
        self.buf.drain(..start);
        if self.buf.len() > MAX_LINE_BYTES {
            return Err(Error::Llm(LlmError::Protocol(format!(
                "an SSE line exceeded {MAX_LINE_BYTES} bytes"
            ))));
        }
        Ok(events)
    }

    /// Ends the stream, returning any events the final bytes completed.
    /// Clean EOF leaves nothing buffered; a partial line or an event missing
    /// its blank line is a framing error, never a silently dropped event.
    pub(crate) fn finish(&mut self) -> Result<Vec<String>> {
        let mut events = Vec::new();
        // A trailing CR did end its line: no LF can follow anymore.
        if self.buf.ends_with(b"\r") {
            let end = self.buf.len() - 1;
            let line = std::mem::take(&mut self.buf);
            if let Some(event) = process_line(&mut self.data, &mut self.data_bytes, &line[..end])? {
                events.push(event);
            }
        }
        // A cut-off line or event is a dropped connection: retryable like any
        // other transport failure.
        if !self.buf.is_empty() {
            return Err(Error::Llm(LlmError::Transport(
                "the stream ended in the middle of an SSE line".to_owned(),
            )));
        }
        if !self.data.is_empty() {
            return Err(Error::Llm(LlmError::Transport(
                "the stream ended before an SSE event's blank line".to_owned(),
            )));
        }
        Ok(events)
    }
}

/// Processes one complete line (without its terminator), returning the event
/// it dispatches, if any.
fn process_line(
    data: &mut Vec<String>,
    data_bytes: &mut usize,
    line: &[u8],
) -> Result<Option<String>> {
    if line.is_empty() {
        if data.is_empty() {
            return Ok(None);
        }
        let payload = data.join("\n");
        data.clear();
        *data_bytes = 0;
        return Ok(Some(payload));
    }
    let text = std::str::from_utf8(line).map_err(|_| {
        Error::Llm(LlmError::Protocol(
            "invalid UTF-8 in an SSE line".to_owned(),
        ))
    })?;
    if text.starts_with(':') {
        return Ok(None); // heartbeat or proxy comment: nothing to decode
    }
    let (field, value) = match text.split_once(':') {
        Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
        None => (text, ""),
    };
    if field == "data" {
        // Charge the joining newline too, so empty `data:` lines cannot
        // accumulate past the bound.
        *data_bytes += value.len() + 1;
        if *data_bytes > MAX_EVENT_BYTES {
            return Err(Error::Llm(LlmError::Protocol(format!(
                "an SSE event exceeded {MAX_EVENT_BYTES} bytes"
            ))));
        }
        data.push(value.to_owned());
    }
    // `event:`, `id:` and `retry:` do not change OpenAI chunk decoding.
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pushes the chunks in order, then finishes; returns all events.
    fn decode(chunks: &[&[u8]]) -> Result<Vec<String>> {
        let mut decoder = SseDecoder::default();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.push(chunk)?);
        }
        events.extend(decoder.finish()?);
        Ok(events)
    }

    fn decode_str(chunks: &[&str]) -> Result<Vec<String>> {
        decode(&chunks.iter().map(|c| c.as_bytes()).collect::<Vec<_>>())
    }

    #[test]
    fn dispatches_on_the_blank_line() {
        assert_eq!(decode_str(&["data: hello\n\n"]).unwrap(), ["hello"]);
        assert_eq!(decode_str(&["data: a\n\ndata: b\n\n"]).unwrap(), ["a", "b"]);
        assert_eq!(decode_str(&["\n\ndata: x\n\n"]).unwrap(), ["x"]);
    }

    #[test]
    fn reassembles_events_split_across_chunks() {
        assert_eq!(
            decode_str(&["da", "ta: hel", "lo\n", "\n"]).unwrap(),
            ["hello"]
        );
        assert_eq!(
            decode_str(&["data: x\n", "\ndata: y", "\n\n"]).unwrap(),
            ["x", "y"]
        );
    }

    #[test]
    fn reassembles_utf8_split_across_chunks() {
        let full = "data: café ☕\n\n";
        let bytes = full.as_bytes();
        // Split at every byte position, including mid-character.
        for split in 0..bytes.len() {
            let events = decode(&[&bytes[..split], &bytes[split..]]).unwrap();
            assert_eq!(events, ["café ☕"], "split at {split}");
        }
    }

    #[test]
    fn handles_crlf_and_lone_cr_endings() {
        assert_eq!(decode_str(&["data: x\r\n\r\n"]).unwrap(), ["x"]);
        // Lone CR terminators: the first ends the data line, the second is
        // the blank line that dispatches it (at EOF, via finish).
        assert_eq!(decode_str(&["data: x\r\r"]).unwrap(), ["x"]);
        assert_eq!(
            decode_str(&["data: x\r", "\rmore: ignored\n\n"]).unwrap(),
            ["x"]
        );
        // CRLF split between the CR and the LF across chunks.
        assert_eq!(decode_str(&["data: x\r", "\n\r\n"]).unwrap(), ["x"]);
    }

    #[test]
    fn ignores_comments_and_non_data_fields() {
        assert_eq!(
            decode_str(&[": heartbeat\n\n"]).unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            decode_str(&[":ping\nevent: message\nid: 7\nretry: 100\ndata: p\n\n"]).unwrap(),
            ["p"]
        );
    }

    #[test]
    fn joins_multiline_data_with_newlines() {
        assert_eq!(decode_str(&["data: a\ndata: b\n\n"]).unwrap(), ["a\nb"]);
    }

    #[test]
    fn strips_exactly_one_leading_space() {
        assert_eq!(decode_str(&["data:  x\n\n"]).unwrap(), [" x"]);
        assert_eq!(decode_str(&["data:x\n\n"]).unwrap(), ["x"]);
    }

    #[test]
    fn a_field_line_without_colon_has_an_empty_value() {
        assert_eq!(decode_str(&["data\n\n"]).unwrap(), [""]);
    }

    #[test]
    fn partial_framing_at_eof_is_an_error() {
        let err = decode_str(&["data: x\n\ndata: {\"cho"]).unwrap_err();
        assert!(err.to_string().contains("middle of an SSE line"), "{err}");
        // A complete data line whose event never saw its blank line.
        let err = decode_str(&["data: [DONE]\n"]).unwrap_err();
        assert!(err.to_string().contains("blank line"), "{err}");
        // A CR-terminated data line still misses its blank line.
        let err = decode_str(&["data: x\r"]).unwrap_err();
        assert!(err.to_string().contains("blank line"), "{err}");
    }

    #[test]
    fn invalid_utf8_errors_without_panic() {
        let err = decode(&[b"data: \xff\xfe\n\n"]).unwrap_err();
        assert!(err.to_string().contains("UTF-8"), "{err}");
    }

    #[test]
    fn unbounded_data_accumulation_errors() {
        // Many `data:` lines with no dispatching blank line.
        let mut decoder = SseDecoder::default();
        let line = format!("data: {}\n", "x".repeat(MAX_LINE_BYTES - 100));
        let mut result = Ok(Vec::new());
        for _ in 0..5 {
            result = decoder.push(line.as_bytes());
            if result.is_err() {
                break;
            }
        }
        let err = result.unwrap_err();
        assert!(err.to_string().contains("SSE event exceeded"), "{err}");
    }

    #[test]
    fn empty_data_lines_count_against_the_event_bound() {
        // Each `data:` line is tiny on the wire but costs an entry in `data`
        // and a newline in the joined payload.
        let mut decoder = SseDecoder::default();
        let chunk = "data:\n".repeat(64 * 1024);
        let mut result = Ok(Vec::new());
        for _ in 0..=MAX_EVENT_BYTES / (64 * 1024) {
            result = decoder.push(chunk.as_bytes());
            if result.is_err() {
                break;
            }
        }
        let err = result.unwrap_err();
        assert!(err.to_string().contains("SSE event exceeded"), "{err}");
    }

    #[test]
    fn overlong_lines_error() {
        let mut decoder = SseDecoder::default();
        let huge = vec![b'x'; MAX_LINE_BYTES + 1];
        let err = decoder.push(&huge).unwrap_err();
        assert!(err.to_string().contains("exceeded"), "{err}");
    }
}
