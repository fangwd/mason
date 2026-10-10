//! Bounded tool previews, spill files, and retention. Capture caps limit both completed
//! outputs and streamed output.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use super::ToolError;

/// Model-visible line budget, truncation notice included.
pub const MAX_LINES: usize = 2_000;
/// Model-visible byte budget, truncation notice included.
pub const MAX_BYTES: usize = 50 * 1024;
/// Read/grep previews clip individual lines at this many characters.
pub const MAX_LINE_LENGTH: usize = 2_000;
/// Default per-call capture cap for streamed output.
pub const DEFAULT_CAPTURE_CAP: u64 = 10 * 1024 * 1024;
/// Default spill retention.
pub const DEFAULT_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// Cleanup runs at most this often.
const CLEANUP_INTERVAL: Duration = Duration::from_secs(60 * 60);
/// Spill files untouched this long may be evicted when the quota is full.
const EVICTABLE_AFTER: Duration = Duration::from_secs(60 * 60);
/// Tail buffer exceeds the preview budget so clipping can discard partial lines.
const TAIL_BYTES: usize = 64 * 1024;

/// Which end of a long output the preview keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// Keep the first lines (tool results, search output).
    Head,
    /// Keep the last lines (command output).
    Tail,
}

/// Injectable wall clock: retention tests never wait for real time.
pub type WallClock = Arc<dyn Fn() -> SystemTime + Send + Sync>;

pub fn system_clock() -> WallClock {
    Arc::new(SystemTime::now)
}

/// Configurable spill directory, retention, capture cap, and clock.
pub struct OutputSpill {
    dir: PathBuf,
    storage_limit: u64,
    quota: tokio::sync::Mutex<()>,
    retention: Duration,
    capture_cap: u64,
    clock: WallClock,
    next_cleanup: Mutex<Option<SystemTime>>,
}

impl std::fmt::Debug for OutputSpill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputSpill")
            .field("dir", &self.dir)
            .field("retention", &self.retention)
            .field("capture_cap", &self.capture_cap)
            .finish_non_exhaustive()
    }
}

impl OutputSpill {
    /// A spill store with default retention, capture cap and the real clock.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            storage_limit: 256 * 1024 * 1024,
            quota: tokio::sync::Mutex::new(()),
            retention: DEFAULT_RETENTION,
            capture_cap: DEFAULT_CAPTURE_CAP,
            clock: system_clock(),
            next_cleanup: Mutex::new(None),
        }
    }

    /// The production default directory: `<tmp>/mason-<uid>` on Unix, where the
    /// temporary directory may be shared between users; `<tmp>/mason` elsewhere.
    pub fn default_dir() -> PathBuf {
        #[cfg(unix)]
        let name = format!("mason-{}", nix::unistd::Uid::current());
        #[cfg(not(unix))]
        let name = "mason".to_owned();
        std::env::temp_dir().join(name)
    }

    /// Create private storage and perform startup retention cleanup.
    pub async fn prepare(&self) -> Result<(), ToolError> {
        create_spill_dir(&self.dir).await?;
        self.maybe_cleanup().await;
        Ok(())
    }

    /// Maximum total logical size of retained and in-progress spill files.
    pub fn with_storage_limit(mut self, bytes: u64) -> Self {
        self.storage_limit = bytes;
        self
    }

    /// Sets the spill retention period.
    pub fn with_retention(mut self, retention: Duration) -> Self {
        self.retention = retention;
        self
    }

    /// Sets the per-call capture cap for streamed output.
    pub fn with_capture_cap(mut self, capture_cap: u64) -> Self {
        self.capture_cap = capture_cap;
        self
    }

    /// Injects a wall clock for retention tests.
    pub fn with_clock(mut self, clock: WallClock) -> Self {
        self.clock = clock;
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn capture_cap(&self) -> u64 {
        self.capture_cap
    }

    /// Caps captured content, spilling oversized previews with truncation notices
    /// included in the display bounds.
    pub async fn truncate(&self, output: &str, keep: Keep) -> Result<String, ToolError> {
        let cap = usize::try_from(self.capture_cap).unwrap_or(usize::MAX);
        let (output, capped) = if output.len() > cap {
            let mut end = cap;
            while end > 0 && !output.is_char_boundary(end) {
                end -= 1;
            }
            (&output[..end], true)
        } else {
            (output, false)
        };
        // Budget notices before clipping to keep the full preview within bounds.
        let cap_suffix = capped.then(|| cap_note(self.capture_cap));
        let suffix_len = cap_suffix.as_ref().map_or(0, String::len);
        let total_lines = output.split('\n').count();
        if total_lines + usize::from(capped) <= MAX_LINES && output.len() + suffix_len <= MAX_BYTES
        {
            return Ok(match &cap_suffix {
                Some(note) => format!("{output}{note}"),
                None => output.to_owned(),
            });
        }
        // Without spill room, the clipped preview still goes back to the model.
        let path = match self.save(output).await {
            Ok(path) => Ok(path),
            Err(ToolError::Failure(reason)) => Err(reason),
            Err(fatal) => return Err(fatal),
        };
        // Reserve worst-case notice sizes before clipping.
        let worst_note = truncation_note(
            total_lines,
            path.as_deref().map_err(String::as_str),
            self.retention,
        );
        let (kept, kept_lines) = clip(
            &output.split('\n').collect::<Vec<_>>(),
            keep,
            MAX_LINES.saturating_sub(3 + usize::from(capped)),
            MAX_BYTES.saturating_sub(worst_note.len() + 2 + suffix_len),
        );
        let dropped = total_lines.saturating_sub(kept_lines);
        let note = truncation_note(
            dropped,
            path.as_deref().map_err(String::as_str),
            self.retention,
        );
        let mut rendered = match keep {
            Keep::Head => format!("{kept}\n\n{note}"),
            Keep::Tail => format!("{note}\n\n{kept}"),
        };
        if let Some(note) = cap_suffix {
            rendered.push_str(&note);
        }
        Ok(rendered)
    }

    /// Saves output under a unique name after rate-limited retention cleanup.
    pub async fn save(&self, output: &str) -> Result<PathBuf, ToolError> {
        if output.len() as u64 > self.capture_cap {
            return Err(ToolError::failure("spill output exceeds its capture cap"));
        }
        let (mut file, path) = self.create_spill_file().await?;
        // The returned path must already hold the bytes: tokio dispatches file
        // writes to a blocking pool, so they can outlive `write_all`.
        async {
            tokio::io::AsyncWriteExt::write_all(&mut file, output.as_bytes()).await?;
            tokio::io::AsyncWriteExt::flush(&mut file).await?;
            file.set_len(output.len() as u64).await
        }
        .await
        .map_err(|err| {
            ToolError::Failure(format!(
                "cannot save the full output to {}: {err}",
                path.display()
            ))
        })?;
        Ok(path)
    }

    /// Creates a unique spill file after rate-limited retention cleanup.
    pub(crate) async fn create_spill_file(&self) -> Result<(tokio::fs::File, PathBuf), ToolError> {
        self.maybe_cleanup().await;
        create_spill_dir(&self.dir).await?;
        let _quota = self.quota.lock().await;
        let lock_path = self.dir.join(".quota-lock");
        let directory_lock = tokio::task::spawn_blocking(move || {
            let mut options = std::fs::OpenOptions::new();
            options.read(true).write(true).create(true).truncate(false);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options
                    .mode(0o600)
                    .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits());
            }
            let file = options.open(lock_path)?;
            file.lock()?;
            Ok::<_, std::io::Error>(file)
        })
        .await
        .map_err(|_| ToolError::failure("spill quota lock failed"))?
        .map_err(|_| ToolError::failure("cannot lock spill quota"))?;
        let mut used = 0u64;
        // (name, size, modified); names start with the creation time in millis.
        let mut files: Vec<(String, u64, SystemTime)> = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.dir)
            .await
            .map_err(|_| ToolError::failure("cannot inspect spill storage"))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|_| ToolError::failure("cannot inspect spill storage"))?
        {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("output-") {
                let meta = entry
                    .metadata()
                    .await
                    .map_err(|_| ToolError::failure("cannot inspect spill file"))?;
                used = used.saturating_add(meta.len());
                files.push((
                    name,
                    meta.len(),
                    meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                ));
            }
        }
        // Make room by evicting the oldest outputs. A file written within the hour
        // may be a capture still in progress, so it stays.
        files.sort();
        let settled = (self.clock)()
            .checked_sub(EVICTABLE_AFTER)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        for (name, size, modified) in files {
            if used.saturating_add(self.capture_cap) <= self.storage_limit {
                break;
            }
            if modified < settled && tokio::fs::remove_file(self.dir.join(&name)).await.is_ok() {
                used = used.saturating_sub(size);
            }
        }
        if used.saturating_add(self.capture_cap) > self.storage_limit {
            return Err(ToolError::failure(
                "spill storage quota exhausted; remove old outputs or increase the quota",
            ));
        }
        let millis = (self.clock)()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = self
            .dir
            .join(format!("output-{millis}-{}.txt", uuid::Uuid::new_v4()));
        let mut options = tokio::fs::OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options.open(&path).await.map_err(|err| {
            ToolError::Failure(format!(
                "cannot save the full output to {}: {err}",
                path.display()
            ))
        })?;
        // Reserve each capture's maximum size while holding the directory quota lock.
        // Truncate the reservation when capture completes. This also coordinates
        // independently constructed OutputSpill values using the same directory.
        file.set_len(self.capture_cap)
            .await
            .map_err(|_| ToolError::failure("cannot reserve spill storage"))?;
        drop(directory_lock);
        Ok((file, path))
    }

    /// Deletes spill files older than the retention period. Best effort:
    /// failures are logged, never fatal.
    pub async fn cleanup(&self, now: SystemTime) {
        let cutoff = now
            .checked_sub(self.retention)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let Ok(mut entries) = tokio::fs::read_dir(&self.dir).await else {
            return;
        };
        loop {
            match entries.next_entry().await {
                Ok(Some(entry)) => {
                    let path = entry.path();
                    let is_spill = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("output-") && name.ends_with(".txt"));
                    if !is_spill {
                        continue;
                    }
                    match tokio::fs::metadata(&path).await {
                        Ok(meta) => {
                            let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                            if modified >= cutoff {
                                continue;
                            }
                            if let Err(err) = tokio::fs::remove_file(&path).await
                                && err.kind() != std::io::ErrorKind::NotFound
                            {
                                tracing::warn!("cannot clean up {}: {err}", path.display());
                            }
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                        Err(err) => tracing::warn!("cannot clean up {}: {err}", path.display()),
                    }
                }
                Ok(None) => return,
                Err(err) => {
                    tracing::warn!("cannot list {}: {err}", self.dir.display());
                    return;
                }
            }
        }
    }

    async fn maybe_cleanup(&self) {
        let now = (self.clock)();
        {
            // Release the cleanup guard before awaiting filesystem work.
            let Ok(mut next) = self.next_cleanup.lock() else {
                return;
            };
            if next.is_some_and(|next| next > now) {
                return;
            }
            *next = Some(now + CLEANUP_INTERVAL);
        }
        self.cleanup(now).await;
    }

    /// Starts a streamed capture (bash): bounded memory, incremental spill,
    /// per-call cap.
    pub fn capture(self: &Arc<Self>) -> OutputCapture {
        OutputCapture::new(self.clone())
    }
}

/// The rendered result of a streamed capture.
#[derive(Debug)]
pub struct CapturedOutput {
    /// The model-visible text: the bounded preview plus notes.
    pub text: String,
    /// The spill file, when the output exceeded the preview bounds.
    pub path: Option<PathBuf>,
    /// Whether the capture cap was hit (the spill holds only the prefix).
    pub capped: bool,
}

/// Bounded stream capture with a tail preview, incremental spilling, and a cap signal.
pub struct OutputCapture {
    spill: Arc<OutputSpill>,
    tail: Vec<u8>,
    total_bytes: u64,
    newlines: u64,
    last_was_newline: bool,
    file: Option<(tokio::fs::File, PathBuf)>,
    capped: bool,
}

impl OutputCapture {
    fn new(spill: Arc<OutputSpill>) -> Self {
        Self {
            spill,
            tail: Vec::new(),
            total_bytes: 0,
            newlines: 0,
            last_was_newline: false,
            file: None,
            capped: false,
        }
    }

    /// Appends a chunk. Returns `true` when the capture cap was reached and the
    /// producer should be stopped; later chunks are discarded. A spill write failure
    /// stops the capture and becomes a tool error; the caller kills the producer and
    /// removes the partial file.
    pub async fn append(&mut self, chunk: &[u8]) -> Result<bool, ToolError> {
        if self.capped {
            return Ok(true);
        }
        let remaining = self.spill.capture_cap.saturating_sub(self.total_bytes);
        let captured = chunk
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let capped = chunk.len() > captured;
        let chunk = &chunk[..captured];
        self.total_bytes += chunk.len() as u64;
        self.newlines += chunk.iter().filter(|byte| **byte == b'\n').count() as u64;
        if let Some(last) = chunk.last() {
            self.last_was_newline = *last == b'\n';
        }
        if self.file.is_none()
            && (self.tail.len() + chunk.len() > MAX_BYTES || self.line_count() > MAX_LINES as u64)
        {
            // Spill the full history before evicting bytes from the in-memory tail.
            let (mut file, path) = self.spill.create_spill_file().await?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &self.tail)
                .await
                .map_err(|err| {
                    ToolError::Failure(format!(
                        "cannot save the full output to {}: {err}",
                        path.display()
                    ))
                })?;
            tokio::io::AsyncWriteExt::write_all(&mut file, chunk)
                .await
                .map_err(|err| {
                    ToolError::Failure(format!(
                        "cannot save the full output to {}: {err}",
                        path.display()
                    ))
                })?;
            self.file = Some((file, path));
        } else if let Some((file, path)) = &mut self.file {
            let path = path.clone();
            tokio::io::AsyncWriteExt::write_all(file, chunk)
                .await
                .map_err(|err| {
                    ToolError::Failure(format!(
                        "cannot save the full output to {}: {err}",
                        path.display()
                    ))
                })?;
        }
        // Keep the newest output in the preview, including after spilling begins.
        self.tail.extend_from_slice(chunk);
        if self.file.is_some() && self.tail.len() > TAIL_BYTES {
            let excess = self.tail.len() - TAIL_BYTES;
            self.tail.drain(..excess);
        }
        if capped {
            self.capped = true;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn path(&self) -> Option<&Path> {
        self.file.as_ref().map(|(_, path)| path.as_path())
    }

    pub fn capped(&self) -> bool {
        self.capped
    }

    /// Returns captured text or a bounded tail preview, including truncation and cap
    /// notices. `trailer` (e.g. an exit-code line) is appended inside the bounds, so
    /// the engine never truncates it away.
    pub async fn finish(mut self, trailer: &str) -> Result<CapturedOutput, ToolError> {
        let capped = self.capped;
        let suffix = format!(
            "{}{trailer}",
            if capped {
                cap_note(self.spill.capture_cap)
            } else {
                String::new()
            }
        );
        let suffix_lines = suffix.matches('\n').count();
        if self.file.is_none() {
            let text = String::from_utf8_lossy(&self.tail).trim_end().to_owned();
            let text = if text.is_empty() {
                "(no output)".to_owned()
            } else {
                text
            };
            if text.len() + suffix.len() <= MAX_BYTES
                && text.split('\n').count() + suffix_lines <= MAX_LINES
            {
                return Ok(CapturedOutput {
                    text: text + &suffix,
                    path: None,
                    capped,
                });
            }
            // The suffix does not fit next to the full text: spill it rather than
            // dropping lines silently.
            let (mut file, path) = self.spill.create_spill_file().await?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &self.tail)
                .await
                .map_err(|err| {
                    ToolError::Failure(format!(
                        "cannot save the full output to {}: {err}",
                        path.display()
                    ))
                })?;
            self.file = Some((file, path));
        }
        let (mut file, path) = self.file.take().expect("the spill file was just ensured");
        tokio::io::AsyncWriteExt::flush(&mut file)
            .await
            .map_err(|err| {
                ToolError::Failure(format!(
                    "cannot save the full output to {}: {err}",
                    path.display()
                ))
            })?;
        file.set_len(self.total_bytes)
            .await
            .map_err(|_| ToolError::failure("cannot finish spill reservation"))?;
        drop(file);
        // The tail may start mid-line once bytes were evicted.
        let partial_first_line = self.total_bytes > self.tail.len() as u64;
        let text = String::from_utf8_lossy(&self.tail);
        let mut lines: Vec<&str> = text.split('\n').collect();
        // Drop the partial first line only when real content follows it; a
        // trailing newline alone splits off an empty element, not a line.
        if partial_first_line && lines.iter().skip(1).any(|line| !line.is_empty()) {
            lines.remove(0);
        }
        let total_lines = self.line_count();
        // Reserve space for the truncation note, the cap notice and the trailer.
        let worst_note = truncation_note(total_lines as usize, Ok(&path), self.spill.retention);
        let (kept, kept_lines) = clip(
            &lines,
            Keep::Tail,
            MAX_LINES.saturating_sub(3 + suffix_lines),
            MAX_BYTES.saturating_sub(worst_note.len() + 2 + suffix.len()),
        );
        let dropped = total_lines.saturating_sub(kept_lines as u64);
        let note = truncation_note(dropped as usize, Ok(&path), self.spill.retention);
        let rendered = format!("{note}\n\n{}{suffix}", kept.trim_end());
        Ok(CapturedOutput {
            text: rendered,
            path: Some(path),
            capped,
        })
    }

    fn line_count(&self) -> u64 {
        if self.total_bytes == 0 {
            0
        } else {
            self.newlines + 1
        }
    }
}

/// Creates the spill directory privately (0700 on Unix) and refuses a symlink or a
/// directory owned by another user: spilled tool output may hold secrets.
async fn create_spill_dir(dir: &Path) -> Result<(), ToolError> {
    let mut builder = tokio::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(0o700);
    builder.create(dir).await.map_err(|err| {
        ToolError::Failure(format!(
            "cannot create the spill directory {}: {err}",
            dir.display()
        ))
    })?;
    // Inspect the path itself: a planted symlink would pass a following check.
    let meta = tokio::fs::symlink_metadata(dir).await.map_err(|err| {
        ToolError::Failure(format!(
            "cannot inspect the spill directory {}: {err}",
            dir.display()
        ))
    })?;
    if !meta.is_dir() {
        return Err(ToolError::Failure(format!(
            "the spill directory {} is a symlink or not a directory; configure a private spill directory",
            dir.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.uid() != nix::unistd::Uid::current().as_raw() {
            return Err(ToolError::Failure(format!(
                "the spill directory {} is owned by another user; configure a private spill directory",
                dir.display()
            )));
        }
    }
    Ok(())
}

fn truncation_note(dropped: usize, path: Result<&Path, &str>, retention: Duration) -> String {
    let days = (retention.as_secs().div_ceil(86_400)).max(1);
    match path {
        Ok(path) => format!(
            "...{dropped} lines truncated. Full output saved to: {}\n\
             Use grep or read with offset/limit to view the rest. The saved output expires after {days} days.",
            path.display()
        ),
        Err(reason) => {
            format!("...{dropped} lines truncated. The full output was not saved: {reason}")
        }
    }
}

fn cap_note(cap: u64) -> String {
    format!(
        "\nOutput exceeded the {cap}-byte capture cap; only the captured prefix was kept, and a running command was stopped."
    )
}

/// Keeps whole lines from the chosen end within the budgets; returns text and the
/// count of whole lines kept. When nothing but empty lines fits before a line too
/// long for the byte budget, that line is clipped at a char boundary instead of
/// dropped, so the preview is never empty.
fn clip(lines: &[&str], keep: Keep, line_budget: usize, byte_budget: usize) -> (String, usize) {
    let mut kept: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let ordered: Box<dyn Iterator<Item = &str>> = match keep {
        Keep::Head => Box::new(lines.iter().copied()),
        Keep::Tail => Box::new(lines.iter().rev().copied()),
    };
    for line in ordered {
        if kept.len() >= line_budget {
            break;
        }
        bytes += line.len() + 1;
        if bytes > byte_budget {
            if kept.iter().all(|line| line.is_empty()) && line_budget > 0 {
                let partial = clip_bytes(line, byte_budget, keep);
                if !partial.is_empty() {
                    return (partial.to_owned(), 0);
                }
            }
            break;
        }
        kept.push(line);
    }
    if keep == Keep::Tail {
        kept.reverse();
    }
    let count = kept.len();
    (kept.join("\n"), count)
}

/// The longest prefix (head) or suffix (tail) of `line` within `budget` bytes.
fn clip_bytes(line: &str, budget: usize, keep: Keep) -> &str {
    let budget = budget.min(line.len());
    match keep {
        Keep::Head => {
            let mut end = budget;
            while !line.is_char_boundary(end) {
                end -= 1;
            }
            &line[..end]
        }
        Keep::Tail => {
            let mut start = line.len() - budget;
            while !line.is_char_boundary(start) {
                start += 1;
            }
            &line[start..]
        }
    }
}

/// Clips a single preview line to [`MAX_LINE_LENGTH`] characters.
pub fn clip_line(line: &str) -> String {
    if line.chars().count() <= MAX_LINE_LENGTH {
        return line.to_owned();
    }
    let clipped: String = line.chars().take(MAX_LINE_LENGTH).collect();
    format!("{clipped}...")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use proptest::prelude::*;

    use super::*;

    fn test_spill(dir: &Path) -> Arc<OutputSpill> {
        Arc::new(OutputSpill::new(dir))
    }

    /// Manually advanced retention-test clock.
    fn fake_clock(start: SystemTime) -> (WallClock, Arc<AtomicU64>) {
        let millis = Arc::new(AtomicU64::new(0));
        let counter = millis.clone();
        let clock: WallClock =
            Arc::new(move || start + Duration::from_millis(counter.load(Ordering::SeqCst)));
        (clock, millis)
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
        let runtime = RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
        });
        runtime.block_on(future)
    }

    fn spill_path(rendered: &str) -> PathBuf {
        let capture = rendered
            .split_once("Full output saved to: ")
            .expect("a truncation note")
            .1;
        PathBuf::from(capture.lines().next().unwrap())
    }

    #[tokio::test]
    async fn short_output_passes_through_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());
        let out = "just a few lines\n".repeat(10);
        assert_eq!(spill.truncate(&out, Keep::Head).await.unwrap(), out);
        assert_eq!(
            std::fs::read_dir(spill.dir())
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("output-"))
                .count(),
            0,
            "nothing spilled"
        );
    }

    #[tokio::test]
    async fn long_output_keeps_its_tail_and_saves_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());
        let output = (0..3000)
            .map(|i| format!("l{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = spill.truncate(&output, Keep::Tail).await.unwrap();
        let expected_dropped = 3000 - (MAX_LINES - 3);
        assert!(
            out.starts_with(&format!(
                "...{expected_dropped} lines truncated. Full output saved to: "
            )),
            "{out}"
        );
        assert!(out.ends_with("l2999"), "the tail ends with the last line");
        let path = spill_path(&out);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap().split('\n').count(),
            3000
        );
        assert!(
            out.split('\n').count() <= MAX_LINES,
            "{} lines",
            out.split('\n').count()
        );
        assert!(out.len() <= MAX_BYTES, "{} bytes", out.len());
    }

    #[tokio::test]
    async fn head_keep_preserves_the_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());
        let output = (0..2500)
            .map(|i| format!("l{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = spill.truncate(&output, Keep::Head).await.unwrap();
        assert!(out.starts_with("l0\nl1\n"), "{out}");
        assert!(
            out.contains("lines truncated. Full output saved to:"),
            "{out}"
        );
        assert!(out.len() <= MAX_BYTES);
        assert!(out.split('\n').count() <= MAX_LINES);
    }

    #[tokio::test]
    async fn spill_files_are_unique_and_old_outputs_expire() {
        let dir = tempfile::tempdir().unwrap();
        let now = SystemTime::now();
        let (clock, advance) = fake_clock(now);
        let spill = Arc::new(OutputSpill::new(dir.path()).with_clock(clock));
        let output = "long\n".repeat(3000);

        let mut paths = Vec::new();
        for _ in 0..4 {
            let out = spill.truncate(&output, Keep::Head).await.unwrap();
            paths.push(spill_path(&out));
        }
        let unique: std::collections::HashSet<_> = paths.iter().collect();
        assert_eq!(unique.len(), 4, "concurrent spills must not collide");

        // Age the first file beyond the retention period.
        let old = now - Duration::from_secs(8 * 24 * 60 * 60);
        filetime::set_file_mtime(&paths[0], filetime::FileTime::from_system_time(old)).unwrap();
        advance.store(60 * 60 * 1000, Ordering::SeqCst); // past the hourly throttle
        let _ = spill.truncate(&output, Keep::Head).await.unwrap(); // triggers cleanup

        assert!(!paths[0].exists(), "the aged file expired");
        assert_eq!(
            std::fs::read_to_string(&paths[1]).unwrap(),
            output,
            "fresh files stay"
        );
    }

    #[tokio::test]
    async fn cleanup_runs_at_most_hourly() {
        let dir = tempfile::tempdir().unwrap();
        let now = SystemTime::now();
        let (clock, advance) = fake_clock(now);
        let spill = Arc::new(OutputSpill::new(dir.path()).with_clock(clock));

        let aged = dir.path().join("output-aged.txt");
        std::fs::write(&aged, "x").unwrap();
        let old = now - Duration::from_secs(8 * 24 * 60 * 60);
        filetime::set_file_mtime(&aged, filetime::FileTime::from_system_time(old)).unwrap();

        spill.save("first").await.unwrap(); // runs the cleanup
        assert!(!aged.exists(), "the first save cleaned up");

        let aged2 = dir.path().join("output-aged2.txt");
        std::fs::write(&aged2, "x").unwrap();
        filetime::set_file_mtime(&aged2, filetime::FileTime::from_system_time(old)).unwrap();
        spill.save("second").await.unwrap(); // within the hour: no cleanup
        assert!(aged2.exists(), "cleanup is throttled to hourly");

        advance.store(61 * 60 * 1000, Ordering::SeqCst);
        spill.save("third").await.unwrap();
        assert!(!aged2.exists(), "after the interval cleanup runs again");
    }

    #[tokio::test]
    async fn a_failed_spill_still_returns_the_clipped_output() {
        // The spill "directory" is a regular file: creation must fail.
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, "not a directory").unwrap();
        let spill = test_spill(&blocker);
        let output = "x\n".repeat(3000);
        let rendered = spill.truncate(&output, Keep::Head).await.unwrap();
        assert!(rendered.starts_with("x\nx\n"), "{rendered}");
        assert!(
            rendered.contains("The full output was not saved: cannot create the spill directory"),
            "{}",
            &rendered[rendered.len() - 200..]
        );
        assert!(rendered.len() <= MAX_BYTES);
    }

    #[tokio::test]
    async fn a_full_quota_evicts_settled_outputs_but_never_recent_ones() {
        let dir = tempfile::tempdir().unwrap();
        let spill = OutputSpill::new(dir.path())
            .with_capture_cap(100)
            .with_storage_limit(250);
        let old = spill.save(&"o".repeat(100)).await.unwrap();
        let two_hours_ago =
            filetime::FileTime::from_system_time(SystemTime::now() - Duration::from_secs(2 * 3600));
        filetime::set_file_mtime(&old, two_hours_ago).unwrap();
        let recent = spill.save(&"r".repeat(100)).await.unwrap();
        // 200 bytes used plus a 100-byte reservation exceeds 250: the old one goes.
        let third = spill.save(&"t".repeat(100)).await.unwrap();
        assert!(!old.exists() && recent.exists() && third.exists());
        // Only recent outputs remain, so the quota now holds.
        assert!(spill.save("x").await.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlinked_spill_directory_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let err = test_spill(&link).save("x").await.unwrap_err();
        assert!(err.to_string().contains("symlink"), "{err}");
        assert_eq!(std::fs::read_dir(&real).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spill_files_and_directories_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(&dir.path().join("spill"));
        let path = spill.save("secret").await.unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::metadata(spill.dir()).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    #[tokio::test]
    async fn a_single_long_line_still_shows_a_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());
        let output = "j".repeat(60_000);
        let rendered = spill.truncate(&output, Keep::Head).await.unwrap();
        let first = rendered.lines().next().unwrap();
        assert!(
            first.len() > 40_000 && first.chars().all(|c| c == 'j'),
            "{first}"
        );
        assert!(rendered.contains("lines truncated. Full output saved to:"));
        assert!(rendered.len() <= MAX_BYTES, "{}", rendered.len());
        assert_eq!(
            std::fs::read_to_string(spill_path(&rendered)).unwrap(),
            output
        );

        // The tail view keeps the end of the line, on a char boundary.
        let output = format!("{}END", "é".repeat(40_000));
        let rendered = spill.truncate(&output, Keep::Tail).await.unwrap();
        assert!(
            rendered.ends_with("ééEND"),
            "{}",
            &rendered[rendered.len() - 20..]
        );
        assert!(rendered.len() <= MAX_BYTES);
    }

    #[tokio::test]
    async fn newline_terminated_long_lines_still_show_content() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());

        // Head: a leading blank line must not stand in for the content.
        let output = format!("\n{}", "j".repeat(60_000));
        let rendered = spill.truncate(&output, Keep::Head).await.unwrap();
        assert!(rendered.contains(&"j".repeat(1_000)), "{rendered}");
        assert!(rendered.len() <= MAX_BYTES, "{}", rendered.len());
        assert_eq!(
            std::fs::read_to_string(spill_path(&rendered)).unwrap(),
            output
        );

        // Tail: the trailing newline splits off an empty line that must not
        // stand in for the content either.
        let output = format!("{}END\n", "j".repeat(60_000));
        let rendered = spill.truncate(&output, Keep::Tail).await.unwrap();
        assert!(rendered.contains("jjEND"), "{rendered}");
        assert!(rendered.len() <= MAX_BYTES, "{}", rendered.len());
    }

    #[tokio::test]
    async fn a_streamed_newline_terminated_long_line_still_shows_its_end() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());
        let mut capture = spill.capture();
        let output = format!("{}END\n", "j".repeat(100_000));
        capture.append(output.as_bytes()).await.unwrap();
        let rendered = capture.finish("\n\n[exit code 0]").await.unwrap();
        assert!(
            rendered.text.ends_with("jjEND\n\n[exit code 0]"),
            "{}",
            rendered.text
        );
        assert!(rendered.text.len() <= MAX_BYTES, "{}", rendered.text.len());
        assert!(
            rendered.text.lines().count() <= MAX_LINES,
            "{}",
            rendered.text.lines().count()
        );
    }

    #[tokio::test]
    async fn a_streamed_single_long_line_still_shows_its_end() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());
        let mut capture = spill.capture();
        let output = format!("{}END", "j".repeat(100_000));
        capture.append(output.as_bytes()).await.unwrap();
        let rendered = capture.finish("\n[exit code 0]").await.unwrap();
        assert!(
            rendered.text.ends_with("jjEND\n[exit code 0]"),
            "{}",
            rendered.text
        );
        assert!(rendered.text.len() <= MAX_BYTES, "{}", rendered.text.len());
    }

    #[tokio::test]
    async fn capture_streams_incrementally_and_caps() {
        let dir = tempfile::tempdir().unwrap();
        // Cap above the 50 KiB spill trigger so the file exists when the cap hits.
        let spill = Arc::new(OutputSpill::new(dir.path()).with_capture_cap(60_000));
        let mut capture = spill.capture();
        let chunk = vec![b'x'; 8_192];
        let mut cap_hit = false;
        for _ in 0..10 {
            if capture.append(&chunk).await.unwrap() {
                cap_hit = true;
                break;
            }
        }
        assert!(cap_hit, "80 KiB exceeds the 60 KiB cap");
        let rendered = capture.finish("").await.unwrap();
        assert!(rendered.capped);
        assert!(rendered.text.contains("capture cap"), "{}", rendered.text);
        let path = rendered.path.expect("a spill file");
        let size = std::fs::metadata(&path).unwrap().len();
        // The file holds exactly the captured prefix, with no chunk overshoot.
        assert_eq!(size, 60_000, "prefix size: {size}");
    }

    #[tokio::test]
    async fn the_tail_keeps_sliding_after_the_spill_starts() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());
        let mut capture = spill.capture();
        let chunk = vec![b'x'; 8_192];
        for _ in 0..10 {
            // 80 KiB: crosses the 50 KiB spill trigger partway through.
            capture.append(&chunk).await.unwrap();
        }
        capture.append(b"\nFINAL-MARKER\n").await.unwrap();
        let rendered = capture.finish("").await.unwrap();
        let path = rendered.path.expect("a spill file");
        assert!(
            rendered.text.contains("FINAL-MARKER"),
            "the preview ends with the newest output: {}",
            rendered.text
        );
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(
            saved.ends_with("FINAL-MARKER\n"),
            "the file holds everything"
        );
        assert_eq!(saved.len(), 10 * 8_192 + "\nFINAL-MARKER\n".len());
    }

    #[tokio::test]
    async fn truncate_respects_the_capture_cap() {
        let dir = tempfile::tempdir().unwrap();
        let spill = Arc::new(OutputSpill::new(dir.path()).with_capture_cap(60_000));
        let line = "z".repeat(100);
        let output = (0..2_000)
            .map(|_| line.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(output.len() > 60_000);
        let rendered = spill.truncate(&output, Keep::Tail).await.unwrap();
        assert!(rendered.contains("capture cap"), "{rendered}");
        // The spill file holds only the captured prefix.
        let entry = std::fs::read_dir(dir.path())
            .unwrap()
            .find(|entry| {
                entry
                    .as_ref()
                    .is_ok_and(|entry| entry.file_name().to_string_lossy().starts_with("output-"))
            })
            .expect("one spill file")
            .unwrap();
        let saved = std::fs::read_to_string(entry.path()).unwrap();
        assert_eq!(saved.len(), 60_000, "the cap bounds the spill file");
        assert_eq!(saved, output[..60_000]);
        // The rendered preview — every notice included — stays within the
        // model-visible bound.
        assert!(
            rendered.len() <= MAX_BYTES,
            "the notices are budgeted into the bound: {} > {MAX_BYTES}",
            rendered.len()
        );
        assert!(
            rendered.lines().count() <= MAX_LINES,
            "the line bound holds: {}",
            rendered.lines().count()
        );
    }

    #[tokio::test]
    async fn capped_previews_budget_every_notice() {
        let dir = tempfile::tempdir().unwrap();
        let spill = Arc::new(OutputSpill::new(dir.path()).with_capture_cap(60_000));
        let mut capture = spill.capture();
        let chunk = "lorem ipsum dolor sit amet\n".repeat(300).into_bytes();
        while !capture.append(&chunk).await.unwrap() {}
        let rendered = capture.finish("").await.unwrap();
        assert!(rendered.capped);
        assert!(rendered.text.contains("capture cap"), "{}", rendered.text);
        assert!(
            rendered.text.len() <= MAX_BYTES,
            "the notices are budgeted into the bound: {} > {MAX_BYTES}",
            rendered.text.len()
        );
        assert!(
            rendered.text.lines().count() <= MAX_LINES,
            "the line bound holds: {}",
            rendered.text.lines().count()
        );
    }

    #[tokio::test]
    async fn capped_short_lines_budget_the_notice_line() {
        let dir = tempfile::tempdir().unwrap();
        let spill = OutputSpill::new(dir.path()).with_capture_cap(60_000);
        let output = "x\n".repeat(50_000);
        for keep in [Keep::Head, Keep::Tail] {
            let rendered = spill.truncate(&output, keep).await.unwrap();
            assert!(rendered.contains("capture cap"));
            assert!(rendered.len() <= MAX_BYTES);
            assert!(
                rendered.lines().count() <= MAX_LINES,
                "preview has {} lines",
                rendered.lines().count()
            );
        }
    }

    #[tokio::test]
    async fn streamed_capped_short_lines_budget_the_notice_line() {
        let dir = tempfile::tempdir().unwrap();
        let spill = Arc::new(OutputSpill::new(dir.path()).with_capture_cap(60_000));
        let mut capture = spill.capture();
        let output = format!("{}END", "x\n".repeat(29_999));
        assert!(capture.append(output.as_bytes()).await.unwrap());
        let rendered = capture.finish("").await.unwrap();
        assert!(rendered.capped);
        assert!(rendered.text.contains("capture cap"));
        assert!(rendered.text.contains("E"));
        assert!(rendered.text.len() <= MAX_BYTES);
        assert!(
            rendered.text.lines().count() <= MAX_LINES,
            "preview has {} lines",
            rendered.text.lines().count()
        );
    }

    #[tokio::test]
    async fn capped_passthrough_stays_inside_the_bound() {
        let dir = tempfile::tempdir().unwrap();
        // The prefix plus the cap notice would exceed MAX_BYTES.
        let spill = Arc::new(OutputSpill::new(dir.path()).with_capture_cap(51_190));
        let output = "q".repeat(120_000);
        let rendered = spill.truncate(&output, Keep::Tail).await.unwrap();
        assert!(rendered.contains("capture cap"), "{rendered}");
        assert!(
            rendered.len() <= MAX_BYTES,
            "the notice is budgeted: {} > {MAX_BYTES}",
            rendered.len()
        );
        assert!(rendered.lines().count() <= MAX_LINES);
    }

    #[tokio::test]
    async fn capture_without_spill_renders_the_trimmed_output() {
        let dir = tempfile::tempdir().unwrap();
        let spill = test_spill(dir.path());
        let mut capture = spill.capture();
        capture.append(b"hello\n").await.unwrap();
        capture.append(b"world\n\n").await.unwrap();
        let rendered = capture.finish("").await.unwrap();
        assert_eq!(rendered.text, "hello\nworld");
        assert!(rendered.path.is_none());

        let empty = spill.capture();
        let rendered = empty.finish("").await.unwrap();
        assert_eq!(rendered.text, "(no output)");
    }

    #[test]
    fn clip_line_respects_the_preview_bound() {
        let long = "é".repeat(MAX_LINE_LENGTH + 100);
        let clipped = clip_line(&long);
        assert_eq!(clipped.chars().count(), MAX_LINE_LENGTH + 3);
        assert!(clipped.ends_with("..."));
        assert_eq!(clip_line("short"), "short");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn truncate_bounds_and_spill_fidelity(
            lines in proptest::collection::vec("[^\n]{0,120}", 0..2_600),
            tail in proptest::bool::ANY,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let spill = test_spill(dir.path());
            let output = lines.join("\n");
            let keep = if tail { Keep::Tail } else { Keep::Head };
            let rendered = block_on(spill.truncate(&output, keep)).unwrap();
            prop_assert!(rendered.split('\n').count() <= MAX_LINES);
            prop_assert!(rendered.len() <= MAX_BYTES);
            if rendered == output {
                prop_assert!(output.split('\n').count() <= MAX_LINES && output.len() <= MAX_BYTES);
            } else {
                let path = spill_path(&rendered);
                prop_assert_eq!(std::fs::read_to_string(&path).unwrap(), output);
                if tail {
                    // truncate() keeps lines byte-exact (the bash capture
                    // path trims separately), so the last line is verbatim.
                    prop_assert!(rendered.ends_with(lines.last().map(String::as_str).unwrap_or("")));
                } else {
                    prop_assert!(rendered.starts_with(lines.first().map(String::as_str).unwrap_or("")));
                }
            }
        }

        /// Multibyte content never breaks clipping (char-boundary safety).
        #[test]
        fn truncate_handles_multibyte_lines(
            lines in proptest::collection::vec("\\p{L}{1,80}", 2_100..2_400),
        ) {
            let dir = tempfile::tempdir().unwrap();
            let spill = test_spill(dir.path());
            let output = lines.join("\n");
            let rendered = block_on(spill.truncate(&output, Keep::Tail)).unwrap();
            prop_assert!(rendered.split('\n').count() <= MAX_LINES);
            prop_assert!(rendered.len() <= MAX_BYTES);
            prop_assert!(rendered.ends_with(lines.last().unwrap().as_str()));
        }
    }
}
