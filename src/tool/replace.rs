/*
 * Derived from opencode (packages/opencode/src/tool/edit.ts @ 1ddb087).
 * https://github.com/sst/opencode — MIT License, Copyright (c) 2025 opencode.
 * See the full license notice in LICENSE under "Third-party notices".
 */

//! Ordered exact and fuzzy edit matching. Ambiguous or disproportionately large matches
//! are rejected.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

const SINGLE_CANDIDATE_SIMILARITY_THRESHOLD: f64 = 0.65;
const MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD: f64 = 0.65;

/// Why an edit could not be applied. Rendered to model-visible text so the
/// model can fix its call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplaceError {
    /// The search and replacement are the same string.
    #[error("no changes to apply: oldString and newString are identical")]
    Identical,
    /// An empty search only makes sense when creating a new file.
    #[error(
        "oldString cannot be empty when editing an existing file; provide the exact text to replace, or use write for an intentional full-file replacement"
    )]
    Empty,
    /// A fuzzy candidate spanned far more text than the search.
    #[error(
        "refusing the replacement because the matched span is much larger than oldString; re-read the file and provide the full exact oldString for the intended replacement"
    )]
    Disproportionate,
    /// Nothing matched, exactly or fuzzily.
    #[error(
        "could not find oldString in the file; it must match exactly, including whitespace, indentation and line endings"
    )]
    NotFound,
    /// Several spans matched; the caller must add context.
    #[error(
        "found multiple matches for oldString; provide more surrounding context to make the match unique"
    )]
    Ambiguous,
}

/// Replaces `old` with `new` in `content`, falling back through the matcher
/// chain when there is no exact match. The first matcher that finds anything
/// decides: without `replace_all` it must find a single span occurring once,
/// since a fuzzier matcher would only guess between the occurrences.
pub fn replace(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<String, ReplaceError> {
    if old == new {
        return Err(ReplaceError::Identical);
    }
    if old.is_empty() {
        return Err(ReplaceError::Empty);
    }
    for matcher in MATCHERS {
        // Repeated-line files yield the same candidate thousands of times; check
        // each distinct span once.
        let mut seen: HashSet<String> = HashSet::new();
        let mut found: Option<(String, usize)> = None;
        for candidate in matcher(content, old) {
            if candidate.is_empty() || !seen.insert(candidate.clone()) {
                continue;
            }
            let Some(index) = content.find(&candidate) else {
                continue;
            };
            if is_disproportionate(&candidate, old) {
                return Err(ReplaceError::Disproportionate);
            }
            if replace_all {
                return Ok(content.replace(&candidate, new));
            }
            if found.is_some() || content.rfind(&candidate) != Some(index) {
                return Err(ReplaceError::Ambiguous);
            }
            found = Some((candidate, index));
        }
        if let Some((candidate, index)) = found {
            let mut out = String::with_capacity(content.len() + new.len());
            out.push_str(&content[..index]);
            out.push_str(new);
            out.push_str(&content[index + candidate.len()..]);
            return Ok(out);
        }
    }
    Err(ReplaceError::NotFound)
}

type Matcher = fn(content: &str, find: &str) -> Vec<String>;

/// Try matchers from exact to progressively fuzzier.
const MATCHERS: [Matcher; 8] = [
    simple,
    line_trimmed,
    block_anchor,
    whitespace_normalized,
    indentation_flexible,
    escape_normalized,
    trimmed_boundary,
    context_aware,
];

fn simple(_content: &str, find: &str) -> Vec<String> {
    vec![find.to_owned()]
}

/// Matches a block whose lines equal the search lines after trimming, and
/// returns the span with the file's original indentation.
fn line_trimmed(content: &str, find: &str) -> Vec<String> {
    let original: Vec<&str> = content.split('\n').collect();
    let (search, newline) = drop_trailing_empty(find.split('\n').collect::<Vec<_>>());
    let mut out = Vec::new();
    if search.is_empty() || original.len() < search.len() {
        return out;
    }
    let offsets = line_offsets(&original);
    for i in 0..=original.len() - search.len() {
        if (0..search.len()).all(|j| original[i + j].trim() == search[j].trim()) {
            out.push(span(&original, &offsets, i, search.len(), newline, content));
        }
    }
    out
}

/// Matches blocks whose first and last trimmed lines anchor the search and
/// whose middle is similar enough (Levenshtein), tolerating a line-count
/// drift of up to 25 %.
fn block_anchor(content: &str, find: &str) -> Vec<String> {
    let original: Vec<&str> = content.split('\n').collect();
    let (search, newline) = drop_trailing_empty(find.split('\n').collect::<Vec<_>>());
    if search.len() < 3 {
        return Vec::new();
    }
    let first = search[0].trim();
    let last = search[search.len() - 1].trim();
    let search_size = search.len();
    let max_line_delta = (search_size / 4).max(1);

    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for (i, line) in original.iter().enumerate() {
        if line.trim() != first {
            continue;
        }
        // Only the first occurrence of the last line after this first line.
        if let Some(j) = (i + 2..original.len()).find(|&j| original[j].trim() == last) {
            let block_size = j - i + 1;
            if block_size.abs_diff(search_size) <= max_line_delta {
                candidates.push((i, j));
            }
        }
    }
    if candidates.is_empty() {
        return Vec::new();
    }
    let similarity_of = |start: usize, end: usize| -> f64 {
        let actual_size = end - start + 1;
        let lines_to_check = (search_size.saturating_sub(2)).min(actual_size.saturating_sub(2));
        if lines_to_check == 0 {
            // No middle lines to compare: accept based on the anchors.
            return 1.0;
        }
        let mut similarity = 0.0;
        let mut j = 1;
        while j + 1 < search_size && start + j < end {
            let original_line = original[start + j].trim();
            let search_line = search[j].trim();
            let max_len = original_line
                .chars()
                .count()
                .max(search_line.chars().count());
            if max_len > 0 {
                let distance = levenshtein(original_line, search_line);
                similarity += 1.0 - distance as f64 / max_len as f64;
            }
            j += 1;
        }
        // All-empty middle lines score zero similarity.
        similarity / lines_to_check as f64
    };

    if candidates.len() == 1 {
        let (start, end) = candidates[0];
        let mut similarity = 0.0;
        let actual_size = end - start + 1;
        let lines_to_check = (search_size.saturating_sub(2)).min(actual_size.saturating_sub(2));
        if lines_to_check == 0 {
            similarity = 1.0;
        } else {
            let mut j = 1;
            while j + 1 < search_size && start + j < end {
                let original_line = original[start + j].trim();
                let search_line = search[j].trim();
                let max_len = original_line
                    .chars()
                    .count()
                    .max(search_line.chars().count());
                if max_len > 0 {
                    let distance = levenshtein(original_line, search_line);
                    similarity += (1.0 - distance as f64 / max_len as f64) / lines_to_check as f64;
                    if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD {
                        break;
                    }
                }
                j += 1;
            }
        }
        if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD {
            let offsets = line_offsets(&original);
            return vec![span(
                &original,
                &offsets,
                start,
                end - start + 1,
                newline,
                content,
            )];
        }
        return Vec::new();
    }

    let mut best: Option<(usize, usize)> = None;
    let mut max_similarity = -1.0;
    for &(start, end) in &candidates {
        let similarity = similarity_of(start, end);
        if similarity > max_similarity {
            max_similarity = similarity;
            best = Some((start, end));
        }
    }
    match best {
        Some((start, end)) if max_similarity >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD => {
            let offsets = line_offsets(&original);
            vec![span(
                &original,
                &offsets,
                start,
                end - start + 1,
                newline,
                content,
            )]
        }
        _ => Vec::new(),
    }
}

/// Matches single lines that contain the whitespace-normalized search, and
/// multi-line blocks that equal it after normalization.
fn whitespace_normalized(content: &str, find: &str) -> Vec<String> {
    let normalized_find = normalize_whitespace(find);
    let mut out = Vec::new();
    let lines: Vec<&str> = content.split('\n').collect();
    // Depends only on the search: compiled once, not per matching line.
    let words: Vec<String> = find.split_whitespace().map(regex::escape).collect();
    let within_line = (!words.is_empty())
        .then(|| Regex::new(&words.join(r"\s+")).ok())
        .flatten();
    for line in &lines {
        let normalized = normalize_whitespace(line);
        if normalized == normalized_find {
            out.push((*line).to_owned());
        } else if normalized.contains(&normalized_find)
            && let Some(re) = &within_line
            && let Some(hit) = re.find(line)
        {
            out.push(hit.as_str().to_owned());
        }
    }
    let find_lines: Vec<&str> = find.split('\n').collect();
    if find_lines.len() > 1 && find_lines.len() <= lines.len() {
        for i in 0..=lines.len() - find_lines.len() {
            let block = lines[i..i + find_lines.len()].join("\n");
            if normalize_whitespace(&block) == normalized_find {
                out.push(block);
            }
        }
    }
    out
}

/// Matches blocks that equal the search after removing their common
/// indentation.
fn indentation_flexible(content: &str, find: &str) -> Vec<String> {
    let normalized_find = remove_indentation(find);
    let content_lines: Vec<&str> = content.split('\n').collect();
    let find_lines: Vec<&str> = find.split('\n').collect();
    let mut out = Vec::new();
    if find_lines.is_empty() || content_lines.len() < find_lines.len() {
        return out;
    }
    for i in 0..=content_lines.len() - find_lines.len() {
        let block = content_lines[i..i + find_lines.len()].join("\n");
        if remove_indentation(&block) == normalized_find {
            out.push(block);
        }
    }
    out
}

/// Matches after unescaping common string escapes (`\n`, `\t`, quotes,
/// backslashes, line continuations, `\$`) on either side.
fn escape_normalized(content: &str, find: &str) -> Vec<String> {
    let unescaped_find = unescape(find);
    let mut out = Vec::new();
    if content.contains(&unescaped_find) {
        out.push(unescaped_find.clone());
    }
    let lines: Vec<&str> = content.split('\n').collect();
    let find_lines = unescaped_find.split('\n').count();
    if find_lines <= lines.len() {
        for i in 0..=lines.len() - find_lines {
            let block = lines[i..i + find_lines].join("\n");
            if unescape(&block) == unescaped_find {
                out.push(block);
            }
        }
    }
    out
}

/// Matches the trimmed search, or blocks whose trimmed form equals it.
fn trimmed_boundary(content: &str, find: &str) -> Vec<String> {
    let trimmed = find.trim();
    if trimmed == find {
        // Already trimmed: `simple` covered it.
        return Vec::new();
    }
    let mut out = Vec::new();
    if content.contains(trimmed) {
        out.push(trimmed.to_owned());
    }
    let lines: Vec<&str> = content.split('\n').collect();
    let find_lines: Vec<&str> = find.split('\n').collect();
    if lines.len() < find_lines.len() {
        return out;
    }
    for i in 0..=lines.len() - find_lines.len() {
        let block = lines[i..i + find_lines.len()].join("\n");
        if block.trim() == trimmed {
            out.push(block);
        }
    }
    out
}

/// Matches blocks anchored by the search's first and last lines where at
/// least half of the middle lines agree.
fn context_aware(content: &str, find: &str) -> Vec<String> {
    let find_lines = find.split('\n').collect::<Vec<_>>();
    if find_lines.len() < 3 {
        return Vec::new();
    }
    let (find_lines, newline) = drop_trailing_empty(find_lines);
    let content_lines: Vec<&str> = content.split('\n').collect();
    let first = find_lines[0].trim();
    let last = find_lines[find_lines.len() - 1].trim();
    let mut out = Vec::new();
    for (i, line) in content_lines.iter().enumerate() {
        if line.trim() != first {
            continue;
        }
        // Only the first occurrence of the last line after this first line.
        let Some(j) = (i + 2..content_lines.len()).find(|&j| content_lines[j].trim() == last)
        else {
            continue;
        };
        let block_lines = &content_lines[i..=j];
        if block_lines.len() == find_lines.len() {
            let mut matching = 0usize;
            let mut total_non_empty = 0usize;
            for k in 1..block_lines.len() - 1 {
                let block_line = block_lines[k].trim();
                let find_line = find_lines[k].trim();
                if !block_line.is_empty() || !find_line.is_empty() {
                    total_non_empty += 1;
                    if block_line == find_line {
                        matching += 1;
                    }
                }
            }
            if total_non_empty == 0 || matching * 2 >= total_non_empty {
                let mut block = block_lines.join("\n");
                if newline && j + 1 < content_lines.len() {
                    block.push('\n');
                }
                out.push(block);
                // Continue scanning later anchors; replacement still requires a unique
                // span.
            }
        }
    }
    out
}

/// The vendored guard against fuzzy matches that swallow most of the file.
fn is_disproportionate(search: &str, old: &str) -> bool {
    let old_lines = old.split('\n').count();
    let search_lines = search.split('\n').count();
    if search_lines >= (old_lines + 3).max(old_lines * 2) {
        return true;
    }
    if old_lines == 1 {
        return false;
    }
    let old_len = old.trim().chars().count();
    search.trim().chars().count() > (old_len + 500).max(old_len * 4)
}

/// Drops the empty line a trailing newline splits off, reporting whether the
/// search ended with one so the matched span can keep its own newline.
fn drop_trailing_empty(mut lines: Vec<&str>) -> (Vec<&str>, bool) {
    let newline = lines.last() == Some(&"");
    if newline {
        lines.pop();
    }
    (lines, newline)
}

/// Byte offsets where each line of the `\n`-split content starts, so spans cost
/// O(1) instead of a rescan from the top per candidate.
fn line_offsets(original: &[&str]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(original.len());
    let mut offset = 0;
    for line in original {
        offsets.push(offset);
        offset += line.len() + 1; // the newline
    }
    offsets
}

/// The content span of `len` (at least one) lines starting at line `start`,
/// including the last line's newline when `newline` is set and the file has one.
fn span(
    original: &[&str],
    offsets: &[usize],
    start: usize,
    len: usize,
    newline: bool,
    content: &str,
) -> String {
    let last = start + len - 1;
    let byte_start = offsets[start];
    let byte_end = offsets[last] + original[last].len() + usize::from(newline);
    content[byte_start.min(content.len())..byte_end.min(content.len())].to_owned()
}

fn normalize_whitespace(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\s+").expect(r"\s+ compiles"));
    re.replace_all(text, " ").trim().to_owned()
}

fn remove_indentation(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let indents: Vec<usize> = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.chars().take_while(|c| c.is_whitespace()).count())
        .collect();
    let Some(&min_indent) = indents.iter().min() else {
        return text.to_owned();
    };
    lines
        .iter()
        .map(|line| {
            if line.trim().is_empty() {
                (*line).to_owned()
            } else {
                let byte_indent = line
                    .char_indices()
                    .nth(min_indent)
                    .map_or(line.len(), |(offset, _)| offset);
                line[byte_indent..].to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn unescape(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re =
        RE.get_or_init(|| Regex::new(r#"\\([ntr'"`\n\\$])"#).expect("the escape pattern compiles"));
    re.replace_all(text, |caps: &regex::Captures| -> String {
        match &caps[1] {
            "n" => "\n".to_owned(),
            "t" => "\t".to_owned(),
            "r" => "\r".to_owned(),
            "'" => "'".to_owned(),
            "\"" => "\"".to_owned(),
            "`" => "`".to_owned(),
            "\\" => "\\".to_owned(),
            "\n" => "\n".to_owned(),
            "$" => "$".to_owned(),
            other => other.to_owned(),
        }
    })
    .into_owned()
}

/// Levenshtein distance over Unicode scalar values.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() || b.is_empty() {
        return a.len().max(b.len());
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate().map(|(i, c)| (i + 1, c)) {
        current[0] = i;
        for (j, &cb) in b.iter().enumerate().map(|(j, c)| (j + 1, c)) {
            let cost = usize::from(ca != cb);
            current[j] = (previous[j] + 1)
                .min(current[j - 1] + 1)
                .min(previous[j - 1] + cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn exact_unique_matches_replace_once() {
        let content = "alpha\nbeta\ngamma\n";
        assert_eq!(
            replace(content, "beta", "BETA", false).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
    }

    #[test]
    fn exact_multiple_matches_are_ambiguous_unless_replace_all() {
        let content = "a\na\n";
        assert_eq!(
            replace(content, "a", "b", false),
            Err(ReplaceError::Ambiguous)
        );
        assert_eq!(replace(content, "a", "b", true).unwrap(), "b\nb\n");
    }

    #[test]
    fn ambiguous_matches_do_not_fall_through_to_fuzzier_matchers() {
        // The exact search occurs twice; a fuzzy matcher could pick one by its indentation.
        let content = "fn a() {\n    foo();\n}\nfn b() {\n        foo();\n}\n";
        assert_eq!(
            replace(content, "foo();", "bar();", false),
            Err(ReplaceError::Ambiguous)
        );
        // Two distinct trimmed-line spans, each unique on its own.
        let content = "\tx = 1;\n  x = 1;\n";
        assert_eq!(
            replace(content, "x = 1; ", "x = 2;", false),
            Err(ReplaceError::Ambiguous)
        );
    }

    #[test]
    fn identical_and_empty_searches_are_rejected() {
        assert_eq!(replace("x", "a", "a", false), Err(ReplaceError::Identical));
        assert_eq!(replace("x", "", "a", false), Err(ReplaceError::Empty));
    }

    #[test]
    fn missing_search_is_not_found() {
        assert_eq!(
            replace("alpha\n", "beta", "x", false),
            Err(ReplaceError::NotFound)
        );
    }

    #[test]
    fn line_trimmed_matching_keeps_file_indentation() {
        // Different indentation exercises the trimmed-line matcher.
        let content = "if (x) {\n    return 1;\n}\nreturn 1;\n";
        let out = replace(
            content,
            "if (x) {\nreturn 1;\n}",
            "if (x) {\n    return 2;\n}",
            false,
        )
        .unwrap();
        assert_eq!(out, "if (x) {\n    return 2;\n}\nreturn 1;\n");
    }

    #[test]
    fn block_anchor_matches_a_single_similar_candidate() {
        let content = "fn main() {\n    let x = 1;\n    let y = 2;\n    println!(\"{x}{y}\");\n}\n";
        let find = "fn main() {\n    let x = 1;\n    let z = 3;\n    println!(\"{x}{y}\");\n}\n";
        let out = replace(content, find, "REPLACED\n", false).unwrap();
        assert_eq!(out, "REPLACED\n");
    }

    #[test]
    fn a_search_longer_than_the_file_is_not_found() {
        assert_eq!(
            replace("one line", "p\nq", "q", false),
            Err(ReplaceError::NotFound)
        );
        assert_eq!(
            replace("a\nb", "a\nb\nc\nd", "x", false),
            Err(ReplaceError::NotFound)
        );
        assert_eq!(
            replace("  a", "a\nb\nc\n", "x", false),
            Err(ReplaceError::NotFound)
        );
    }

    #[test]
    fn fuzzy_matches_keep_the_searchs_trailing_newline() {
        // Indentation differs, so only the line-trimmed matcher hits; its span
        // must end after the newline the search ended with.
        assert_eq!(
            replace("x\nfoo\ny\n", "  foo\n", "bar\n", false).unwrap(),
            "x\nbar\ny\n"
        );
        assert_eq!(
            replace(
                "a\n    foo();\n    baz();\nb\n",
                "foo();\nbaz();\n",
                "qux();\n",
                false
            )
            .unwrap(),
            "a\nqux();\nb\n"
        );
        // Without a trailing newline in the search the span stops before it.
        assert_eq!(
            replace("x\nfoo\ny\n", "  foo", "bar", false).unwrap(),
            "x\nbar\ny\n"
        );
        // A file ending without a newline still matches a search that has one.
        assert_eq!(
            replace("x\n  foo", "foo\n", "bar\n", false).unwrap(),
            "x\nbar\n"
        );
        // The context-aware matcher keeps it too.
        assert_eq!(
            replace(
                "header\nsame a\nsame b\nfooter\ntail\n",
                "header\nsame a\nDIFFERENT\nfooter\n",
                "block!\n",
                false
            )
            .unwrap(),
            "block!\ntail\n"
        );
    }

    #[test]
    fn block_anchor_picks_the_most_similar_candidate() {
        let content = concat!("start\naaa bbb ccc\nend\ntail\n", "start\nxxx 111\nend\n",);
        let find = "start\naaa bbb ccc\nend";
        // Two anchored candidates; the first block is an exact middle match.
        let out = replace(content, find, "<<X>>", false).unwrap();
        assert!(out.starts_with("<<X>>\ntail\n"), "{out}");
    }

    #[test]
    fn whitespace_normalized_substring_matching() {
        let content = "let   value = compute( a , b );\n";
        let out = replace(content, "compute( a , b )", "compute(a, b)", false).unwrap();
        assert_eq!(out, "let   value = compute(a, b);\n");
    }

    #[test]
    fn indentation_flexible_matching() {
        let content = "if (a) {\n        doIt();\n    }\n";
        let out = replace(content, "if (a) {\n    doIt();\n}", "done();", false).unwrap();
        assert_eq!(out, "done();\n");
    }

    #[test]
    fn escape_normalized_matching() {
        let content = "println!(\"line\\nbreak\");\n";
        let out = replace(content, "println!(\"line\\nbreak\");", "ok();", false).unwrap();
        assert_eq!(out, "ok();\n");
        let content = "a\tb\n";
        let out = replace(content, "a\\tb", "tab", false).unwrap();
        assert_eq!(out, "tab\n");
    }

    #[test]
    fn trimmed_boundary_matching() {
        let content = "middle of the file\nTARGET LINE\nend\n";
        let out = replace(content, "\nTARGET LINE\n", "\nREPLACED\n", false).unwrap();
        assert_eq!(out, "middle of the file\nREPLACED\nend\n");
    }

    #[test]
    fn context_aware_matching_requires_half_the_middle_lines() {
        let content = "header\nsame a\nsame b\nfooter\n";
        let find = "header\nsame a\nDIFFERENT\nfooter";
        let out = replace(content, find, "block!", false).unwrap();
        assert_eq!(out, "block!\n");
        let find = "header\nX\nY\nfooter";
        assert_eq!(
            replace(content, find, "block!", false),
            Err(ReplaceError::NotFound)
        );
    }

    #[test]
    fn disproportionate_fuzzy_matches_are_refused() {
        // A whitespace-normalized multi-line block whose span dwarfs the
        // search: 4x+500 trimmed characters is the vendored refusal rule.
        let pad = " ".repeat(2_000);
        let content = format!("a{pad}b\nc{pad}d\n");
        assert_eq!(
            replace(&content, "a b\nc d", "q", false),
            Err(ReplaceError::Disproportionate)
        );
    }

    #[test]
    fn repeated_line_files_stay_fast_when_ambiguous() {
        // Every line is a candidate for every matcher; the scan must stay linear.
        let content = "    return 1;\n".repeat(200_000);
        let started = std::time::Instant::now();
        assert_eq!(
            replace(&content, "return 1;", "return 2;", false),
            Err(ReplaceError::Ambiguous)
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn repeated_line_files_stay_fast_when_the_search_is_a_substring() {
        // Every line contains the search: the whitespace matcher's regex is
        // compiled once, not once per line.
        let content = "    return 1; // note\n".repeat(200_000);
        let started = std::time::Instant::now();
        assert_eq!(
            replace(&content, "return 1;", "return 2;", false),
            Err(ReplaceError::Ambiguous)
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("same", "same"), 0);
    }

    proptest! {
        #[test]
        fn unique_exact_replacement_splices(
            prefix in "[a-z0-9 ]{0,20}",
            old in "[a-z0-9 ]{1,20}",
            suffix in "[a-z0-9 ]{0,20}",
            new in "[a-z0-9 ]{1,20}",
        ) {
            // Sentinels keep `old` unique in the content and out of the noise.
            prop_assume!(old != new);
            let content = format!("{prefix}\u{1}{old}\u{2}{suffix}");
            // The random noise around the sentinels could still contain `old`.
            prop_assume!(content.matches(&old).count() == 1);
            let out = replace(&content, &old, &new, false).unwrap();
            prop_assert_eq!(out, content.replacen(&old, &new, 1));
        }

        /// replace_all swaps every occurrence in one pass.
        #[test]
        fn replace_all_swaps_every_occurrence(
            old in "[a-z0-9 ]{1,10}",
            new in "[a-z0-9 ]{1,10}",
            noise in "[a-z0-9 ]{0,10}",
        ) {
            prop_assume!(old != new);
            let content = format!("{old}\u{1}{noise}\u{1}{old}");
            let out = replace(&content, &old, &new, true).unwrap();
            prop_assert_eq!(out, content.replace(&old, &new));
        }

        /// Ambiguous candidates are never selected arbitrarily: an exact
        /// search occurring twice always errors.
        #[test]
        fn ambiguous_exact_matches_never_replace(
            old in "[a-z0-9 ]{1,10}",
            new in "[a-z0-9 ]{1,10}",
        ) {
            prop_assume!(old != new);
            let content = format!("{old}\u{1}{old}");
            prop_assert_eq!(replace(&content, &old, &new, false), Err(ReplaceError::Ambiguous));
        }
    }
}
