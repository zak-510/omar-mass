//! MASS agent protocol: roles, message envelope, and answer parsing.
//! One shared envelope and parser, so any block's output can feed another's.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Agent roles from the paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Predictor,
    Aggregator,
    Reflector,
    Refiner,
    Debator,
    Summarizer,
    Executor,
}

impl Role {
    /// Short token used in agent/session names and file names.
    pub fn token(&self) -> &'static str {
        match self {
            Role::Predictor => "pred",
            Role::Aggregator => "agg",
            Role::Reflector => "refl",
            Role::Refiner => "refine",
            Role::Debator => "deb",
            Role::Summarizer => "sum",
            Role::Executor => "exec",
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.token())
    }
}

/// Request envelope written to `<run_dir>/inbox/<id>.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub id: String,
    pub sender: String,
    pub receiver: String,
    pub timestamp_ns: u64,
    pub run_id: String,
    pub task_id: String,
    pub role: Role,
    pub round: usize,
    /// Fully rendered, self-contained role prompt.
    pub payload: String,
    /// Absolute path the agent must write its reply to.
    pub reply_path: String,
}

/// Reply parsed from the agent's reply file. Parsing is lenient: a
/// bare-text file is accepted as `content` so a quoting slip doesn't lose
/// the inference.
#[derive(Debug, Clone, Deserialize)]
pub struct Reply {
    /// Echo of the request id; not used for routing.
    #[serde(default)]
    #[allow(dead_code)]
    pub id: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub sender: String,
    pub content: String,
}

impl Reply {
    pub fn parse(raw: &str) -> Reply {
        if let Ok(reply) = serde_json::from_str::<Reply>(raw) {
            if !reply.content.trim().is_empty() {
                return reply;
            }
        }
        // Some weak backends wrap the reply object in a JSON array
        // (`[{...}]`); unwrap to the first non-empty element.
        if let Ok(arr) = serde_json::from_str::<Vec<Reply>>(raw) {
            if let Some(reply) = arr.into_iter().find(|r| !r.content.trim().is_empty()) {
                return reply;
            }
        }
        Reply {
            id: String::new(),
            sender: String::new(),
            content: raw.to_string(),
        }
    }
}

/// Extract the final answer from a role output. Prefers the last
/// `<answer>...</answer>` pair; falls back to the last `Answer:` line.
pub fn parse_answer(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let mut last: Option<(usize, usize)> = None;
    let mut from = 0;
    while let Some(open_rel) = lower[from..].find("<answer>") {
        let start = from + open_rel + "<answer>".len();
        match lower[start..].find("</answer>") {
            Some(close_rel) => {
                last = Some((start, start + close_rel));
                from = start + close_rel + "</answer>".len();
            }
            None => break,
        }
    }
    if let Some((start, end)) = last {
        let answer = text[start..end].trim();
        if !answer.is_empty() {
            return Some(answer.to_string());
        }
    }

    // Some models emit \answer{...} instead of the requested <answer> tags.
    if let Some(a) = extract_last_braced(text, "\\answer{") {
        return Some(a);
    }

    // Fallback: last "Answer:" line (DSPy output field).
    let mut fallback = None;
    for line in text.lines() {
        let trimmed = line.trim().trim_start_matches(['*', '#', ' ']);
        if let Some(rest) = strip_prefix_ci(trimmed, "answer:") {
            let rest = rest.trim();
            if !rest.is_empty() {
                fallback = Some(rest.to_string());
            }
        }
    }
    if fallback.is_some() {
        return fallback;
    }

    // Last resort: a `\boxed{...}` expression. Weaker models often skip the
    // requested tags/field but still box the final answer in LaTeX.
    extract_last_boxed(text)
}

/// Content of the last brace-balanced `\boxed{...}`, if any.
fn extract_last_boxed(text: &str) -> Option<String> {
    extract_last_braced(text, "\\boxed{")
}

/// Content of the last brace-balanced `<tag>...}` occurrence, where `tag`
/// ends in the opening `{` (e.g. `"\\boxed{"`, `"\\answer{"`). Handles nested
/// braces (e.g. `\boxed{\frac{1}{2}}`); UTF-8 safe since only ASCII `{`/`}`
/// are matched.
fn extract_last_braced(text: &str, tag: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut result = None;
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find(tag) {
        let open = search_from + rel + tag.len();
        let mut depth = 1usize;
        let mut i = open;
        while i < bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        if depth != 0 {
            break; // unbalanced; give up
        }
        let inner = text[open..i].trim();
        if !inner.is_empty() {
            result = Some(inner.to_string());
        }
        search_from = i + 1;
    }
    result
}

/// Reflector output: a correctness verdict plus free-form feedback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reflection {
    pub correct: bool,
    pub feedback: String,
}

/// Parse a reflector output. `correct` is true when the last
/// `Correctness:` line contains "true". Feedback is the last `Feedback:`
/// section, or the whole text if absent.
pub fn parse_reflection(text: &str) -> Reflection {
    let mut correct = false;
    let mut feedback_start: Option<usize> = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim().trim_start_matches(['*', '#', ' ']);
        if let Some(rest) = strip_prefix_ci(trimmed, "correctness:") {
            correct = rest.to_lowercase().contains("true");
        }
        if strip_prefix_ci(trimmed, "feedback:").is_some() {
            feedback_start = Some(offset);
        }
        offset += line.len();
    }
    let feedback = match feedback_start {
        Some(start) => {
            let section = &text[start..];
            // Cut a trailing Correctness line out of the feedback.
            match find_ci(section, "correctness:") {
                Some(pos) => section[..pos].trim().to_string(),
                None => section.trim().to_string(),
            }
        }
        None => text.trim().to_string(),
    };
    Reflection { correct, feedback }
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack.to_lowercase().find(&needle.to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_answer_takes_last_tag_pair() {
        let text = "draft <answer>3</answer> revised <answer>64</answer> done";
        assert_eq!(parse_answer(text).as_deref(), Some("64"));
    }

    #[test]
    fn parse_answer_handles_multiline_and_case() {
        let text = "Reasoning...\n<ANSWER>\n\\frac{1}{2}\n</ANSWER>";
        assert_eq!(parse_answer(text).as_deref(), Some("\\frac{1}{2}"));
    }

    #[test]
    fn parse_answer_falls_back_to_answer_line() {
        let text = "Reasoning: We compute.\nAnswer: 42\nMore prose";
        assert_eq!(parse_answer(text).as_deref(), Some("42"));
    }

    #[test]
    fn parse_answer_prefers_tags_over_field() {
        let text = "Answer: 1\n<answer>2</answer>";
        assert_eq!(parse_answer(text).as_deref(), Some("2"));
    }

    #[test]
    fn parse_answer_none_when_empty() {
        assert_eq!(parse_answer("no answer here"), None);
        assert_eq!(parse_answer("<answer> </answer>"), None);
    }

    #[test]
    fn parse_answer_falls_back_to_boxed() {
        // No tags, no "Answer:" line, so the boxed value is the last resort.
        let text = "We minimize and find \\boxed{17} at the end.";
        assert_eq!(parse_answer(text).as_deref(), Some("17"));
        // Nested braces handled; the last box wins.
        let nested = "first \\boxed{1} then \\boxed{\\frac{1}{2}}";
        assert_eq!(parse_answer(nested).as_deref(), Some("\\frac{1}{2}"));
    }

    #[test]
    fn parse_answer_recognizes_latex_answer_command() {
        // Weak models sometimes emit \answer{...} instead of <answer></answer>.
        let text = "Thus the smallest real number is $\\answer{5.5}$.";
        assert_eq!(parse_answer(text).as_deref(), Some("5.5"));
        // Even embedded in an array-wrapped reply blob.
        let blob = r#"[{"id":"x","content":"... is $\answer{5.5}$."}]"#;
        assert_eq!(parse_answer(blob).as_deref(), Some("5.5"));
    }

    #[test]
    fn reply_unwraps_single_element_array() {
        let raw = r#"[{"id":"x","sender":"a","content":"hello <answer>9</answer>"}]"#;
        let reply = Reply::parse(raw);
        assert_eq!(reply.content, "hello <answer>9</answer>");
        assert_eq!(parse_answer(&reply.content).as_deref(), Some("9"));
    }

    #[test]
    fn parse_answer_prefers_tags_and_field_over_boxed() {
        assert_eq!(
            parse_answer("Answer: 7\nwork: \\boxed{9}").as_deref(),
            Some("7")
        );
        assert_eq!(
            parse_answer("<answer>7</answer>\n\\boxed{9}").as_deref(),
            Some("7")
        );
    }

    #[test]
    fn parse_reflection_true_verdict() {
        let text = "Reasoning: looks right.\nFeedback: clean derivation.\nCorrectness: True";
        let r = parse_reflection(text);
        assert!(r.correct);
        assert_eq!(r.feedback, "Feedback: clean derivation.");
    }

    #[test]
    fn parse_reflection_false_by_default() {
        let r = parse_reflection("Feedback: sign error in step 2.\nCorrectness: False");
        assert!(!r.correct);
        assert!(r.feedback.contains("sign error"));
    }

    #[test]
    fn parse_reflection_without_fields_uses_whole_text() {
        let r = parse_reflection("the answer is wrong because ...");
        assert!(!r.correct);
        assert_eq!(r.feedback, "the answer is wrong because ...");
    }

    #[test]
    fn reply_parses_json_and_bare_text() {
        let json = r#"{"id":"x","sender":"pred-1","content":"<answer>5</answer>"}"#;
        let reply = Reply::parse(json);
        assert_eq!(reply.sender, "pred-1");
        assert_eq!(reply.content, "<answer>5</answer>");

        let bare = Reply::parse("plain text <answer>7</answer>");
        assert_eq!(bare.content, "plain text <answer>7</answer>");
    }

    #[test]
    fn envelope_round_trips() {
        let env = Envelope {
            id: "e1".into(),
            sender: "runner".into(),
            receiver: "mass-ab12-pred-1".into(),
            timestamp_ns: 42,
            run_id: "ab12".into(),
            task_id: "math-001".into(),
            role: Role::Predictor,
            round: 0,
            payload: "Question: 1+1?".into(),
            reply_path: "/tmp/x.json".into(),
        };
        let json = serde_json::to_string(&env).unwrap();
        let back: Envelope = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, Role::Predictor);
        assert_eq!(back.receiver, env.receiver);
    }
}
