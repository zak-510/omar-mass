//! MATH answer normalization, equivalence, and rule-based majority vote.

use std::collections::HashMap;

/// Normalize a MATH answer for comparison: strip LaTeX wrappers, spacing,
/// fraction notation, thousands separators.
pub fn normalize_answer(answer: &str) -> String {
    let mut s = answer.trim().to_string();

    // Strip surrounding $...$ or \( ... \).
    loop {
        let t = s.trim();
        let stripped = t
            .strip_prefix("$$")
            .and_then(|x| x.strip_suffix("$$"))
            .or_else(|| t.strip_prefix('$').and_then(|x| x.strip_suffix('$')))
            .or_else(|| t.strip_prefix("\\(").and_then(|x| x.strip_suffix("\\)")));
        match stripped {
            Some(inner) => s = inner.to_string(),
            None => break,
        }
    }

    if let Some(inner) = unwrap_command(&s, "\\boxed") {
        s = inner;
    }
    while let Some(inner) = rewrite_first_command(&s, "\\text") {
        s = inner;
    }

    for (from, to) in [
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\ ", ""),
        ("\\$", ""),
        ("\\%", ""),
        ("%", ""),
        ("^{\\circ}", ""),
        ("^\\circ", ""),
        ("\\cdot", "*"),
    ] {
        s = s.replace(from, to);
    }

    // \frac{a}{b} -> (a)/(b).
    while let Some(rewritten) = rewrite_frac(&s) {
        s = rewritten;
    }

    s.retain(|c| !c.is_whitespace() && c != '~');

    while let Some(t) = s.strip_suffix(['.', ',', ';']) {
        s = t.to_string();
    }

    s = remove_thousands_commas(&s);

    s.to_lowercase()
}

/// Strip a single-variable assignment prefix: "x=5" -> "5".
fn strip_assignment(s: &str) -> &str {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(v), Some('=')) if v.is_ascii_alphabetic() => chars.as_str(),
        _ => s,
    }
}

/// True when two MATH answers are equal: normalized string match, with a
/// numeric fallback.
pub fn answers_equal(a: &str, b: &str) -> bool {
    let na = normalize_answer(a);
    let nb = normalize_answer(b);
    if na == nb || strip_assignment(&na) == strip_assignment(&nb) {
        return true;
    }
    match (
        parse_number(strip_assignment(&na)),
        parse_number(strip_assignment(&nb)),
    ) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
        _ => false,
    }
}

/// Majority vote over candidates (self-consistency). Returns the winning
/// bucket's original string, ties broken toward the earliest answer.
pub fn majority_vote(answers: &[String]) -> Option<String> {
    if answers.is_empty() {
        return None;
    }
    let mut buckets: HashMap<String, (usize, usize)> = HashMap::new();
    for (i, answer) in answers.iter().enumerate() {
        let key = normalize_answer(answer);
        let entry = buckets.entry(key).or_insert((0, i));
        entry.0 += 1;
    }
    let (_, &(count, first)) = buckets.iter().max_by(|a, b| {
        (a.1 .0, std::cmp::Reverse(a.1 .1)).cmp(&(b.1 .0, std::cmp::Reverse(b.1 .1)))
    })?;
    let _ = count;
    Some(answers[first].clone())
}

/// Parse a normalized answer as a number: plain decimals and `(a)/(b)`.
fn parse_number(s: &str) -> Option<f64> {
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let (num, den) = s.split_once('/')?;
    // Drop parens so a sign can sit inside or outside: -(35)/(9) == (-35)/(9).
    let strip = |x: &str| -> String { x.chars().filter(|c| !matches!(c, '(' | ')')).collect() };
    let n = strip(num).trim().parse::<f64>().ok()?;
    let d = strip(den).trim().parse::<f64>().ok()?;
    if d == 0.0 {
        None
    } else {
        Some(n / d)
    }
}

/// If the whole string is `\cmd{...}`, return the inner content.
fn unwrap_command(s: &str, cmd: &str) -> Option<String> {
    let t = s.trim();
    let rest = t.strip_prefix(cmd)?.trim_start();
    let inner = rest.strip_prefix('{')?;
    let close = matching_brace(inner)?;
    if inner[close + 1..].trim().is_empty() {
        Some(inner[..close].to_string())
    } else {
        None
    }
}

/// Replace the first `\cmd{...}` with its inner content.
fn rewrite_first_command(s: &str, cmd: &str) -> Option<String> {
    let start = s.find(cmd)?;
    let after = &s[start + cmd.len()..];
    let brace_rel = after.find('{')?;
    if !after[..brace_rel].trim().is_empty() {
        return None;
    }
    let inner = &after[brace_rel + 1..];
    let close = matching_brace(inner)?;
    Some(format!(
        "{}{}{}",
        &s[..start],
        &inner[..close],
        &inner[close + 1..]
    ))
}

/// Rewrite the first `\frac{a}{b}` as `(a)/(b)`.
fn rewrite_frac(s: &str) -> Option<String> {
    let start = s.find("\\frac")?;
    let after = &s[start + "\\frac".len()..];
    let first = after.trim_start().strip_prefix('{')?;
    let leading_ws = after.len() - after.trim_start().len();
    let a_end = matching_brace(first)?;
    let a = &first[..a_end];
    let rest = first[a_end + 1..].trim_start();
    let second = rest.strip_prefix('{')?;
    let b_end = matching_brace(second)?;
    let b = &second[..b_end];
    let consumed = start
        + "\\frac".len()
        + leading_ws
        + 1
        + a_end
        + 1
        + (first[a_end + 1..].len() - rest.len())
        + 1
        + b_end
        + 1;
    Some(format!("{}({})/({}){}", &s[..start], a, b, &s[consumed..]))
}

/// Index of the closing brace matching an implicit open brace before `s[0]`.
fn matching_brace(s: &str) -> Option<usize> {
    let mut depth = 1usize;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn remove_thousands_commas(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    for (i, &b) in bytes.iter().enumerate() {
        if b == b','
            && i > 0
            && bytes[i - 1].is_ascii_digit()
            && bytes.len() > i + 3
            && bytes[i + 1..i + 4].iter().all(u8::is_ascii_digit)
            && !bytes.get(i + 4).is_some_and(u8::is_ascii_digit)
        {
            continue;
        }
        out.push(b as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_boxed_and_dollars() {
        assert_eq!(normalize_answer("$\\boxed{64}$"), "64");
        assert_eq!(normalize_answer("  64. "), "64");
    }

    #[test]
    fn normalizes_fractions() {
        assert_eq!(normalize_answer("\\frac{1}{2}"), "(1)/(2)");
        assert_eq!(normalize_answer("\\dfrac{1}{2}"), "(1)/(2)");
        assert!(answers_equal("\\frac{1}{2}", "0.5"));
        assert!(answers_equal("\\frac{128}{2}", "64"));
    }

    #[test]
    fn normalizes_left_right_and_spacing() {
        assert!(answers_equal(
            "\\left( 3, \\frac{\\pi}{2} \\right)",
            "(3,\\frac{\\pi}{2})"
        ));
    }

    #[test]
    fn nested_fraction_normalizes() {
        assert_eq!(normalize_answer("\\frac{\\frac{1}{2}}{3}"), "((1)/(2))/(3)");
    }

    #[test]
    fn thousands_commas_removed_but_tuples_kept() {
        assert!(answers_equal("1,000", "1000"));
        assert_eq!(normalize_answer("(1,2)"), "(1,2)");
    }

    #[test]
    fn text_command_unwrapped() {
        assert!(answers_equal("12\\text{ cm}", "12cm"));
    }

    #[test]
    fn numeric_equivalence() {
        assert!(answers_equal("0.50", "1/2"));
        assert!(!answers_equal("2", "3"));
    }

    #[test]
    fn fraction_sign_placement_equivalent() {
        assert!(answers_equal("-\\frac{35}{9}", "\\frac{-35}{9}"));
        assert!(!answers_equal("-\\frac{35}{9}", "\\frac{35}{9}"));
    }

    #[test]
    fn assignment_prefix_stripped() {
        assert!(answers_equal("x=5", "5"));
        assert!(answers_equal("$k = \\frac{1}{2}$", "1/2"));
        assert!(!answers_equal("x=5", "6"));
        assert!(answers_equal("(1,2)", "(1,2)"));
    }

    #[test]
    fn majority_vote_picks_most_common() {
        let answers = vec![
            "64".to_string(),
            "$64$".to_string(),
            "63".to_string(),
            "\\boxed{64}".to_string(),
        ];
        assert_eq!(majority_vote(&answers).as_deref(), Some("64"));
    }

    #[test]
    fn majority_vote_tie_breaks_to_earliest() {
        let answers = vec!["7".to_string(), "9".to_string()];
        assert_eq!(majority_vote(&answers).as_deref(), Some("7"));
        assert_eq!(majority_vote(&[]), None);
    }
}
