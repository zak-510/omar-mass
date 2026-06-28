//! Role prompt templates from the MASS paper (arXiv:2502.02533, App. D) plus
//! the HARDMath LLM-judge grading prompt (arXiv:2410.09988).

const ANSWER_TAG_INSTRUCTION: &str =
    "Show your final answer bracketed between <answer> and </answer> at the end.";

/// Deterministic per-slot strategy hint: keyed by 1-based chain index so
/// parallel chains diversify without temperature. Slot 1 is neutral, leaving
/// single-chain methods (cot, self-refine, width=1) byte-for-byte unchanged.
fn strategy_hint(slot: usize) -> &'static str {
    match slot {
        2 => " Prefer a numerical approach: estimate the answer, then verify it.",
        3 => " Watch the edge cases and cross-check with an alternate method.",
        4 => " Derive symbolically and simplify carefully before evaluating.",
        5 => " Reason about the asymptotic/limiting behavior to bound the answer.",
        _ => "",
    }
}

/// Zero-shot CoT predictor. A summary from the Summarize block, when present,
/// is passed as extra context. `slot` selects a deterministic strategy hint.
pub fn predictor(question: &str, summary: Option<&str>, slot: usize) -> String {
    let context = match summary {
        Some(s) => format!("Context (summarized): {}\n", s),
        None => String::new(),
    };
    format!(
        "Let's think step by step.{hint} {tag}\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nReasoning: Let's think step by step in order to ${{produce the answer}}. We ...\nAnswer: ${{answer}}\n\n---\n\n{context}Question: {question}\nReasoning: Let's think step by step in order to",
        hint = strategy_hint(slot),
        tag = ANSWER_TAG_INSTRUCTION,
    )
}

/// App. D Reflector: critiques a prediction, emits Feedback + Correctness.
pub fn reflector(question: &str, text: &str) -> String {
    format!(
        "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output 'True' in 'correctness'.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nText: ${{text}}\nReasoning: Let's think step by step in order to ${{produce the correctness}}. We ...\nFeedback: ${{feedback}}\nCorrectness: True/False indicating if answer is correct given the question.\n\n---\n\nQuestion: {question}\nText: {text}\nReasoning: Let's think step by step in order to",
    )
}

/// App. D Refiner: revises a prediction given the reflection.
pub fn refiner(
    question: &str,
    previous_answer: &str,
    reflection: &str,
    correctness: bool,
) -> String {
    format!(
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. Show your final answer bracketed between <answer> and </answer> at the end.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nPrevious answer: ${{previous_answer}}\nReflection: ${{reflection}}\nCorrectness: ${{correctness}}\nThinking: ${{thinking}}\nAnswer: ${{answer}}\n\n---\n\nQuestion: {question}\nPrevious answer: {previous_answer}\nReflection: {reflection}\nCorrectness: {correctness}\nThinking:",
    )
}

/// App. D Debator: sees all other agents' solutions, produces an updated answer.
/// `slot` selects the same deterministic strategy hint as the predictor.
pub fn debator(question: &str, solutions: &[String], slot: usize) -> String {
    let mut listed = String::new();
    for (i, solution) in solutions.iter().enumerate() {
        listed.push_str(&format!("[Agent {}] {}\n\n", i + 1, solution));
    }
    format!(
        "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer.{hint} Keep all comparison of the other agents in your reasoning; the <answer> must be the self-contained final result only, never a reference to another agent by number or label. Show your final answer bracketed between <answer> and </answer> at the end.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nSolutions: the solutions to the question from other agents\nReasoning: Let's think step by step in order to ${{Examine the solutions from other agents}}. We ...\nAnswer: The updated self-contained answer for the question, with no references to other agents. Do not repeat Answer:\n\n---\n\nQuestion: {question}\nSolutions:\n{listed}Reasoning: Let's think step by step in order to",
        hint = strategy_hint(slot),
    )
}

/// App. D long-context Summarizer.
pub fn summarizer(question: &str, context: &str) -> String {
    format!(
        "Based on the question, retrieve relevant information from context that is ONLY helpful in answering the question. Include all key information. Do not repeat context.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nContext: ${{context}}\nSummary: Only generate the summary. Start with Summary:\n\n---\n\nQuestion: {question}\nContext: {context}\nSummary:",
    )
}

/// LLM aggregator for SC@5 and Multi-Agent Debate. Candidates are labeled so the
/// judge can compare them, but it must restate the chosen answer in full — naming
/// a candidate (e.g. "Agent 4") would lose the answer and score 0.
pub fn aggregator(question: &str, answers: &[String]) -> String {
    let mut listed = String::new();
    for (i, answer) in answers.iter().enumerate() {
        listed.push_str(&format!("[Candidate {}] {}\n\n", i + 1, answer));
    }
    format!(
        "Below are candidate final answers to the question. Decide which is most consistent and correct, then output that answer in full as your own. Reproduce the complete answer expression itself; never refer to a candidate by number or label (do not write 'Candidate 2' or 'Agent 4') and do not add a 'consensus' prefix. Show the final answer bracketed between <answer> and </answer> at the end.\n\nQuestion: {question}\nCandidates:\n{listed}",
    )
}

/// Executor request: run the candidate code against the public tests and
/// report the outcome verbatim (agents have real shell access).
pub fn executor(code: &str, tests: &str) -> String {
    format!(
        "You are a code executor with real shell access. Take the candidate solution below, write it to a file together with the test cases, and actually run it (e.g. with python3). Do not reason about whether the code looks correct, execute it. Report exactly what happened in this format:\n\nExecution result: PASS or FAIL\nOutput: the full stdout/stderr including any traceback, verbatim\n\nCandidate solution:\n{code}\n\nTest cases:\n{tests}",
    )
}

/// Per-question-type grading rubric from the HARDMath eval (create_prompt.py).
fn rubric(question_type: &str) -> &'static str {
    match question_type {
        "polynomial_roots" | "polynomial_roots_corrections" => "Check both the small and large epsilon solutions. For each, give full credit if it matches the answer key; give partial credit proportional to the number of matching roots; give no credit if completely wrong. Average the two scores for a final score between 0 and 1.",
        "ODE" => "Check both the small and large regime solutions. For the small regime give full credit only on an exact form match, else no credit. For the large regime give full credit on an exact match, partial credit if the form differs but the numerical value is close, no credit otherwise. Average the two scores for a final score between 0 and 1.",
        "integral" => "Compare the response's analytical approximations against the answer key across the epsilon regimes. Give full credit if they match, partial credit for partially correct regimes, no credit if completely wrong. Final score between 0 and 1.",
        _ => "Give full credit (1.0) if the response's final expression is mathematically equivalent to the answer key, partial credit if close, and no credit (0.0) if wrong. Final score between 0 and 1.",
    }
}

/// Anonymous label for a batched-grade candidate: 0 -> A, 1 -> B, ...
pub fn label(i: usize) -> char {
    (b'A' + i as u8) as char
}

/// Batched grader: one call scores every candidate against the gold solution
/// under the same rubric, each judged independently (no cross-answer ranking).
pub fn judge_batch(answers: &[String], solution: &str, question_type: &str) -> String {
    let mut listed = String::new();
    for (i, a) in answers.iter().enumerate() {
        listed.push_str(&format!("[Answer {}]\n{}\n\n", label(i), a));
    }
    let keys: Vec<String> = (0..answers.len())
        .map(|i| format!("{}: <0-1>", label(i)))
        .collect();
    format!(
        "Score each candidate answer independently against the ground-truth solution using the rubric. Judge each answer ONLY against the ground truth, never relative to the other answers.\n\nGround truth solution: {solution}\n\nRubric:\n{rubric}\n\n{listed}Output every answer's float in [0,1] on one line bracketed between <answer> and </answer>, formatted exactly as: {keys}.",
        rubric = rubric(question_type),
        keys = keys.join(", "),
    )
}

/// HARDMath LLM-judge prompt: the judge sees the response and the ground-truth
/// solution, applies the type rubric, and returns a 0-1 float.
pub fn judge(predicted: &str, solution: &str, question_type: &str) -> String {
    format!(
        "Please take this response: {predicted}\n\nand this ground truth solution: {solution}\n\nand grade the response based on the following criteria:\n{rubric}\n\nGive only the final grade as a float between 0 and 1 bracketed between <answer> and </answer>.",
        rubric = rubric(question_type),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_embeds_question_and_tag() {
        let p = predictor("What is 2+2?", None, 1);
        assert!(p.contains("Let's think step by step."));
        assert!(p.contains("Question: What is 2+2?"));
        assert!(p.contains("<answer>"));
        assert!(!p.contains("Context (summarized)"));
    }

    #[test]
    fn predictor_includes_summary_when_present() {
        let p = predictor("Q", Some("key facts"), 1);
        assert!(p.contains("Context (summarized): key facts"));
    }

    #[test]
    fn predictor_slot1_is_neutral_others_perturbed() {
        // Slot 1 must stay identical so single-chain methods are unchanged.
        let s1 = predictor("Q", None, 1);
        assert!(s1.starts_with("Let's think step by step. Show your final answer"));
        // Slots >1 get a distinct deterministic hint.
        let s2 = predictor("Q", None, 2);
        let s3 = predictor("Q", None, 3);
        assert!(s2.contains("numerical approach"));
        assert!(s3.contains("edge cases"));
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);
        // Reproducible: same slot yields the same prompt.
        assert_eq!(s2, predictor("Q", None, 2));
    }

    #[test]
    fn reflector_and_refiner_carry_inputs() {
        let r = reflector("Q1", "candidate text");
        assert!(r.contains("Text: candidate text"));
        assert!(r.contains("Correctness: True/False"));

        let f = refiner("Q1", "4", "Feedback: wrong sign", false);
        assert!(f.contains("Previous answer: 4"));
        assert!(f.contains("Reflection: Feedback: wrong sign"));
        assert!(f.contains("Correctness: false"));
        assert!(f.contains("<answer>"));
    }

    #[test]
    fn debator_lists_all_solutions() {
        let d = debator("Q", &["sol a".into(), "sol b".into(), "sol c".into()], 1);
        assert!(d.contains("[Agent 1] sol a"));
        assert!(d.contains("[Agent 3] sol c"));
        assert!(d.contains("<answer>"));
        // <answer> must stay a self-contained result, no agent-reference leak.
        assert!(d.contains("self-contained final result only"));
        // Slot 1 neutral; slot 2 carries the strategy hint.
        assert!(!d.contains("numerical approach"));
        assert!(debator("Q", &["x".into()], 2).contains("numerical approach"));
    }

    #[test]
    fn summarizer_and_aggregator_render() {
        let s = summarizer("Q", "long context here");
        assert!(s.contains("Context: long context here"));
        let a = aggregator("Q", &["1".into(), "2".into()]);
        assert!(a.contains("[Candidate 2] 2"));
        assert!(a.contains("<answer>"));
        assert!(a.contains("never refer to a candidate"));

        let e = executor("def add(a,b): return a+b", "assert add(1,2)==3");
        assert!(e.contains("assert add(1,2)==3"));
        assert!(e.contains("Execution result: PASS or FAIL"));
    }

    #[test]
    fn judge_batch_anonymizes_and_demands_independent_scores() {
        let j = judge_batch(&["ans one".into(), "ans two".into()], "gold", "integral");
        assert!(j.contains("[Answer A]\nans one"));
        assert!(j.contains("[Answer B]\nans two"));
        assert!(j.contains("never relative to the other answers"));
        assert!(j.contains("A: <0-1>, B: <0-1>"));
        assert!(j.contains("epsilon regimes")); // integral rubric carried
                                                // No method names leak into the prompt.
        assert!(!j.to_lowercase().contains("cot") && !j.contains("debate"));
    }

    #[test]
    fn judge_carries_solution_and_type_rubric() {
        let j = judge("eps=0.14", "gold derivation", "polynomial_roots");
        assert!(j.contains("eps=0.14"));
        assert!(j.contains("gold derivation"));
        assert!(j.contains("small and large epsilon"));
        assert!(j.contains("<answer>"));
        // Unknown type falls back to the equivalence rubric.
        assert!(judge("a", "b", "integral").contains("epsilon regimes"));
        assert!(
            judge("a", "b", "nondimensionalization_numeric").contains("mathematically equivalent")
        );
    }
}
