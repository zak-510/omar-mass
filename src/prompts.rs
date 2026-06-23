//! Role prompt templates from the MASS paper (arXiv:2502.02533, App. D) plus
//! the HARDMath LLM-judge grading prompt (arXiv:2410.09988).

const ANSWER_TAG_INSTRUCTION: &str =
    "Show your final answer bracketed between <answer> and </answer> at the end.";

/// Zero-shot CoT predictor. A summary from the Summarize block, when present,
/// is passed as extra context.
pub fn predictor(question: &str, summary: Option<&str>) -> String {
    let context = match summary {
        Some(s) => format!("Context (summarized): {}\n", s),
        None => String::new(),
    };
    format!(
        "Let's think step by step. {tag}\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nReasoning: Let's think step by step in order to ${{produce the answer}}. We ...\nAnswer: ${{answer}}\n\n---\n\n{context}Question: {question}\nReasoning: Let's think step by step in order to",
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
pub fn debator(question: &str, solutions: &[String]) -> String {
    let mut listed = String::new();
    for (i, solution) in solutions.iter().enumerate() {
        listed.push_str(&format!("[Agent {}] {}\n\n", i + 1, solution));
    }
    format!(
        "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. Show your final answer bracketed between <answer> and </answer> at the end.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nSolutions: the solutions to the question from other agents\nReasoning: Let's think step by step in order to ${{Examine the solutions from other agents}}. We ...\nAnswer: The updated answer for the question. Do not repeat Answer:\n\n---\n\nQuestion: {question}\nSolutions:\n{listed}Reasoning: Let's think step by step in order to",
    )
}

/// App. D long-context Summarizer.
pub fn summarizer(question: &str, context: &str) -> String {
    format!(
        "Based on the question, retrieve relevant information from context that is ONLY helpful in answering the question. Include all key information. Do not repeat context.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nContext: ${{context}}\nSummary: Only generate the summary. Start with Summary:\n\n---\n\nQuestion: {question}\nContext: {context}\nSummary:",
    )
}

/// LLM aggregator (Multi-Agent Debate baseline; SC uses the rule vote).
pub fn aggregator(question: &str, answers: &[String]) -> String {
    let mut listed = String::new();
    for (i, answer) in answers.iter().enumerate() {
        listed.push_str(&format!("[Agent {}] {}\n\n", i + 1, answer));
    }
    format!(
        "These are the final solutions to the question from multiple agents. Judge the solutions and make the final prediction: pick the most consistent and correct answer. Show your final answer bracketed between <answer> and </answer> at the end.\n\nQuestion: {question}\nSolutions:\n{listed}",
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
        let p = predictor("What is 2+2?", None);
        assert!(p.contains("Let's think step by step."));
        assert!(p.contains("Question: What is 2+2?"));
        assert!(p.contains("<answer>"));
        assert!(!p.contains("Context (summarized)"));
    }

    #[test]
    fn predictor_includes_summary_when_present() {
        let p = predictor("Q", Some("key facts"));
        assert!(p.contains("Context (summarized): key facts"));
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
        let d = debator("Q", &["sol a".into(), "sol b".into(), "sol c".into()]);
        assert!(d.contains("[Agent 1] sol a"));
        assert!(d.contains("[Agent 3] sol c"));
        assert!(d.contains("<answer>"));
    }

    #[test]
    fn summarizer_and_aggregator_render() {
        let s = summarizer("Q", "long context here");
        assert!(s.contains("Context: long context here"));
        let a = aggregator("Q", &["1".into(), "2".into()]);
        assert!(a.contains("[Agent 2] 2"));
        assert!(a.contains("<answer>"));

        let e = executor("def add(a,b): return a+b", "assert add(1,2)==3");
        assert!(e.contains("assert add(1,2)==3"));
        assert!(e.contains("Execution result: PASS or FAIL"));
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
