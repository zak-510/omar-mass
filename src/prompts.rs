//! MATH role prompt templates ported from the MASS paper (arXiv:2502.02533).
//!
//! App. D templates (Predictor / Reflector / Refiner / Debator / Summarizer)
//! plus the App. E optimized predictor. The base predictor uses the DSPy
//! `Answer:` field; we append a tag instruction so one parser covers every
//! role, with `Answer:` as the fallback.

/// Which predictor instruction to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictorKind {
    /// App. D zero-shot CoT predictor ("Let's think step by step.").
    Cot,
    /// App. E best-discovered MATH predictor (instruction + exemplar).
    Optimized,
}

const ANSWER_TAG_INSTRUCTION: &str =
    "Show your final answer bracketed between <answer> and </answer> at the end.";

/// App. E best MATH predictor: instruction plus its worked exemplar.
const OPTIMIZED_PREDICTOR_INSTRUCTION: &str = "Let's think step by step to solve the given problem. Clearly explain your reasoning process, showing all intermediate calculations and justifications. Express your final answer as a single numerical value or simplified expression enclosed within <answer> </answer> tags. Avoid extraneous text or explanations outside of the core reasoning and final answer.";

const OPTIMIZED_PREDICTOR_EXEMPLAR: &str = r"Question: What is the average of the two smallest positive integer solutions to the congruence $$14u \equiv 46 \pmod{100}~?$$
Reasoning: Let's think step by step in order to find the average of the two smallest positive integer solutions to the congruence $14u \equiv 46 \pmod{100}$. We are looking for solutions to $14u \equiv 46 \pmod{100}$. We can divide the congruence by 2 to get $7u \equiv 23 \pmod{50}$. We want to find the inverse of 7 modulo 50. We use the Euclidean Algorithm: \begin{align*} 50 &= 7 \cdot 7 + 1 \\ 1 &= 50 - 7 \cdot 7 \end{align*} Thus, $7(-7) \equiv 1 \pmod{50}$, so $7^{-1} \equiv -7 \equiv 43 \pmod{50}$. Multiplying $7u \equiv 23 \pmod{50}$ by 43 gives $u \equiv 23 \cdot 43 \pmod{50}$. $23 \cdot 43 = 989 = 50 \cdot 19 + 39$, so $23 \cdot 43 \equiv 39 \pmod{50}$. Therefore, $u \equiv 39 \pmod{50}$. The two smallest positive integer solutions are $u = 39$ and $u = 39+50=89$. The average of these two solutions is $\frac{39+89}{2} = \frac{128}{2} = 64$.
Answer: 64";

/// Render the predictor prompt for a question. When a summary from the
/// Summarize block is present it is provided as additional context.
pub fn predictor(kind: PredictorKind, question: &str, summary: Option<&str>) -> String {
    let context = match summary {
        Some(s) => format!("Context (summarized): {}\n", s),
        None => String::new(),
    };
    match kind {
        PredictorKind::Cot => format!(
            "Let's think step by step. {tag}\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nReasoning: Let's think step by step in order to ${{produce the answer}}. We ...\nAnswer: ${{answer}}\n\n---\n\n{context}Question: {question}\nReasoning: Let's think step by step in order to",
            tag = ANSWER_TAG_INSTRUCTION,
            context = context,
            question = question,
        ),
        PredictorKind::Optimized => format!(
            "{instruction}\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nReasoning: Let's think step by step in order to ${{produce the answer}}. We ...\nAnswer: ${{answer}}\n\n---\n\n{exemplar}\n\n---\n\n{context}Question: {question}\nReasoning: Let's think step by step in order to",
            instruction = OPTIMIZED_PREDICTOR_INSTRUCTION,
            exemplar = OPTIMIZED_PREDICTOR_EXEMPLAR,
            context = context,
            question = question,
        ),
    }
}

/// App. D MATH Reflector: critiques a prediction, emits Feedback +
/// Correctness fields.
pub fn reflector(question: &str, text: &str) -> String {
    format!(
        "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output 'True' in 'correctness'.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nText: ${{text}}\nReasoning: Let's think step by step in order to ${{produce the correctness}}. We ...\nFeedback: ${{feedback}}\nCorrectness: True/False indicating if answer is correct given the question.\n\n---\n\nQuestion: {question}\nText: {text}\nReasoning: Let's think step by step in order to",
        question = question,
        text = text,
    )
}

/// App. D MATH Refiner: revises a prediction given the reflection.
pub fn refiner(
    question: &str,
    previous_answer: &str,
    reflection: &str,
    correctness: bool,
) -> String {
    format!(
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. Show your final answer bracketed between <answer> and </answer> at the end.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nPrevious answer: ${{previous_answer}}\nReflection: ${{reflection}}\nCorrectness: ${{correctness}}\nThinking: ${{thinking}}\nAnswer: ${{answer}}\n\n---\n\nQuestion: {question}\nPrevious answer: {previous_answer}\nReflection: {reflection}\nCorrectness: {correctness}\nThinking:",
        question = question,
        previous_answer = previous_answer,
        reflection = reflection,
        correctness = correctness,
    )
}

/// App. D MATH Debator: sees all other agents' solutions, produces an
/// updated answer.
pub fn debator(question: &str, solutions: &[String]) -> String {
    let mut listed = String::new();
    for (i, solution) in solutions.iter().enumerate() {
        listed.push_str(&format!("[Agent {}] {}\n\n", i + 1, solution));
    }
    format!(
        "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. Show your final answer bracketed between <answer> and </answer> at the end.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nSolutions: the solutions to the question from other agents\nReasoning: Let's think step by step in order to ${{Examine the solutions from other agents}}. We ...\nAnswer: The updated answer for the question. Do not repeat Answer:\n\n---\n\nQuestion: {question}\nSolutions:\n{listed}Reasoning: Let's think step by step in order to",
        question = question,
        listed = listed,
    )
}

/// App. D long-context Summarizer. Exercises the Summarize path; MATH
/// itself has no long context.
pub fn summarizer(question: &str, context: &str) -> String {
    format!(
        "Based on the question, retrieve relevant information from context that is ONLY helpful in answering the question. Include all key information. Do not repeat context.\n\n---\n\nFollow the following format.\n\nQuestion: ${{question}}\nContext: ${{context}}\nSummary: Only generate the summary. Start with Summary:\n\n---\n\nQuestion: {question}\nContext: {context}\nSummary:",
        question = question,
        context = context,
    )
}

/// LLM aggregator (used by the debate baseline; MATH itself uses the
/// rule-based majority vote, not this).
pub fn aggregator(question: &str, answers: &[String]) -> String {
    let mut listed = String::new();
    for (i, answer) in answers.iter().enumerate() {
        listed.push_str(&format!("[Agent {}] {}\n\n", i + 1, answer));
    }
    format!(
        "These are the final solutions to the question from multiple agents. Judge the solutions and make the final prediction: pick the most consistent and correct answer. Show your final answer bracketed between <answer> and </answer> at the end.\n\nQuestion: {question}\nSolutions:\n{listed}",
        question = question,
        listed = listed,
    )
}

/// Executor request: actually run the candidate code against the public
/// tests (agents have real shell access) and report the outcome verbatim.
pub fn executor(code: &str, tests: &str) -> String {
    format!(
        "You are a code executor with real shell access. Take the candidate solution below, write it to a file together with the test cases, and actually run it (e.g. with python3). Do not reason about whether the code looks correct, execute it. Report exactly what happened in this format:\n\nExecution result: PASS or FAIL\nOutput: the full stdout/stderr including any traceback, verbatim\n\nCandidate solution:\n{code}\n\nTest cases:\n{tests}",
        code = code,
        tests = tests,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_cot_embeds_question_and_tag_convention() {
        let p = predictor(PredictorKind::Cot, "What is 2+2?", None);
        assert!(p.contains("Let's think step by step."));
        assert!(p.contains("Question: What is 2+2?"));
        assert!(p.contains("<answer>"));
        assert!(!p.contains("Context (summarized)"));
    }

    #[test]
    fn predictor_optimized_includes_exemplar() {
        let p = predictor(PredictorKind::Optimized, "Q", None);
        assert!(p.contains("14u \\equiv 46"));
        assert!(p.contains("Answer: 64"));
        assert!(p.contains("<answer> </answer>"));
    }

    #[test]
    fn predictor_includes_summary_when_present() {
        let p = predictor(PredictorKind::Cot, "Q", Some("key facts"));
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
    fn summarizer_and_executor_render() {
        let s = summarizer("Q", "long context here");
        assert!(s.contains("Context: long context here"));
        assert!(s.contains("Do not repeat context."));

        let e = executor("def add(a,b): return a+b", "assert add(1,2)==3");
        assert!(e.contains("assert add(1,2)==3"));
        assert!(e.contains("Execution result: PASS or FAIL"));
    }

    #[test]
    fn aggregator_lists_answers() {
        let a = aggregator("Q", &["1".into(), "2".into()]);
        assert!(a.contains("[Agent 2] 2"));
        assert!(a.contains("<answer>"));
    }
}
