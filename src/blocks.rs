//! The five topology building blocks.
//! Each function builds the wave of role calls for one stage. Pure prompt
//! construction; the runner dispatches them and feeds results back in.

use crate::prompts::{self, PredictorKind};
use crate::protocol::Role;

/// One LLM call: which session (role, slot), which round, rendered prompt.
#[derive(Debug, Clone)]
pub struct CallSpec {
    pub role: Role,
    pub slot: usize,
    pub round: usize,
    pub payload: String,
}

/// Summarize: each chain compresses the context, or re-summarizes its own
/// previous summary on later rounds.
pub fn summarize_wave(
    width: usize,
    round: usize,
    question: &str,
    context: &str,
    prev: &[Option<String>],
) -> Vec<CallSpec> {
    (1..=width)
        .map(|slot| {
            let source = prev
                .get(slot - 1)
                .and_then(|s| s.as_deref())
                .unwrap_or(context);
            CallSpec {
                role: Role::Summarizer,
                slot,
                round,
                payload: prompts::summarizer(question, source),
            }
        })
        .collect()
}

/// Aggregate (parallel part): N predictors answer independently.
pub fn predict_wave(
    kind: PredictorKind,
    width: usize,
    question: &str,
    summaries: &[Option<String>],
) -> Vec<CallSpec> {
    (1..=width)
        .map(|slot| CallSpec {
            role: Role::Predictor,
            slot,
            round: 0,
            payload: prompts::predictor(
                kind,
                question,
                summaries.get(slot - 1).and_then(|s| s.as_deref()),
            ),
        })
        .collect()
}

/// Execute: run each live chain's candidate code against the public tests.
pub fn execute_wave(candidates: &[String], alive: &[bool], tests: &str) -> Vec<CallSpec> {
    candidates
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(i, candidate)| CallSpec {
            role: Role::Executor,
            slot: i + 1,
            round: 0,
            payload: prompts::executor(candidate, tests),
        })
        .collect()
}

/// Reflect (critique half): reflectors review each active chain's latest
/// solution. A chain goes inactive once it stops early on a True verdict.
pub fn reflect_wave(
    round: usize,
    question: &str,
    texts: &[String],
    active: &[bool],
) -> Vec<CallSpec> {
    texts
        .iter()
        .enumerate()
        .filter(|(i, _)| active[*i])
        .map(|(i, text)| CallSpec {
            role: Role::Reflector,
            slot: i + 1,
            round,
            payload: prompts::reflector(question, text),
        })
        .collect()
}

/// Reflect (revision half): refiners revise chains the reflector marked
/// not yet correct.
pub fn refine_wave(
    round: usize,
    question: &str,
    answers: &[String],
    reflections: &[Option<crate::protocol::Reflection>],
) -> Vec<CallSpec> {
    reflections
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.as_ref().map(|r| (i, r)))
        .filter(|(_, r)| !r.correct)
        .map(|(i, r)| CallSpec {
            role: Role::Refiner,
            slot: i + 1,
            round,
            payload: prompts::refiner(question, &answers[i], &r.feedback, r.correct),
        })
        .collect()
}

/// Debate: one fully connected round. Every live chain sees all live
/// chains' current solutions and produces an updated answer.
pub fn debate_wave(
    round: usize,
    question: &str,
    texts: &[String],
    alive: &[bool],
) -> Vec<CallSpec> {
    let solutions: Vec<String> = texts
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(_, t)| t.clone())
        .collect();
    texts
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(i, _)| CallSpec {
            role: Role::Debator,
            slot: i + 1,
            round,
            payload: prompts::debator(question, &solutions),
        })
        .collect()
}

/// LLM aggregation call over the chains' final solutions.
pub fn aggregate_call(question: &str, texts: &[String]) -> CallSpec {
    CallSpec {
        role: Role::Aggregator,
        slot: 0,
        round: 0,
        payload: prompts::aggregator(question, texts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Reflection;

    #[test]
    fn summarize_uses_context_then_previous_summary() {
        let wave = summarize_wave(2, 0, "Q", "raw ctx", &[None, None]);
        assert_eq!(wave.len(), 2);
        assert!(wave[0].payload.contains("raw ctx"));

        let prev = vec![Some("sum-1".to_string()), Some("sum-2".to_string())];
        let wave2 = summarize_wave(2, 1, "Q", "raw ctx", &prev);
        assert!(wave2[0].payload.contains("sum-1"));
        assert!(!wave2[0].payload.contains("raw ctx"));
        assert!(wave2[1].payload.contains("sum-2"));
    }

    #[test]
    fn predict_wave_has_one_call_per_chain() {
        let wave = predict_wave(PredictorKind::Cot, 3, "Q", &[]);
        assert_eq!(wave.len(), 3);
        assert_eq!(wave[2].slot, 3);
        assert!(wave.iter().all(|c| c.role == Role::Predictor));
    }

    #[test]
    fn reflect_wave_skips_inactive_chains() {
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let wave = reflect_wave(0, "Q", &texts, &[true, false, true]);
        assert_eq!(wave.len(), 2);
        assert_eq!(wave[0].slot, 1);
        assert_eq!(wave[1].slot, 3);
    }

    #[test]
    fn refine_wave_only_for_incorrect_verdicts() {
        let answers = vec!["1".into(), "2".into()];
        let reflections = vec![
            Some(Reflection {
                correct: true,
                feedback: "fine".into(),
            }),
            Some(Reflection {
                correct: false,
                feedback: "wrong".into(),
            }),
        ];
        let wave = refine_wave(0, "Q", &answers, &reflections);
        assert_eq!(wave.len(), 1);
        assert_eq!(wave[0].slot, 2);
        assert!(wave[0].payload.contains("wrong"));
    }

    #[test]
    fn debate_wave_is_fully_connected() {
        let texts = vec!["sol-a".into(), "sol-b".into(), "sol-c".into()];
        let wave = debate_wave(1, "Q", &texts, &[true, true, true]);
        assert_eq!(wave.len(), 3);
        for call in &wave {
            assert!(call.payload.contains("sol-a"));
            assert!(call.payload.contains("sol-b"));
            assert!(call.payload.contains("sol-c"));
        }
    }

    #[test]
    fn debate_wave_excludes_dead_chains() {
        let texts = vec!["sol-a".into(), "dead".into(), "sol-c".into()];
        let wave = debate_wave(0, "Q", &texts, &[true, false, true]);
        assert_eq!(wave.len(), 2);
        assert_eq!(wave[1].slot, 3);
        assert!(!wave[0].payload.contains("dead"));
    }

    #[test]
    fn execute_wave_carries_code_and_tests() {
        let wave = execute_wave(&["def f(): pass".into()], &[true], "assert f() is None");
        assert_eq!(wave[0].role, Role::Executor);
        assert!(wave[0].payload.contains("def f(): pass"));
        assert!(wave[0].payload.contains("assert f() is None"));
    }
}
