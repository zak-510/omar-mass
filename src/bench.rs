//! HARDMath validation harness: run a method over the subset, score each
//! answer with the LLM judge (arXiv:2410.09988); accuracy = mean score.

use crate::mailbox;
use crate::runner::{InstanceResult, ModelConfig, Runner, RunnerOptions, TaskInstance};
use crate::topology::{AggregatorMode, TopologyConfig};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The four baseline methods, expressed as topologies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Method {
    /// Chain-of-thought: a single predictor.
    Cot,
    /// Self-refine: predictor + up to 4 reflect rounds.
    #[value(name = "self-refine")]
    SelfRefine,
    /// Self-consistency @5: 5 parallel predictors, LLM aggregator picks the
    /// most consistent answer (rule vote can't match open-form expressions).
    Sc5,
    /// Multi-agent debate: 3 agents x 2 rounds + LLM aggregator (10 calls).
    Debate,
}

impl Method {
    pub fn preset(&self) -> (TopologyConfig, AggregatorMode) {
        match self {
            Method::Cot => (TopologyConfig::default(), AggregatorMode::Rule),
            Method::SelfRefine => (
                TopologyConfig {
                    reflect: 4,
                    ..Default::default()
                },
                AggregatorMode::Rule,
            ),
            Method::Sc5 => (
                TopologyConfig {
                    aggregate: 5,
                    ..Default::default()
                },
                AggregatorMode::Llm,
            ),
            Method::Debate => (
                TopologyConfig {
                    aggregate: 3,
                    debate: 2,
                    ..Default::default()
                },
                AggregatorMode::Llm,
            ),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Method::Cot => "cot",
            Method::SelfRefine => "self-refine",
            Method::Sc5 => "sc5",
            Method::Debate => "debate",
        }
    }
}

/// One HARDMath problem from `data/hardmath_subset.jsonl`.
#[derive(Debug, Clone, Deserialize)]
pub struct Problem {
    pub id: String,
    pub question: String,
    /// Ground-truth worked solution, handed to the LLM judge.
    pub solution: String,
    /// Final gold answer, for logging.
    pub answer: String,
    pub question_type: String,
}

pub fn load_problems(path: &Path) -> Result<Vec<Problem>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read dataset {}", path.display()))?;
    let mut problems = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let problem: Problem = serde_json::from_str(line)
            .with_context(|| format!("Bad JSONL line {} in {}", i + 1, path.display()))?;
        problems.push(problem);
    }
    if problems.is_empty() {
        bail!("Dataset {} has no problems", path.display());
    }
    Ok(problems)
}

/// Deterministic subset selection: rotate by seed, take n.
pub fn select<T: Clone>(items: &[T], n: usize, seed: u64) -> Vec<T> {
    let len = items.len();
    let n = n.min(len);
    let start = (seed as usize) % len;
    (0..n).map(|i| items[(start + i) % len].clone()).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredInstance {
    pub task_id: String,
    pub question_type: String,
    pub gold: String,
    pub predicted: Option<String>,
    /// Judge score in [0,1].
    pub score: f64,
    pub llm_calls: usize,
    pub failures: usize,
}

/// Per-question-type breakdown (mean score over that type's instances).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeStats {
    pub n: usize,
    pub accuracy: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchSummary {
    pub method: String,
    pub backend: String,
    pub model: Option<String>,
    /// Parallel-chain width actually run.
    pub aggregate: usize,
    pub n: usize,
    pub seed: u64,
    /// Mean judge score across instances.
    pub accuracy: f64,
    /// Mean score per question_type (some rubrics, e.g. ODE, score much lower).
    pub by_type: BTreeMap<String, TypeStats>,
    pub total_llm_calls: usize,
    pub total_failures: usize,
    pub run_id: String,
    pub instances: Vec<ScoredInstance>,
}

pub struct BenchArgs {
    pub method: Method,
    pub data: PathBuf,
    pub n: usize,
    pub seed: u64,
    /// Max agents resident/called at once (0 = unlimited). Below a wave's
    /// width, the wave runs in sequential batches.
    pub max_concurrent: usize,
    /// Resume a previous run: load the partial file and skip done problems.
    pub resume: bool,
    pub model: ModelConfig,
    pub timeout: Duration,
    pub out: Option<PathBuf>,
}

fn make_summary(
    args: &BenchArgs,
    aggregate: usize,
    instances: &[ScoredInstance],
    run_id: &str,
) -> BenchSummary {
    let total_score: f64 = instances.iter().map(|s| s.score).sum();
    let mut sums: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    for s in instances {
        let e = sums.entry(s.question_type.clone()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += s.score;
    }
    let by_type = sums
        .into_iter()
        .map(|(t, (n, sum))| {
            (
                t,
                TypeStats {
                    n,
                    accuracy: sum / n as f64,
                },
            )
        })
        .collect();
    BenchSummary {
        method: args.method.name().to_string(),
        backend: args.model.backend.clone(),
        model: args.model.model.clone(),
        aggregate,
        n: instances.len(),
        seed: args.seed,
        accuracy: total_score / instances.len().max(1) as f64,
        by_type,
        total_llm_calls: instances.iter().map(|s| s.llm_calls).sum(),
        total_failures: instances.iter().map(|s| s.failures).sum(),
        run_id: run_id.to_string(),
        instances: instances.to_vec(),
    }
}

fn runner_id_from_partial(partial_path: &std::path::Path) -> String {
    std::fs::read_to_string(partial_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<BenchSummary>(&raw).ok())
        .map(|s| s.run_id)
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()[..4].to_string())
}

pub fn run_bench(args: BenchArgs) -> Result<BenchSummary> {
    let problems = load_problems(&args.data)?;
    let subset = select(&problems, args.n, args.seed);
    let (topology, aggregator) = args.method.preset();

    let out_path = args.out.clone();
    let partial_path = out_path
        .as_deref()
        .map(|p| p.with_extension("partial.json"))
        .unwrap_or_else(|| std::env::temp_dir().join("mass_partial.json"));

    // Resume: load already-completed instances and skip those problem IDs.
    let mut instances: Vec<ScoredInstance> = Vec::new();
    if args.resume {
        if let Ok(raw) = std::fs::read_to_string(&partial_path) {
            if let Ok(prior) = serde_json::from_str::<BenchSummary>(&raw) {
                eprintln!(
                    "[mass] resume: loaded {}/{} instances from {}",
                    prior.instances.len(),
                    subset.len(),
                    partial_path.display()
                );
                instances = prior.instances;
            }
        }
    }
    let done_ids: std::collections::HashSet<String> =
        instances.iter().map(|i| i.task_id.clone()).collect();
    let remaining: Vec<&Problem> = subset
        .iter()
        .filter(|p| !done_ids.contains(&p.id))
        .collect();

    if remaining.is_empty() {
        eprintln!("[mass] resume: all {} instances already done", subset.len());
    } else {
        eprintln!(
            "[mass] bench method={} n={} ({} remaining) seed={} topology={:?} backend={} model={:?}",
            args.method.name(),
            subset.len(),
            remaining.len(),
            args.seed,
            topology,
            args.model.backend,
            args.model.model
        );

        let mut runner = Runner::setup(RunnerOptions {
            topology,
            aggregator,
            model: args.model.clone(),
            timeout: args.timeout,
            max_concurrent: args.max_concurrent,
            with_grader: true,
            run_root: None,
        })?;
        eprintln!(
            "[mass] run {} ready; results under {}",
            runner.run_id,
            runner.run_dir.root.display()
        );

        let total = subset.len();
        let done_so_far = instances.len();
        for (i, problem) in remaining.iter().enumerate() {
            let task = TaskInstance {
                id: problem.id.clone(),
                question: problem.question.clone(),
                context: None,
                tests: None,
            };
            let scored = match runner.run_instance(&task) {
                Ok(InstanceResult {
                    answer,
                    llm_calls,
                    failures,
                    ..
                }) => {
                    let score = match answer.as_deref() {
                        Some(a) => runner
                            .grade(&problem.id, a, &problem.solution, &problem.question_type)
                            .unwrap_or(0.0),
                        None => 0.0,
                    };
                    ScoredInstance {
                        task_id: problem.id.clone(),
                        question_type: problem.question_type.clone(),
                        gold: problem.answer.clone(),
                        predicted: answer,
                        score,
                        llm_calls,
                        failures,
                    }
                }
                Err(err) => {
                    eprintln!("[mass] instance {} failed: {err:#}", problem.id);
                    ScoredInstance {
                        task_id: problem.id.clone(),
                        question_type: problem.question_type.clone(),
                        gold: problem.answer.clone(),
                        predicted: None,
                        score: 0.0,
                        llm_calls: 0,
                        failures: 0,
                    }
                }
            };
            eprintln!(
                "[mass] [{}/{}] {} predicted={:?} score={:.2}",
                done_so_far + i + 1,
                total,
                problem.id,
                scored.predicted,
                scored.score
            );
            instances.push(scored);

            // Flush partial results after every problem so a crash is resumable.
            let partial = make_summary(&args, topology.aggregate, &instances, "partial");
            let _ = mailbox::write_json_atomic(&partial_path, &partial);
        }

        runner.teardown()?;
    }

    let summary = make_summary(
        &args,
        topology.aggregate,
        &instances,
        &runner_id_from_partial(&partial_path),
    );
    let out = out_path.unwrap_or_else(|| partial_path.with_extension("json"));
    mailbox::write_json_atomic(&out, &summary)?;
    let _ = std::fs::remove_file(&partial_path);
    eprintln!(
        "[mass] {}: mean score {:.3} over {} (calls={}, failures={}) -> {}",
        summary.method,
        summary.accuracy,
        summary.n,
        summary.total_llm_calls,
        summary.total_failures,
        out.display()
    );
    for (t, st) in &summary.by_type {
        eprintln!("[mass]   {t}: {:.3} (n={})", st.accuracy, st.n);
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_respect_cap_and_shapes() {
        for method in [Method::Cot, Method::SelfRefine, Method::Sc5, Method::Debate] {
            let (topology, aggregator) = method.preset();
            topology.validate().unwrap();
            topology.check_cap(aggregator).unwrap();
        }
        let (debate, mode) = Method::Debate.preset();
        assert_eq!(debate.llm_calls(mode), 10); // 3x3 + 1, matching the paper
        let (sc5, mode) = Method::Sc5.preset();
        assert_eq!(sc5.llm_calls(mode), 6); // 5 predictors + LLM aggregator
        assert_eq!(mode, AggregatorMode::Llm);
    }

    #[test]
    fn make_summary_breaks_down_by_type() {
        let inst = |t: &str, score: f64| ScoredInstance {
            task_id: "x".into(),
            question_type: t.into(),
            gold: "g".into(),
            predicted: Some("p".into()),
            score,
            llm_calls: 1,
            failures: 0,
        };
        let instances = vec![
            inst("ODE", 0.0),
            inst("ODE", 0.1),
            inst("polynomial_roots", 0.8),
        ];
        let args = BenchArgs {
            method: Method::Cot,
            data: PathBuf::new(),
            n: 3,
            seed: 0,
            max_concurrent: 0,
            resume: false,
            model: ModelConfig::default(),
            timeout: Duration::from_secs(1),
            out: None,
        };
        let s = make_summary(&args, 1, &instances, "test");
        assert!((s.accuracy - 0.3).abs() < 1e-9); // (0+0.1+0.8)/3
        assert_eq!(s.by_type["ODE"].n, 2);
        assert!((s.by_type["ODE"].accuracy - 0.05).abs() < 1e-9);
        assert!((s.by_type["polynomial_roots"].accuracy - 0.8).abs() < 1e-9);
    }

    #[test]
    fn vendored_dataset_loads_and_selects() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/hardmath_subset.jsonl");
        let problems = load_problems(&path).unwrap();
        assert_eq!(problems.len(), 300);
        for p in &problems {
            assert!(!p.question.is_empty());
            assert!(!p.solution.is_empty());
            assert!(!p.id.is_empty());
        }
        let a = select(&problems, 20, 0);
        let b = select(&problems, 20, 0);
        assert_eq!(a.len(), 20);
        assert_eq!(a[0].id, b[0].id); // deterministic
        let c = select(&problems, 20, 7);
        assert_ne!(a[0].id, c[0].id); // seed shifts the slice
    }

    #[test]
    fn dataset_balanced_across_types() {
        use std::collections::BTreeMap;
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/hardmath_subset.jsonl");
        let problems = load_problems(&path).unwrap();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for p in &problems {
            *counts.entry(p.question_type.clone()).or_default() += 1;
        }
        assert_eq!(counts.len(), 6, "six question types: {counts:?}");
        assert!(counts.values().all(|&c| c == 50), "50 each: {counts:?}");
    }
}
