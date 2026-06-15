//! MATH validation harness: runs a baseline method over a dataset subset
//! and scores it against gold answers. Relative comparison only, see
//! VALIDATION.md.

use crate::prompts::PredictorKind;
use crate::runner::{InstanceResult, ModelConfig, Runner, RunnerOptions, TaskInstance};
use crate::topology::{AggregatorMode, TopologyConfig};
use crate::{mailbox, math};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Baseline methods from the paper (App. B.2), expressed as topologies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Method {
    /// Chain-of-thought: a single predictor.
    Cot,
    /// Self-consistency @9: Aggregate-9 with rule-based majority vote.
    Sc9,
    /// Multi-agent debate: 3 agents x 3 rounds + aggregator (10 calls).
    Mad,
    /// Self-refine: predictor + 4 reflect rounds (worst case 9 calls).
    Reflect,
    /// SC@9 with the paper's App. E tuned predictor prompt (the MASS-found
    /// MATH topology, Table 2).
    #[value(name = "sc9-tuned")]
    Sc9Tuned,
}

impl Method {
    pub fn preset(&self) -> (TopologyConfig, AggregatorMode, PredictorKind) {
        match self {
            Method::Cot => (
                TopologyConfig {
                    aggregate: 1,
                    ..Default::default()
                },
                AggregatorMode::Rule,
                PredictorKind::Cot,
            ),
            Method::Sc9 => (
                TopologyConfig {
                    aggregate: 9,
                    ..Default::default()
                },
                AggregatorMode::Rule,
                PredictorKind::Cot,
            ),
            Method::Mad => (
                TopologyConfig {
                    aggregate: 3,
                    debate: 2,
                    ..Default::default()
                },
                AggregatorMode::Llm,
                PredictorKind::Cot,
            ),
            Method::Reflect => (
                TopologyConfig {
                    aggregate: 1,
                    reflect: 4,
                    ..Default::default()
                },
                AggregatorMode::Rule,
                PredictorKind::Cot,
            ),
            Method::Sc9Tuned => (
                TopologyConfig {
                    aggregate: 9,
                    ..Default::default()
                },
                AggregatorMode::Rule,
                PredictorKind::Optimized,
            ),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Method::Cot => "cot",
            Method::Sc9 => "sc9",
            Method::Mad => "mad",
            Method::Reflect => "reflect",
            Method::Sc9Tuned => "sc9-tuned",
        }
    }
}

/// One MATH problem from `data/math_subset.jsonl`.
#[derive(Debug, Clone, Deserialize)]
pub struct MathProblem {
    pub id: String,
    pub problem: String,
    pub answer: String,
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub level: u32,
}

pub fn load_problems(path: &Path) -> Result<Vec<MathProblem>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read dataset {}", path.display()))?;
    let mut problems = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let problem: MathProblem = serde_json::from_str(line)
            .with_context(|| format!("Bad JSONL line {} in {}", i + 1, path.display()))?;
        problems.push(problem);
    }
    if problems.is_empty() {
        bail!("Dataset {} has no problems", path.display());
    }
    Ok(problems)
}

/// Deterministic subset selection: rotate by seed, take n. The vendored
/// subset is sorted by level, so a contiguous slice is not difficulty
/// balanced. Use select_stratified for a representative subset.
pub fn select<T: Clone>(items: &[T], n: usize, seed: u64) -> Vec<T> {
    let len = items.len();
    let n = n.min(len);
    let start = (seed as usize) % len;
    (0..n).map(|i| items[(start + i) % len].clone()).collect()
}

/// Difficulty-balanced subset: take ~n/levels problems from each level so a
/// run spans L1..L5 instead of the all-easy head. Deterministic in seed, and
/// consecutive seeds pick disjoint per-level windows, so every topology on the
/// same seed sees the same set while different seeds give different sets.
pub fn select_stratified(problems: &[MathProblem], n: usize, seed: u64) -> Vec<MathProblem> {
    use std::collections::BTreeMap;
    let n = n.min(problems.len());
    let mut by_level: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, p) in problems.iter().enumerate() {
        by_level.entry(p.level).or_default().push(i);
    }
    if by_level.is_empty() {
        return Vec::new();
    }
    let num_levels = by_level.len();
    let base = n / num_levels;
    let mut remainder = n % num_levels;
    let mut out = Vec::with_capacity(n);
    for idxs in by_level.values() {
        // Distribute any remainder to the lowest levels first.
        let mut k = base;
        if remainder > 0 {
            k += 1;
            remainder -= 1;
        }
        let k = k.min(idxs.len());
        if k == 0 {
            continue;
        }
        let len = idxs.len();
        let start = (seed as usize).wrapping_mul(k) % len;
        for j in 0..k {
            out.push(problems[idxs[(start + j) % len]].clone());
        }
    }
    out
}

#[derive(Debug, Serialize)]
pub struct ScoredInstance {
    pub task_id: String,
    pub subject: String,
    pub level: u32,
    pub gold: String,
    pub predicted: Option<String>,
    pub correct: bool,
    pub llm_calls: usize,
    pub failures: usize,
}

#[derive(Debug, Serialize)]
pub struct BenchSummary {
    pub method: String,
    pub backend: String,
    pub model: Option<String>,
    /// Effective parallel-chain width actually run (after any -k override),
    /// so the summary records SC@5 vs SC@9 regardless of method label.
    pub aggregate: usize,
    pub n: usize,
    pub seed: u64,
    pub correct: usize,
    pub accuracy: f64,
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
    /// Restrict to a single difficulty level before selection (e.g. `5` =
    /// hardest-only). `None` keeps all levels.
    pub level: Option<u32>,
    /// Balance the subset across difficulty levels (see
    /// [`select_stratified`]) instead of taking a contiguous slice.
    pub stratified: bool,
    /// Override the preset's parallel-chain width (e.g. SC@5 = sc9 method
    /// with `aggregate_k = 5`). Must be a legal width or the run bails.
    pub aggregate_k: Option<usize>,
    /// Max agents resident/called at once (0 = unlimited). Below a wave's
    /// width, the wave runs in sequential batches, which lets a memory-bound
    /// host run wide topologies like SC@5 locally.
    pub max_concurrent: usize,
    pub model: ModelConfig,
    pub timeout: Duration,
    pub out: Option<PathBuf>,
}

/// Apply an optional aggregate-width override to a preset topology,
/// re-validating dimensions and the LLM-call cap so a bad `-k` fails before
/// any agent spawns.
pub fn apply_aggregate_k(
    mut topology: TopologyConfig,
    aggregate_k: Option<usize>,
    aggregator: AggregatorMode,
) -> Result<TopologyConfig> {
    if let Some(k) = aggregate_k {
        topology.aggregate = k;
        topology.validate()?;
        topology.check_cap(aggregator)?;
    }
    Ok(topology)
}

pub fn run_bench(args: BenchArgs) -> Result<BenchSummary> {
    let mut problems = load_problems(&args.data)?;
    if let Some(level) = args.level {
        problems.retain(|p| p.level == level);
        if problems.is_empty() {
            bail!("Dataset has no problems at level {level}");
        }
    }
    let subset = if args.stratified {
        select_stratified(&problems, args.n, args.seed)
    } else {
        select(&problems, args.n, args.seed)
    };
    let (topology, aggregator, predictor) = args.method.preset();
    let topology = apply_aggregate_k(topology, args.aggregate_k, aggregator)?;

    eprintln!(
        "[mass] bench method={} n={} seed={} topology={:?} backend={} model={:?}",
        args.method.name(),
        subset.len(),
        args.seed,
        topology,
        args.model.backend,
        args.model.model
    );

    let mut runner = Runner::setup(RunnerOptions {
        topology,
        aggregator,
        predictor,
        model: args.model.clone(),
        timeout: args.timeout,
        max_concurrent: args.max_concurrent,
        run_root: None,
    })?;
    eprintln!(
        "[mass] run {} ready ({} agents); results under {}",
        runner.run_id,
        topology.session_specs(aggregator).len(),
        runner.run_dir.root.display()
    );

    let mut instances = Vec::new();
    let total = subset.len();
    for (i, problem) in subset.iter().enumerate() {
        let task = TaskInstance {
            id: problem.id.clone(),
            question: problem.problem.clone(),
            context: None,
            tests: None,
        };
        let outcome = runner.run_instance(&task);
        let scored = match outcome {
            Ok(InstanceResult {
                answer,
                llm_calls,
                failures,
                ..
            }) => {
                let correct = answer
                    .as_deref()
                    .is_some_and(|a| math::answers_equal(a, &problem.answer));
                ScoredInstance {
                    task_id: problem.id.clone(),
                    subject: problem.subject.clone(),
                    level: problem.level,
                    gold: problem.answer.clone(),
                    predicted: answer,
                    correct,
                    llm_calls,
                    failures,
                }
            }
            Err(err) => {
                eprintln!("[mass] instance {} failed: {err:#}", problem.id);
                ScoredInstance {
                    task_id: problem.id.clone(),
                    subject: problem.subject.clone(),
                    level: problem.level,
                    gold: problem.answer.clone(),
                    predicted: None,
                    correct: false,
                    llm_calls: 0,
                    failures: 0,
                }
            }
        };
        eprintln!(
            "[mass] [{}/{}] {} gold={} predicted={:?} correct={}",
            i + 1,
            total,
            problem.id,
            problem.answer,
            scored.predicted,
            scored.correct
        );
        instances.push(scored);
    }

    let correct = instances.iter().filter(|s| s.correct).count();
    let summary = BenchSummary {
        method: args.method.name().to_string(),
        backend: args.model.backend.clone(),
        model: args.model.model.clone(),
        aggregate: topology.aggregate,
        n: instances.len(),
        seed: args.seed,
        correct,
        accuracy: correct as f64 / instances.len().max(1) as f64,
        total_llm_calls: instances.iter().map(|s| s.llm_calls).sum(),
        total_failures: instances.iter().map(|s| s.failures).sum(),
        run_id: runner.run_id.clone(),
        instances,
    };

    let out = args
        .out
        .unwrap_or_else(|| runner.run_dir.root.join("summary.json"));
    mailbox::write_json_atomic(&out, &summary)?;
    eprintln!(
        "[mass] {}: {}/{} = {:.1}% (calls={}, failures={}) -> {}",
        summary.method,
        summary.correct,
        summary.n,
        summary.accuracy * 100.0,
        summary.total_llm_calls,
        summary.total_failures,
        out.display()
    );

    runner.teardown()?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_respect_cap_and_shapes() {
        for method in [
            Method::Cot,
            Method::Sc9,
            Method::Mad,
            Method::Reflect,
            Method::Sc9Tuned,
        ] {
            let (topology, aggregator, _) = method.preset();
            topology.validate().unwrap();
            topology.check_cap(aggregator).unwrap();
        }
        let (mad, mode, _) = Method::Mad.preset();
        assert_eq!(mad.llm_calls(mode), 10); // 3x3 + 1, matching the paper
        let (sc9, mode, _) = Method::Sc9.preset();
        assert_eq!(sc9.llm_calls(mode), 9);
    }

    #[test]
    fn level_filter_keeps_only_that_level() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/math_subset.jsonl");
        let problems = load_problems(&path).unwrap();
        let l5: Vec<_> = problems.iter().filter(|p| p.level == 5).cloned().collect();
        assert!(
            l5.len() >= 20,
            "expected >=20 level-5 problems, got {}",
            l5.len()
        );
        // n=20 over the level-5-only set is deterministic (same problems for
        // every method that passes the same seed).
        let a = select(&l5, 20, 0);
        let b = select(&l5, 20, 0);
        assert_eq!(a.len(), 20);
        assert!(a.iter().all(|p| p.level == 5));
        assert_eq!(
            a.iter().map(|p| &p.id).collect::<Vec<_>>(),
            b.iter().map(|p| &p.id).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn stratified_balances_levels_and_is_disjoint_across_seeds() {
        use std::collections::{BTreeMap, HashSet};
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/math_subset.jsonl");
        let problems = load_problems(&path).unwrap();

        let run1 = select_stratified(&problems, 20, 0);
        assert_eq!(run1.len(), 20);
        // 4 problems from each of the 5 levels.
        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        for p in &run1 {
            *counts.entry(p.level).or_default() += 1;
        }
        assert_eq!(counts.len(), 5, "all five levels represented");
        assert!(counts.values().all(|&c| c == 4), "balanced: {counts:?}");

        // Deterministic: same seed -> same set.
        let run1b = select_stratified(&problems, 20, 0);
        assert_eq!(
            run1.iter().map(|p| &p.id).collect::<Vec<_>>(),
            run1b.iter().map(|p| &p.id).collect::<Vec<_>>(),
        );

        // Consecutive seeds (run 1 vs run 2) are disjoint problem sets.
        let run2 = select_stratified(&problems, 20, 1);
        let ids1: HashSet<_> = run1.iter().map(|p| &p.id).collect();
        assert!(
            run2.iter().all(|p| !ids1.contains(&p.id)),
            "run 1 and run 2 should share no problems",
        );
    }

    #[test]
    fn aggregate_k_override_sets_width_and_guards_cap() {
        let (sc9, mode, _) = Method::Sc9.preset();
        // No override leaves the preset untouched.
        assert_eq!(apply_aggregate_k(sc9, None, mode).unwrap().aggregate, 9);
        // SC@5 = sc9 preset narrowed to width 5.
        let sc5 = apply_aggregate_k(sc9, Some(5), mode).unwrap();
        assert_eq!(sc5.aggregate, 5);
        assert_eq!(sc5.llm_calls(mode), 5);
        // Off-dimension widths are rejected before any spawn.
        assert!(apply_aggregate_k(sc9, Some(4), mode).is_err());
        // A width that would blow the 10-call cap is rejected too.
        let (mass, mmode, _) = Method::Sc9Tuned.preset();
        assert!(apply_aggregate_k(mass, Some(3), mmode).unwrap().aggregate == 3);
        let (summ, smode) = (
            TopologyConfig {
                summarize: 1,
                ..Default::default()
            },
            AggregatorMode::Rule,
        );
        assert!(apply_aggregate_k(summ, Some(9), smode).is_err());
    }

    #[test]
    fn vendored_dataset_loads_and_selects() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/math_subset.jsonl");
        let problems = load_problems(&path).unwrap();
        assert!(problems.len() >= 100, "expected >=100 vendored problems");
        for p in &problems {
            assert!(!p.problem.is_empty());
            assert!(!p.answer.is_empty());
            assert!(!p.id.is_empty());
        }
        let a = select(&problems, 20, 0);
        let b = select(&problems, 20, 0);
        assert_eq!(a.len(), 20);
        assert_eq!(a[0].id, b[0].id); // deterministic
        let c = select(&problems, 20, 7);
        assert_ne!(a[0].id, c[0].id); // seed shifts the slice
    }
}
