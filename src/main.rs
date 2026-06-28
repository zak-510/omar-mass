//! omar-mass: MASS topology building blocks for OMAR. Run a task through a
//! topology, benchmark HARDMath, exercise a block, or clean up. See README.

mod bench;
mod blocks;
mod grade;
mod mailbox;
mod math;
mod omar;
mod prompts;
mod protocol;
mod runner;
mod topology;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use runner::{ModelConfig, Runner, RunnerOptions, TaskInstance};
use std::path::PathBuf;
use std::time::Duration;
use topology::{AggregatorMode, TopologyConfig};

#[derive(Parser)]
#[command(
    name = "omar-mass",
    about = "MASS topology building blocks on OMAR",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Block {
    Aggregate,
    Reflect,
    Debate,
    Summarize,
    Execute,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliAggregator {
    Rule,
    Llm,
}

#[derive(Subcommand)]
enum Command {
    /// Run one task instance through a topology config.
    Run {
        /// Topology JSON, e.g. '{"aggregate":5}' or '{"debate":2,"aggregate":3}'.
        #[arg(long)]
        topology: String,
        /// The question / problem statement.
        #[arg(long)]
        question: String,
        /// Optional long-context file (Summarize block input).
        #[arg(long)]
        context_file: Option<PathBuf>,
        /// Optional public test cases (Execute block input).
        #[arg(long)]
        tests: Option<String>,
        /// Aggregation: rule-based majority vote or LLM aggregator.
        #[arg(long, value_enum, default_value = "rule")]
        aggregator: CliAggregator,
        /// Max agents resident/called at once (0 = unlimited).
        #[arg(long, default_value_t = 0)]
        max_concurrent: usize,
        #[command(flatten)]
        model: ModelArgs,
        /// Per-wave reply timeout in seconds.
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
    },
    /// Run a HARDMath baseline (validation harness).
    Bench {
        /// Baseline method.
        #[arg(long, value_enum)]
        method: bench::Method,
        /// Number of problems.
        #[arg(long, default_value_t = 100)]
        n: usize,
        /// Subset rotation seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Max agents resident/called at once (0 = unlimited). Below the
        /// topology width, wide waves run in sequential batches.
        #[arg(long, default_value_t = 0)]
        max_concurrent: usize,
        /// HARDMath JSONL dataset (default: vendored subset).
        #[arg(long)]
        data: Option<PathBuf>,
        /// Output summary path (default: <run_dir>/summary.json).
        #[arg(long)]
        out: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArgs,
        /// Resume a previous run: load the .partial.json file next to --out,
        /// skip already-completed problem IDs, and merge new results in.
        #[arg(long)]
        resume: bool,
        /// Reuse the warm agent pool across problems instead of respawning
        /// before each one: ~2x fewer Haiku turns, but trades per-problem
        /// isolation (a reused session carries prior context forward).
        #[arg(long)]
        no_reset: bool,
        /// Skip inline grading: save predictions only, to be scored later by the
        /// batched `grade` pass.
        #[arg(long)]
        no_grade: bool,
        /// Per-wave reply timeout in seconds.
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
    },
    /// Batch-grade saved predictions: one blind, shuffled judge call per problem
    /// scores every method against the gold (paired evaluation).
    Grade {
        /// Methods to grade together (must have <method>.seed<seed>.json in --dir).
        #[arg(long, value_enum, value_delimiter = ',', default_values = ["cot", "self-refine", "sc5", "debate"])]
        methods: Vec<bench::Method>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Directory holding the per-method prediction files.
        #[arg(long)]
        dir: PathBuf,
        /// HARDMath JSONL dataset (default: vendored subset).
        #[arg(long)]
        data: Option<PathBuf>,
        /// Backend to spawn the grader with.
        #[arg(long, default_value = "claude")]
        backend: String,
        /// Grader model: defaults to Sonnet for stronger instruction-following
        /// (the solvers under test stay on whatever they ran with).
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: Option<String>,
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
    },
    /// Exercise one building block end-to-end on a built-in tiny instance.
    DemoBlock {
        #[arg(long, value_enum)]
        block: Block,
        #[command(flatten)]
        model: ModelArgs,
        /// Per-wave reply timeout in seconds.
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
    },
    /// Kill leaked MASS agents (sessions named mass<run>-<role>-<slot>).
    Teardown {
        /// Only agents of this run id; default: every mass* agent.
        #[arg(long)]
        run: Option<String>,
    },
}

#[derive(Debug, clap::Args)]
struct ModelArgs {
    /// OMAR backend to spawn agents with.
    #[arg(long, default_value = "claude")]
    backend: String,
    /// Backend model. Defaults to Haiku 4.5, the validated MASS backbone.
    #[arg(long, default_value = "claude-haiku-4-5-20251001")]
    model: Option<String>,
}

impl ModelArgs {
    fn to_config(&self) -> ModelConfig {
        ModelConfig {
            backend: self.backend.clone(),
            model: self.model.clone(),
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run {
            topology,
            question,
            context_file,
            tests,
            aggregator,
            max_concurrent,
            model,
            timeout_secs,
        } => {
            let topology: TopologyConfig =
                serde_json::from_str(&topology).context("Invalid --topology JSON")?;
            let context = match context_file {
                Some(path) => Some(
                    std::fs::read_to_string(&path)
                        .with_context(|| format!("Failed to read {}", path.display()))?,
                ),
                None => None,
            };
            let task = TaskInstance {
                id: format!("adhoc-{}", &uuid::Uuid::new_v4().simple().to_string()[..6]),
                question,
                context,
                tests,
            };
            let result = run_once(
                topology,
                match aggregator {
                    CliAggregator::Rule => AggregatorMode::Rule,
                    CliAggregator::Llm => AggregatorMode::Llm,
                },
                model.to_config(),
                Duration::from_secs(timeout_secs),
                max_concurrent,
                &task,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Command::Bench {
            method,
            n,
            seed,
            max_concurrent,
            resume,
            no_reset,
            no_grade,
            data,
            out,
            model,
            timeout_secs,
        } => {
            let data = data.unwrap_or_else(default_dataset);
            // With --out the full summary is already persisted to that file; only
            // echo the JSON to stdout when there's no out file to read it from,
            // so resumable runs don't flood the log with placeholder score:0.0.
            let echo_json = out.is_none();
            let summary = bench::run_bench(bench::BenchArgs {
                method,
                data,
                n,
                seed,
                max_concurrent,
                resume,
                reset_each_problem: !no_reset,
                grade: !no_grade,
                model: model.to_config(),
                timeout: Duration::from_secs(timeout_secs),
                out,
            })?;
            if echo_json {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            }
            Ok(())
        }
        Command::Grade {
            methods,
            seed,
            dir,
            data,
            backend,
            model,
            timeout_secs,
        } => {
            let data = data.unwrap_or_else(default_dataset);
            grade::run_grade(grade::GradeArgs {
                seed,
                data,
                methods,
                dir,
                model: ModelConfig { backend, model },
                timeout: Duration::from_secs(timeout_secs),
            })?;
            Ok(())
        }
        Command::DemoBlock {
            block,
            model,
            timeout_secs,
        } => {
            let (topology, aggregator, task) = demo_instance(block);
            let result = run_once(
                topology,
                aggregator,
                model.to_config(),
                Duration::from_secs(timeout_secs),
                0,
                &task,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Command::Teardown { run } => teardown(run),
    }
}

fn run_once(
    topology: TopologyConfig,
    aggregator: AggregatorMode,
    model: ModelConfig,
    timeout: Duration,
    max_concurrent: usize,
    task: &TaskInstance,
) -> Result<runner::InstanceResult> {
    let mut runner = Runner::setup(RunnerOptions {
        topology,
        aggregator,
        model,
        timeout,
        max_concurrent,
        with_grader: false,
        reset_each_problem: true,
        run_root: None,
    })?;
    eprintln!(
        "[mass] run {} ready; results under {}",
        runner.run_id,
        runner.run_dir.root.display()
    );
    let result = runner.run_instance(task);
    runner.teardown()?;
    result
}

/// Built-in tiny instances that exercise each block's path end-to-end.
fn demo_instance(block: Block) -> (TopologyConfig, AggregatorMode, TaskInstance) {
    let math_task = |id: &str| TaskInstance {
        id: id.to_string(),
        question: "What is the sum of the first 10 positive integers?".to_string(),
        context: None,
        tests: None,
    };
    match block {
        Block::Aggregate => (
            TopologyConfig { aggregate: 3, ..Default::default() },
            AggregatorMode::Rule,
            math_task("demo-aggregate"),
        ),
        Block::Reflect => (
            TopologyConfig { reflect: 1, ..Default::default() },
            AggregatorMode::Rule,
            math_task("demo-reflect"),
        ),
        Block::Debate => (
            TopologyConfig { debate: 1, ..Default::default() },
            AggregatorMode::Rule,
            math_task("demo-debate"),
        ),
        Block::Summarize => (
            TopologyConfig { summarize: 1, ..Default::default() },
            AggregatorMode::Rule,
            TaskInstance {
                id: "demo-summarize".to_string(),
                question: "In which year did Ada Lovelace publish her notes on the Analytical Engine?".to_string(),
                context: Some(demo_long_context()),
                tests: None,
            },
        ),
        Block::Execute => (
            TopologyConfig { reflect: 1, execute: 1, ..Default::default() },
            AggregatorMode::Rule,
            TaskInstance {
                id: "demo-execute".to_string(),
                question: "Write a Python function `is_palindrome(s)` that returns True when the lowercased alphanumeric characters of s read the same forwards and backwards. Provide only the code inside <answer></answer> tags.".to_string(),
                context: None,
                tests: Some(
                    "assert is_palindrome('A man, a plan, a canal: Panama') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('') == True".to_string(),
                ),
            },
        ),
    }
}

fn demo_long_context() -> String {
    let filler = "The Analytical Engine was a proposed mechanical general-purpose computer designed by Charles Babbage. \
It incorporated an arithmetic logic unit, control flow in the form of conditional branching and loops, and integrated memory. \
Many unrelated details about Victorian engineering, brass gears, punched cards inspired by the Jacquard loom, and funding disputes with the British government filled the historical record. ";
    let key = "Ada Lovelace translated Luigi Menabrea's memoir on the engine and published her extensive notes, including what is regarded as the first computer program, in 1843. ";
    format!("{}{}{}", filler.repeat(30), key, filler.repeat(30))
}

fn default_dataset() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/hardmath_subset.jsonl");
    if manifest.is_file() {
        manifest
    } else {
        PathBuf::from("mass/data/hardmath_subset.jsonl")
    }
}

/// Kill leaked MASS agents by name prefix.
fn teardown(run: Option<String>) -> Result<()> {
    let prefix = match &run {
        Some(id) => format!("mass{}-", id),
        None => "mass".to_string(),
    };
    let mut mcp = omar::OmarClient::start()?;
    let agents = mcp.list_agents()?;
    let victims: Vec<String> = agents
        .into_iter()
        .filter(|name| name.starts_with(&prefix))
        .collect();
    if victims.is_empty() {
        println!("No MASS agents matching prefix '{prefix}'.");
        return Ok(());
    }
    mcp.log_justification(
        "omar-mass",
        "mass_teardown",
        &format!(
            "Cleaning up {} leaked MASS agents (prefix '{}').",
            victims.len(),
            prefix
        ),
    )?;
    for name in &victims {
        match mcp.kill_agent(name) {
            Ok(()) => println!("killed {name}"),
            Err(err) => println!("failed to kill {name}: {err}"),
        }
    }
    for (id, name) in mcp.list_projects()? {
        if name.starts_with("mass-") && mcp.complete_project(id).is_ok() {
            println!("completed project {name} ({id})");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_no_reset_flag_parses() {
        let on = Cli::parse_from(["omar-mass", "bench", "--method", "cot", "--no-reset"]);
        match on.command {
            Command::Bench { no_reset, .. } => assert!(no_reset),
            _ => panic!("expected bench"),
        }
        // Default keeps the per-problem respawn (statelessness) on.
        let off = Cli::parse_from(["omar-mass", "bench", "--method", "cot"]);
        match off.command {
            Command::Bench { no_reset, .. } => assert!(!no_reset),
            _ => panic!("expected bench"),
        }
    }
}
