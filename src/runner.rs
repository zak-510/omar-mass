//! The MASS composer/runner.
//! Spawns one persistent OMAR agent per (role, chain-slot), routes block
//! waves through the file mailbox in order [summarize, reflect, debate,
//! aggregate] (plus execute on the predictor), parses the final <answer>,
//! and tears every agent down. See PLAN.md sections 5 and 6 for the design.

use crate::blocks::{self, CallSpec};
use crate::mailbox::{self, RunDir, WaitOutcome};
use crate::math;
use crate::omar::OmarClient;
use crate::prompts::PredictorKind;
use crate::protocol::{self, Envelope, Reply, Role};
use crate::topology::{AggregatorMode, SessionSpec, TopologyConfig};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Budget per readiness attempt. Must clear a slow cold-start: a local 7B
/// processing the ~7k-token charter takes ~100-120s, and wide topologies
/// share one model server so several charters prefill at once.
const READY_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(600);
/// How many times to respawn a laggard before failing the instance.
const MAX_READY_ATTEMPTS: usize = 4;
const POLL_INTERVAL: Duration = Duration::from_millis(2000);
/// Re-nudge a stuck request only well past any plausible solve time, so a
/// working agent that is still solving is never interrupted.
const DISPATCH_NUDGE_AFTER: Duration = Duration::from_secs(180);
const DISPATCH_NUDGE_EVERY: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// OMAR backend: claude | codex | cursor | opencode | agy.
    pub backend: String,
    /// Optional backend model override (spawn_agent `model`).
    pub model: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            backend: "claude".to_string(),
            model: None,
        }
    }
}

/// One task instance to push through the topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInstance {
    pub id: String,
    pub question: String,
    /// Long context for the Summarize block (unused for plain MATH).
    #[serde(default)]
    pub context: Option<String>,
    /// Public test cases for the Execute block (coding tasks).
    #[serde(default)]
    pub tests: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RunnerOptions {
    pub topology: TopologyConfig,
    pub aggregator: AggregatorMode,
    pub predictor: PredictorKind,
    pub model: ModelConfig,
    /// Per-wave reply timeout.
    pub timeout: Duration,
    /// Max agents resident at once (0 = unlimited). Below a wave's width, the
    /// wave runs in sequential batches over a small pool, resetting between
    /// them. Lets a memory-bound host run wide topologies like SC@5 on 16GB.
    pub max_concurrent: usize,
    /// Override the run-dir root (default $OMAR_DIR or $HOME/.omar, + mass/runs).
    pub run_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CallRecord {
    pub envelope_id: String,
    pub role: Role,
    pub slot: usize,
    pub round: usize,
    pub ok: bool,
    pub answer: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstanceResult {
    pub task_id: String,
    pub answer: Option<String>,
    pub llm_calls: usize,
    pub failures: usize,
    pub records: Vec<CallRecord>,
}

pub struct Runner {
    mcp: OmarClient,
    pub run_id: String,
    project_id: usize,
    pub run_dir: RunDir,
    opts: RunnerOptions,
    sessions: Vec<SessionSpec>,
    /// True when every agent holds only its fresh charter (just spawned or
    /// reset). The first problem uses this; later ones reset back to it.
    fresh: bool,
    torn_down: bool,
}

fn default_run_root() -> PathBuf {
    let omar_dir = std::env::var_os("OMAR_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".omar")
        });
    omar_dir.join("mass").join("runs")
}

/// Cap the session pool to max_concurrent agents per role (slot 0, the
/// aggregator, is always kept). 0 means unlimited.
fn cap_sessions(full: Vec<SessionSpec>, max_concurrent: usize) -> Vec<SessionSpec> {
    if max_concurrent == 0 {
        full
    } else {
        full.into_iter()
            .filter(|s| s.slot == 0 || s.slot <= max_concurrent)
            .collect()
    }
}

fn sanitize_id(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

impl Runner {
    /// Validate the topology, spawn the agent pool, wait until all are ready.
    pub fn setup(opts: RunnerOptions) -> Result<Runner> {
        opts.topology.validate()?;
        opts.topology.check_cap(opts.aggregator)?;

        let run_id: String = uuid::Uuid::new_v4().simple().to_string()[..4].to_string();
        let root = opts.run_root.clone().unwrap_or_else(default_run_root);
        let run_dir = RunDir::create(root.join(&run_id))?;
        let mut mcp = OmarClient::start()?;

        // Cap resident agents at max_concurrent; wider waves batch in dispatch.
        let sessions = cap_sessions(
            opts.topology.session_specs(opts.aggregator),
            opts.max_concurrent,
        );
        let project_id = mcp.add_project(&format!("mass-{run_id}"))?;
        mcp.log_justification(
            "omar-mass",
            "mass_spawn_topology",
            &format!(
                "Spawning {} MASS agents (topology {:?}, {} worst-case LLM calls/inference) for run {} to serve topology inference requests.",
                sessions.len(),
                opts.topology,
                opts.topology.llm_calls(opts.aggregator),
                run_id
            ),
        )?;

        let mut runner = Runner {
            mcp,
            run_id,
            project_id,
            run_dir,
            opts,
            sessions,
            fresh: true,
            torn_down: false,
        };

        // Tear down on any failure so a half-spawned run never leaks agents.
        if let Err(err) = runner.spawn_pool().and_then(|()| runner.ensure_ready()) {
            let _ = runner.teardown();
            return Err(err);
        }
        Ok(runner)
    }

    /// Respawn every agent with a fresh charter via OMAR's verified spawn path,
    /// so the protocol lands. This is why resets respawn rather than /clear.
    fn spawn_pool(&mut self) -> Result<()> {
        self.spawn_specs(&self.sessions.clone())
    }

    /// Respawn the given agents with fresh charters, killing any stale session
    /// of the same name and clearing its ready marker first.
    fn spawn_specs(&mut self, specs: &[SessionSpec]) -> Result<()> {
        // Kill any stale agent reusing one of our deterministic names.
        let existing = self.mcp.list_agents().unwrap_or_default();
        // Run from the invoking dir, not the run dir: backend CLIs gate unknown
        // folders behind a trust prompt that would deadlock the spawn. Mailbox
        // paths are absolute, so any working dir is fine.
        let workdir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .display()
            .to_string();
        for spec in specs {
            let name = spec.short_name(&self.run_id);
            if existing.iter().any(|a| a == &name) {
                let _ = self.mcp.kill_agent(&name);
            }
            // Drop any stale ready marker so we block on the fresh signal.
            let _ = std::fs::remove_file(self.run_dir.ready(&name));
            let charter = charter(&name, spec, &self.run_dir);
            // Keep the charter on disk so wait_ready can re-point a laggard at it.
            std::fs::write(self.run_dir.charter(&name), &charter)
                .with_context(|| format!("Failed to write charter for {name}"))?;
            self.mcp
                .spawn_agent(
                    &name,
                    self.project_id,
                    &charter,
                    &self.opts.model.backend,
                    self.opts.model.model.as_deref(),
                    &workdir,
                )
                .with_context(|| format!("Failed to spawn agent {name}"))?;
        }
        Ok(())
    }

    /// Block until every agent is ready, respawning laggards between attempts.
    /// A weak backend sometimes skips the touch handshake, and a respawn
    /// recovers it where waiting longer would not.
    fn ensure_ready(&mut self) -> Result<()> {
        for attempt in 1..=MAX_READY_ATTEMPTS {
            let missing = self.wait_ready_once(READY_ATTEMPT_TIMEOUT);
            if missing.is_empty() {
                return Ok(());
            }
            let laggards: Vec<SessionSpec> = missing.iter().map(|&i| self.sessions[i]).collect();
            let names: Vec<String> = laggards
                .iter()
                .map(|s| s.short_name(&self.run_id))
                .collect();
            if attempt == MAX_READY_ATTEMPTS {
                bail!(
                    "Agents never became ready after {MAX_READY_ATTEMPTS} attempts: {}",
                    names.join(", ")
                );
            }
            eprintln!(
                "[mass] readiness attempt {attempt}/{MAX_READY_ATTEMPTS} timed out for {}; respawning",
                names.join(", ")
            );
            self.spawn_specs(&laggards)?;
        }
        unreachable!("loop either returns or bails on the final attempt")
    }

    /// Poll until all agents are ready or timeout elapses, returning the
    /// session indices still missing. Re-points laggards at their charter once.
    fn wait_ready_once(&mut self, timeout: Duration) -> Vec<usize> {
        let paths: Vec<PathBuf> = self
            .sessions
            .iter()
            .map(|spec| self.run_dir.ready(&spec.short_name(&self.run_id)))
            .collect();
        // Ready files are empty markers, so poll for existence directly.
        let start = std::time::Instant::now();
        let mut nudged = vec![false; paths.len()];
        loop {
            let missing: Vec<usize> = paths
                .iter()
                .enumerate()
                .filter(|(_, p)| !p.exists())
                .map(|(i, _)| i)
                .collect();
            if missing.is_empty() || start.elapsed() >= timeout {
                return missing;
            }
            // Halfway through, re-point laggards at their on-disk charter in
            // case the spawn-time delivery was swallowed.
            if start.elapsed() >= timeout / 2 {
                for &i in &missing {
                    if !nudged[i] {
                        nudged[i] = true;
                        let name = self.sessions[i].short_name(&self.run_id);
                        let charter_path = self.run_dir.charter(&name);
                        let _ = self.mcp.send_input(
                            &name,
                            &format!(
                                "Your task: read the file {} and follow it exactly.",
                                charter_path.display()
                            ),
                        );
                    }
                }
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn agent_name(&self, role: Role, slot: usize) -> String {
        SessionSpec { role, slot }.short_name(&self.run_id)
    }

    /// Dispatch a wave, honoring max_concurrent. A wave that fits runs as one
    /// parallel batch; a wider wave runs in sequential batches with a pool
    /// reset between them. Returns replies in the original wave order.
    fn dispatch(
        &mut self,
        task_id: &str,
        wave: &[CallSpec],
        records: &mut Vec<CallRecord>,
    ) -> Vec<Option<String>> {
        let k = self.opts.max_concurrent;
        if k == 0 || wave.len() <= k {
            return self.dispatch_batch(task_id, wave, records);
        }
        let mut results: Vec<Option<String>> = Vec::with_capacity(wave.len());
        for (b, chunk) in wave.chunks(k).enumerate() {
            if b > 0 {
                // Fresh context for the next independent batch of samples.
                if let Err(err) = self.reset_agents() {
                    eprintln!("[mass] batch reset failed (continuing): {err:#}");
                }
            }
            // Map this batch onto the pool's slots 1..=k.
            let remapped: Vec<CallSpec> = chunk
                .iter()
                .enumerate()
                .map(|(i, c)| CallSpec {
                    slot: i + 1,
                    ..c.clone()
                })
                .collect();
            let mut batch = self.dispatch_batch(task_id, &remapped, records);
            results.append(&mut batch);
        }
        results
    }

    /// Run one wave in parallel: write all envelopes, notify all agents, then
    /// collect replies. None means send failure or timeout. Appends to records.
    fn dispatch_batch(
        &mut self,
        task_id: &str,
        wave: &[CallSpec],
        records: &mut Vec<CallRecord>,
    ) -> Vec<Option<String>> {
        let mut reply_paths = Vec::with_capacity(wave.len());
        let mut send_ok = Vec::with_capacity(wave.len());
        let mut ids = Vec::with_capacity(wave.len());
        for call in wave {
            let id = format!(
                "{}-{}-{}-r{}-{}",
                sanitize_id(task_id),
                call.role.token(),
                call.slot,
                call.round,
                &uuid::Uuid::new_v4().simple().to_string()[..8]
            );
            let receiver = self.agent_name(call.role, call.slot);
            let reply_path = self.run_dir.outbox(&id);
            let envelope = Envelope {
                id: id.clone(),
                sender: "runner".to_string(),
                receiver: receiver.clone(),
                timestamp_ns: mailbox::now_ns(),
                run_id: self.run_id.clone(),
                task_id: task_id.to_string(),
                role: call.role,
                round: call.round,
                payload: call.payload.clone(),
                reply_path: reply_path.display().to_string(),
            };
            let inbox = self.run_dir.inbox(&id);
            let sent = mailbox::write_json_atomic(&inbox, &envelope)
                .and_then(|()| {
                    self.mcp
                        .send_input(&receiver, &format!("MASS_REQUEST {}", inbox.display()))
                })
                .is_ok();
            reply_paths.push(reply_path);
            send_ok.push(sent);
            ids.push((id, inbox, receiver));
        }

        // Wait only on calls whose notification went out.
        let waited: Vec<PathBuf> = reply_paths
            .iter()
            .zip(&send_ok)
            .filter(|(_, ok)| **ok)
            .map(|(p, _)| p.clone())
            .collect();
        let wait_index: Vec<usize> = send_ok
            .iter()
            .enumerate()
            .filter(|(_, ok)| **ok)
            .map(|(i, _)| i)
            .collect();
        let mcp = &mut self.mcp;
        let outcomes = mailbox::wait_for_files(
            &waited,
            self.opts.timeout,
            POLL_INTERVAL,
            DISPATCH_NUDGE_AFTER,
            DISPATCH_NUDGE_EVERY,
            |k| {
                let (_, inbox, receiver) = &ids[wait_index[k]];
                let _ = mcp.send_input(receiver, &format!("MASS_REQUEST {}", inbox.display()));
            },
        );

        let mut results = vec![None; wave.len()];
        for (k, outcome) in outcomes.into_iter().enumerate() {
            if let WaitOutcome::Ready(raw) = outcome {
                results[wait_index[k]] = Some(Reply::parse(&raw).content);
            }
        }
        for (i, call) in wave.iter().enumerate() {
            records.push(CallRecord {
                envelope_id: ids[i].0.clone(),
                role: call.role,
                slot: call.slot,
                round: call.round,
                ok: results[i].is_some(),
                answer: results[i].as_deref().and_then(protocol::parse_answer),
            });
        }
        results
    }

    /// Reset the pool so the next problem is an independent, stateless run,
    /// matching the paper. We reuse persistent agents for speed, so without
    /// this they would carry prior transcripts forward and bias later answers.
    /// Kill and respawn through the verified path rather than /clear, which
    /// races the backend's popup and wedges the pane.
    fn reset_agents(&mut self) -> Result<()> {
        self.spawn_pool()?;
        self.ensure_ready()
    }

    /// Run one task through the topology and return the final answer. Also
    /// persisted under <run_dir>/results/<task_id>.json.
    pub fn run_instance(&mut self, task: &TaskInstance) -> Result<InstanceResult> {
        // Reset before each problem so it is independent. The first problem
        // already runs against the fresh charters written at setup.
        if !self.fresh {
            self.reset_agents()
                .with_context(|| format!("Failed to reset agents before task {}", task.id))?;
        }
        self.fresh = false;

        let cfg = self.opts.topology;
        let width = cfg.width();
        let question = task.question.clone();
        let mut records: Vec<CallRecord> = Vec::new();

        // -- Summarize ----------------------------------------------------
        let mut summaries: Vec<Option<String>> = vec![None; width];
        if cfg.summarize > 0 {
            if let Some(context) = &task.context {
                for round in 0..cfg.summarize {
                    let wave = blocks::summarize_wave(width, round, &question, context, &summaries);
                    let replies = self.dispatch(&task.id, &wave, &mut records);
                    for (call, reply) in wave.iter().zip(replies) {
                        if let Some(content) = reply {
                            summaries[call.slot - 1] = Some(content);
                        }
                    }
                }
            }
        }

        // -- Predict (the parallel part of Aggregate) ---------------------
        let wave = blocks::predict_wave(self.opts.predictor, width, &question, &summaries);
        let replies = self.dispatch(&task.id, &wave, &mut records);
        let mut texts: Vec<String> = vec![String::new(); width];
        let mut answers: Vec<String> = vec![String::new(); width];
        let mut alive: Vec<bool> = vec![false; width];
        for (call, reply) in wave.iter().zip(replies) {
            if let Some(content) = reply {
                let idx = call.slot - 1;
                answers[idx] = protocol::parse_answer(&content).unwrap_or_default();
                texts[idx] = content;
                alive[idx] = true;
            }
        }
        if !alive.iter().any(|&a| a) {
            bail!("All {} predictor calls failed for task {}", width, task.id);
        }

        // -- Execute (attaches to the predictor; real code execution) -----
        if cfg.execute > 0 {
            if let Some(tests) = &task.tests {
                let wave = blocks::execute_wave(&answers, &alive, tests);
                let replies = self.dispatch(&task.id, &wave, &mut records);
                for (call, reply) in wave.iter().zip(replies) {
                    if let Some(content) = reply {
                        let idx = call.slot - 1;
                        texts[idx] = format!("{}\n\nExecution result:\n{}", texts[idx], content);
                    }
                }
            }
        }

        // -- Reflect (reflector -> refiner, early stop on True) -----------
        let mut reflecting = alive.clone();
        for round in 0..cfg.reflect {
            if !reflecting.iter().any(|&a| a) {
                break;
            }
            let wave = blocks::reflect_wave(round, &question, &texts, &reflecting);
            let replies = self.dispatch(&task.id, &wave, &mut records);
            let mut reflections: Vec<Option<protocol::Reflection>> = vec![None; width];
            for (call, reply) in wave.iter().zip(replies) {
                let idx = call.slot - 1;
                match reply {
                    Some(content) => {
                        let reflection = protocol::parse_reflection(&content);
                        if reflection.correct {
                            reflecting[idx] = false; // stop criterion
                        } else {
                            reflections[idx] = Some(reflection);
                        }
                    }
                    None => reflecting[idx] = false, // reflector lost; keep answer
                }
            }
            let wave = blocks::refine_wave(round, &question, &answers, &reflections);
            if wave.is_empty() {
                break;
            }
            let replies = self.dispatch(&task.id, &wave, &mut records);
            for (call, reply) in wave.iter().zip(replies) {
                if let Some(content) = reply {
                    let idx = call.slot - 1;
                    if let Some(answer) = protocol::parse_answer(&content) {
                        answers[idx] = answer;
                        texts[idx] = content;
                    }
                }
            }
        }

        // -- Debate (fully connected rounds across chains) -----------------
        for round in 0..cfg.debate {
            let wave = blocks::debate_wave(round, &question, &texts, &alive);
            let replies = self.dispatch(&task.id, &wave, &mut records);
            for (call, reply) in wave.iter().zip(replies) {
                if let Some(content) = reply {
                    let idx = call.slot - 1;
                    if let Some(answer) = protocol::parse_answer(&content) {
                        answers[idx] = answer;
                        texts[idx] = content;
                    }
                }
            }
        }

        // -- Aggregate ------------------------------------------------------
        let live_answers: Vec<String> = answers
            .iter()
            .zip(&alive)
            .filter(|(a, &ok)| ok && !a.is_empty())
            .map(|(a, _)| a.clone())
            .collect();
        let final_answer = match self.opts.aggregator {
            AggregatorMode::Rule => math::majority_vote(&live_answers),
            AggregatorMode::Llm => {
                let live_texts: Vec<String> = texts
                    .iter()
                    .zip(&alive)
                    .filter(|(_, &ok)| ok)
                    .map(|(t, _)| t.clone())
                    .collect();
                let call = blocks::aggregate_call(&question, &live_texts);
                let replies = self.dispatch(&task.id, &[call], &mut records);
                replies[0]
                    .as_deref()
                    .and_then(protocol::parse_answer)
                    // Fall back to the rule vote if the LLM aggregator fails.
                    .or_else(|| math::majority_vote(&live_answers))
            }
        };

        let result = InstanceResult {
            task_id: task.id.clone(),
            answer: final_answer,
            llm_calls: records.len(),
            failures: records.iter().filter(|r| !r.ok).count(),
            records,
        };
        mailbox::write_json_atomic(&self.run_dir.result(&sanitize_id(&task.id)), &result)?;
        Ok(result)
    }

    /// Kill every agent of this run and complete the project. Idempotent;
    /// also called best-effort on drop.
    pub fn teardown(&mut self) -> Result<()> {
        if self.torn_down {
            return Ok(());
        }
        let _ = self.mcp.log_justification(
            "omar-mass",
            "mass_teardown",
            &format!(
                "Tearing down run {}: shutting down and killing all topology agents.",
                self.run_id
            ),
        );
        let mut errors = Vec::new();
        for spec in &self.sessions.clone() {
            let name = spec.short_name(&self.run_id);
            let _ = self.mcp.send_input(&name, "MASS_SHUTDOWN");
            if let Err(err) = self.mcp.kill_agent(&name) {
                errors.push(format!("{name}: {err}"));
            }
        }
        if let Err(err) = self.mcp.complete_project(self.project_id) {
            errors.push(format!("complete_project: {err}"));
        }
        self.torn_down = true;
        if errors.is_empty() {
            Ok(())
        } else {
            bail!("Teardown left residue: {}", errors.join("; "))
        }
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        if !self.torn_down {
            let _ = self.teardown();
        }
    }
}

/// The spawn-time charter for one agent: its role plus the mailbox protocol.
/// Role-specific behavior comes from each request's payload.
fn charter(name: &str, spec: &SessionSpec, run_dir: &RunDir) -> String {
    let ready = run_dir.ready(name);
    format!(
        "You are agent '{name}', a MASS topology worker with role '{role}'.\n\
         CRITICAL, how 'waiting' works: you are a PASSIVE request handler. Each MASS_REQUEST is delivered to you as a new message. To 'wait' you simply FINISH your turn and stop; the next message will wake you automatically. NEVER write a loop, NEVER run a command that blocks or polls for input (no `while`/`read`/`tail -f`/`sleep`/`inotifywait`/'message loop'), and NEVER try to watch a file or directory yourself. Running any such command makes you busy and unable to receive the next request, which will break the whole run. Act only when a message arrives, then stop.\n\
         Follow this protocol exactly:\n\
         1. First, create the empty file {ready} (e.g. with a single `touch` command). Then STOP and end your turn. Do nothing else (do not start any waiting/polling command) until a request message arrives.\n\
         2. Requests arrive as single lines of the form: MASS_REQUEST <absolute path to a JSON file>. Read that file. Its 'payload' field is your complete, self-contained instruction; follow it exactly and treat every request independently of earlier ones.\n\
         3. Write your reply to the exact file named in the request's 'reply_path' field, atomically: write '<reply_path>.tmp' first, then rename it to 'reply_path'. The reply file must be one JSON object: {{\"id\": \"<the request id>\", \"sender\": \"{name}\", \"content\": \"<your complete response>\"}}. Put your entire response, including any <answer></answer> tags the payload asks for, inside the 'content' string, properly JSON-escaped. Use your file-writing tool rather than shell echo to avoid quoting mistakes.\n\
         4. After writing the reply file, STOP and end your turn. Do not poll or loop; just wait passively for the next MASS_REQUEST message. If you receive a request you already answered, write the same reply file again.\n\
         5. Your task never completes on its own. Do NOT output [TASK COMPLETE] and do NOT notify your parent until you receive the line MASS_SHUTDOWN; when you do, follow your normal completion protocol.",
        name = name,
        role = spec.role,
        ready = ready.display(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_sessions_limits_physical_pool() {
        use crate::topology::{AggregatorMode, TopologyConfig};
        // SC@5: five parallel predictor chains.
        let sc5 = TopologyConfig {
            aggregate: 5,
            ..Default::default()
        };
        let full = sc5.session_specs(AggregatorMode::Rule);
        assert_eq!(full.len(), 5);
        // No cap leaves all five resident.
        assert_eq!(cap_sessions(full.clone(), 0).len(), 5);
        // Cap=1 keeps one predictor; the other 4 chains run as later batches.
        let capped = cap_sessions(full.clone(), 1);
        assert_eq!(capped.len(), 1);
        assert_eq!(capped[0].slot, 1);
        // Cap=2 keeps two.
        assert_eq!(cap_sessions(full, 2).len(), 2);
        // The LLM aggregator (slot 0) is always retained despite a tight cap.
        let mad = TopologyConfig {
            aggregate: 3,
            debate: 2,
            ..Default::default()
        };
        let capped = cap_sessions(mad.session_specs(AggregatorMode::Llm), 1);
        assert!(capped
            .iter()
            .any(|s| s.role == Role::Aggregator && s.slot == 0));
    }

    #[test]
    fn charter_contains_protocol_essentials() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = RunDir::create(tmp.path().join("r")).unwrap();
        let spec = SessionSpec {
            role: Role::Predictor,
            slot: 1,
        };
        let text = charter("massab12-pred-1", &spec, &run_dir);
        assert!(text.contains("MASS_REQUEST"));
        assert!(text.contains("MASS_SHUTDOWN"));
        assert!(text.contains("reply_path"));
        assert!(text.contains("ready"));
        assert!(text.contains("role 'pred'"));
        // The agent must stay passive, or it goes busy and misses requests.
        assert!(text.contains("NEVER write a loop"));
        assert!(text.contains("end your turn"));
    }

    #[test]
    fn sanitize_id_is_filename_safe() {
        assert_eq!(sanitize_id("test/algebra_1.json"), "test-algebra_1-json");
    }

    #[test]
    fn task_instance_parses_minimal_json() {
        let t: TaskInstance = serde_json::from_str(r#"{"id":"x","question":"1+1?"}"#).unwrap();
        assert!(t.context.is_none());
        assert!(t.tests.is_none());
    }
}
