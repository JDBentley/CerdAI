# Architecture

This document describes the intended shape of CerdAI. Much of it is not built yet — the [roadmap](ROADMAP.md) tracks what is. It is written so a visitor can understand the system before most of it exists.

## The distributed system

CerdAI runs across three machines, split by timescale.

### Windows desktop — RTX 3070 Ti — the player (fast loop)

Everything that must happen in real time, kept local because latency matters.

- **Capture** — grabs frames (and audio) from the running game via OS capture APIs.
- **Adapter** — the per-game glue: screen regions, action space, episode start/stop, reset procedure. This is an *adapter*, not a "wrapper": it turns a game into a `step`/`reset` interface built purely from capture + input + OCR, and it never reads hidden game state.
- **Perception** — downscales frames to the small resolution the world model consumes, and reads the on-screen HUD (timer, score, health).
- **Reward estimator** — turns visible HUD state and events into a scalar reward.
- **Actor / policy runtime** — runs the network's forward pass at a fixed 10–20 Hz to choose actions.
- **Input emitter** — translates the chosen action into keyboard / mouse / controller events.
- **Trainer** — the hand-built autodiff engine, layers, world model, and hierarchy; drains the replay buffer and updates weights on the GPU.
- **Replay buffer** — stores trajectories for the trainer to sample.
- **Run logger** — writes structured run artifacts and triggers clip saves.

The actor and trainer run **concurrently**: one loop acts at a fixed rate (repeating its last action on frames it isn't ready), while a separate trainer consumes the replay buffer and periodically pushes fresh weights to the actor. This decouples training from real-time control.

### MacBook Pro — the scientist (slow loop)

After-the-fact analysis and cross-experiment memory. Not in the millisecond path.

- **LLM runtime** — a local language model, spoken to over HTTP.
- **Scientist orchestrator** — after each run, diagnoses failures, compares against past runs, and proposes the next experiment.
- **Tiered memory** — a hand-built MemGPT: *core* (compact working context), *recall* (recent runs), *archival* (embeddings + similarity search over all past runs).
- **Note generator** — writes human-readable markdown design notes.

### HP 800 Mini — the research server

- **Experiment database** (SQLite → Postgres later) — run metadata, metrics, failure tags.
- **Event / API server** — receives run summaries from the desktop, serves them to the Mac.
- **Metrics dashboard** — reward trends, survival curves, the transfer matrix.
- **Replay index** — a catalog of clip / checkpoint paths so large artifacts are referenced, not moved.
- **Artifact references** — stores paths and metadata for checkpoints, clips, logs, and replay exports kept on the NAS.

### NAS - durable artifact storage

Stores large and ling-lived project artifacts outside the compute and coordination machines.

- **Checkpoints** - saved model and optimizer state.
- **Experimental configurations** - reproducible settings for completed and queued runs.
- **Logs and metrics history** - durable records used for analysis and comparison.
- **Replay exports** - stored trajectories and selected replay data.
- **Recorded episode and training clips** - source material for review and video production.
- **Scientist reports** - experiment analyses and recommendations.

### Windows Laptop - Livestream Production

Handles eventual stream productiona fter the agent has meaningful training activity to show.

- **Video ingest** - recieves the training desktops display through a capture card.
- **OBS Production** - assembles the stream layout and recorded output.
- **Dashboard rendering** - displays the metrics dashboard as a browser source.
- **Encoding and broadcast** - handles stream encoding and delivery.
- **Workload isolation** - keeps recording and streaming overhead off the training desktop.


### Data flow

Desktop → 800 Mini: small JSON run summaries and metrics over HTTP.
Desktop → NAS: checkpoints, clips, logs, replay exports, and other large artifacts.
800 Mini → Mac: the scientist pulls summaries and selected clips.
Mac → Desktop: the proposed next-experiment config flows back to the trainer, closing the loop.

## The learning system — three models, fused

- **World model (DreamerV3-style / RSSM).** Fast within-game control. Learns latent dynamics from pixels and trains an actor-critic entirely by imagining rollouts in that latent space. Operates on small downscaled frames, so 8 GB of VRAM is workable.
- **Hierarchy (Director-style).** A manager proposes latent goals inside the world model's latent space; a worker learns to achieve them. A "chain of command" — the manager thinks in seconds ("clear this wave"), the worker in frames ("dodge, aim, move"). Begins at two levels; the intended end state is a three-level temporal abstraction (subgoals feeding subgoals at progressively coarser timescales).
- **Tiered memory (MemGPT-style), in the scientist.** Slow cross-experiment research memory that lets accumulated knowledge carry from one experiment — and one game — to the next.

### Three orthogonal hierarchies

These are easy to conflate; they are different axes:

- **Control abstraction** — commander / soldier (manager / worker), *inside the player*.
- **Timescale** — fast control vs. slow research (player vs. scientist).
- **Memory tiers** — core / recall / archival, *inside the scientist*.

The public framing is **commander / soldier / scientist**: three roles, two kinds — a control hierarchy plus a research overseer.

### The transfer question

The research core is whether learning transfers across games, and it separates into two things the experiments are designed to tell apart: **representation transfer** (do the world model's learned latents/weights help on a new game?) and **knowledge/skill transfer** (does the scientist's accumulated understanding bootstrap a new game faster?). Attribution comes from ablations — flat vs. hierarchical, with-memory vs. without.

## Planned crate map

The workspace grows as milestones land. Crates are added only when they contain real, working code.

| Crate | Node | Role | Status |
| --- | --- | --- | --- |
| `autodiff` | desktop | scalar reverse-mode autodiff | **complete** |
| `tensor` | desktop | tensor autodiff | planned | **in progress** |
| `nn` | desktop | network layers (linear, conv, GRU, …) | planned |
| `kernels` | desktop | hand-written GPU kernels (CubeCL) | planned |
| `worldmodel` | desktop | the RSSM world model | planned |
| `hrl` | desktop | manager / worker hierarchy | planned |
| `perception` | desktop | frame downscaling + HUD reading | planned |
| `reward` | desktop | reward estimation from pixels | planned |
| `adapter` | desktop | per-game `step`/`reset` adapters | planned |
| `actor` | desktop | real-time policy runtime + input | planned |
| `trainer` | desktop | training loop orchestration | planned |
| `scientist` | mac | LLM orchestration loop | planned |
| `memory` | mac | tiered MemGPT-style memory | planned |
| `server` | 800 mini | event API + experiment DB | planned |
| `dashboard` | 800 mini | metrics + transfer matrix | planned |

(Names are indicative and may consolidate as the code takes shape.)

## The by-hand boundary

Written for this project: the autodiff engine, the layers, the world model, the hierarchy, perception and reward, the adapters, input, the scientist orchestration and tiered memory, the event server and dashboard, the run logger — and the GPU kernels (in Rust via CubeCL, used standalone as a kernel-authoring tool, not as a framework).

Stood on but not rebuilt: the CUDA driver and GPU, OS capture/input APIs, a local LLM runtime on a socket, SQLite, OBS, and optionally an OCR engine.

Nothing in the first list wraps PyTorch, Gym, Burn, or a memory framework — the shapes exist because reinforcement learning and memory management require them, but the implementations are all written here.

## Reference architectures

Read-and-understand references, not dependencies (the by-hand constraint rules out importing them):

- **Director** — *Deep Hierarchical Planning from Pixels* — hierarchy over a Dreamer world model (arXiv:2206.04114).
- **HIEROS** — multi-level hierarchical imagination, the >2-level extension (arXiv:2310.05167).
- **Voyager** — LLM skill library + retrieval, lifelong learning (arXiv:2305.16291).
- **DreamerV3** — the world-model base (Hafner et al., Nature 2025).
- **MemGPT** — the tiered-memory pattern.
