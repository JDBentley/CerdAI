## What this is

CERD is an agent that learns to play commercial video games the way a person does: it watches the screen and presses keys. It does **not** read game memory or call any game API — perception is raw pixels, action is emitted input.

Under the hood it fuses three ideas:

- a **world model** (DreamerV3-style) for fast, within-game control learned by imagining futures in a latent space,
- a **hierarchy** (Director-style manager/worker) — a "chain of command" where a commander sets goals and a soldier executes them,
- a **tiered memory system** (MemGPT-style) run by a separate "scientist" that studies each run and carries knowledge across experiments.

The research question driving the whole project: **does learning one game make the next one faster?**

## The point: no frameworks

The entire project is written by hand. There is no PyTorch, no Gym, no RL library — not even a borrowed autodiff or tensor crate. The goal is to understand every component from first principles, down to the math that makes a neural network learn.

| Written by hand (this project) | Standing on (the floor — not rebuilt) |
| --- | --- |
| automatic-differentiation engine | CUDA driver + GPU |
| neural-network layers | operating-system capture / input APIs |
| the Dreamer world model | a local LLM runtime (spoken to over HTTP) |
| the hierarchy (manager / worker) | SQLite |
| the tiered memory system | OBS (screen recording) |
| the GPU kernels (in CubeCL) | *(optionally)* an OCR engine |

"By hand" always terminates somewhere — but everything in the left column is code written for this project, not a library imported into it.

## Architecture

Three machines, each with a job:

- **Windows desktop (RTX 3070 Ti)** — the *player*: captures the screen, runs the policy, emits input, and trains the world model and hierarchy. The fast loop.
- **MacBook Pro** — the *scientist*: runs a local LLM and the hand-built tiered memory, studies each run, and proposes the next experiment. The slow loop.
- **HP 800 Mini** — the *research server*: experiment database, event API, metrics dashboard, replay index, shared storage.

Full detail — the data flow, the three hierarchies, the crate map, and the reference architectures — is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Roadmap

The build is dependency-ordered so that no module stacks more than one new hard thing at a time:

1. **Scalar autodiff** ← *in progress*
2. Tensor autodiff + CPU-validated layers
3. GPU kernels (CubeCL)
4. World model (RSSM)
5. Hierarchy (manager / worker)
6. Tiered memory (the scientist)
7. *(stretch / research)* skill discovery + recall

Milestone detail and status live in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Repository layout

```
cerd/
├── Cargo.toml            workspace root (dependency-free by design)
├── rust-toolchain.toml   pinned stable toolchain + rustfmt/clippy
├── docs/
│   ├── ARCHITECTURE.md   the three-node system, crate map, data flow
│   └── ROADMAP.md        build sequence, milestone status, validation plan
└── crates/
    └── autodiff/         milestone 1 — the scalar autodiff engine
```

This is a **Cargo workspace**. It starts with a single crate and gains new members (`tensor`, `nn`, `worldmodel`, `hrl`, `perception`, …) as each milestone is built — the planned crate map is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Crates are added when they hold real, working code, not before.

## Building

Requires a Rust toolchain (installed via [rustup](https://rustup.rs)); the version is pinned in `rust-toolchain.toml`.

```sh
cargo build            # build the workspace
cargo test             # run the tests
cargo run -p cerd-autodiff --example demo   # (once the engine + demo exist)
```

Only the `autodiff` crate exists so far, and its implementation is written out step by step in the series rather than committed pre-built.

## The build-log

This repository is the code companion to a video series that builds the whole system from nothing — including the parts that break on camera. The commit history is meant to be read as a story: each milestone is a chapter.

## License

[MIT](LICENSE) © 2026 Jesse Bentley
