# Roadmap

The build is ordered by dependency, so that no module ever stacks more than one unfamiliar hard thing at a time. Each milestone has a clear "done and demonstrable" bar — the project favours finished-and-small over ambitious-and-unfinished, and each milestone is an independently showable unit.

Status key: **[ ]** planned · **[~]** in progress · **[x]** done.

## Foundation

- **[x] 1 · Scalar autodiff.** A scalar reverse-mode automatic-differentiation engine in Rust, standard library only. *Done bar:* analytic gradients match numerical (finite-difference) gradients to ~1e-8 on a composed expression.
- **[~] 2 · Tensor autodiff + layers.** Extend to tensors; implement network layers (linear, activations) and validate them on CPU. *Done bar:* a tiny network trains on a toy task (MNIST-scale) end to end.
- **[ ] 3 · GPU kernels.** Move the hot paths onto the RTX 3070 Ti with hand-written CubeCL kernels. *Done bar:* the same training runs on the GPU with matching results and a real speedup.

## The game-learning engine

- **[ ] 4 · World model (RSSM).** The DreamerV3-style latent world model, trained flat (no hierarchy yet). *Done bar:* it learns a validated environment (see below) from pixels.
- **[ ] 5 · Hierarchy (manager / worker).** A Director-style manager proposing latent goals and a worker achieving them, on top of the proven world model. *Done bar:* the hierarchical agent matches or beats the flat one on a long-horizon task — measured as an ablation, so the lift is attributable.
- **[ ] 6 · Tiered memory (the scientist).** The hand-built MemGPT-style memory and the LLM orchestration loop that studies runs and proposes experiments. *Done bar:* the scientist's notes measurably shorten the path on a new run.

## Validation strategy

The learning stack is validated on **Crafter** — an open-source 2D survival game that DreamerV3 is benchmarked on, and whose environment *pauses while the agent thinks* — **before** it meets screen capture on a live game. This isolates "is the learner correct?" from "is the perception/reward correct?", so that when a real game misbehaves, the cause is obvious. Crafter is a private engineering test; the public thesis stays with full commercial games.

The first commercial target is **Vampire Survivors**: fast episodes (death → restart) and an on-screen timer + gold count that make a clean reward signal (survival-per-tick plus a death penalty). Getting reward reliably from pixels is the linchpin risk.

## North-star research arc (stretch)

- **[ ] 7 · Skill discovery + recall.** Discover reusable skills during play, store them, and recall them on demand for a new game — the mechanism the transfer thesis is really about. The *recall* half is a known pattern (a skill library is a vector store with embedding retrieval — which the scientist's archival memory already is); the *discovery* half, under the screen-only constraint, is genuine open research. Strictly downstream of the full foundation — the marquee experiment, never a starting point.

## Building in public

Each milestone maps to a chapter of the video build-log, including the parts that break. The commit history is meant to read as the story of the engine being built from nothing.
