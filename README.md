# ARC-AGI-3 RL scaffold

## Local setup and verification

The original supplied ZIP is preserved in base commit `809b718`.
Its source folders were extracted directly into the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Repairs: replay observations are stored lazily as packed bits, preserving
zero-padded history exactly. The default capacity is now 10,000 transitions
(625 MiB of observation payload at full capacity, plus metadata and batches),
versus approximately 97.7 GiB in the original 50,000-transition buffer.
Only binary observations are accepted. Sampling reconstructs float32 tensors.
This capacity change must be recorded in experiment configurations.

Training and submission clear history on explicit and game-over resets.
Automatic reset actions consume budget and incur the training action cost.
The initial reset remains excluded from the adapter's action counter; verify
this convention against the submission runner before comparing action budgets.
The adapter rejects invalid action indices and steps after an episode ends.

DQN checkpoints now contain plain configuration dictionaries and load with
weights-only deserialization. Older scaffold checkpoints containing pickled
DQNConfig objects are not supported by this loader. Submission honours the
requested device and does not allocate a replay buffer. Custom policy classes
passed to the factory must accept device and inference_only keywords.

Tests cover mocked adapter behaviour, reset observation parity, packed replay,
a CPU gradient update, checkpoint round trips, and missing scoring inputs.
Real-game execution, competition runner integration, and the current official
scoring methodology have not been verified. Local RHAE is a scaffold estimate,
not an official score; missing human baselines now raise an error.
The existing evaluation harness still needs genuine seed propagation and
validated action accounting before it can support multi-seed result claims.

## Original scaffold notes (historical, not independently verified)

# ARC-AGI-3 RL Assignment — working scaffold

COMS4061A/7071A · due 27 Oct 2026, 23:59 SAST

## What's in here, and what's actually verified

Everything in `environment/`, `rl_agents/`, and `evaluation/` is written
against the **real** ARC-AGI-3 interface — I cloned
`arcprize/ARC-AGI-3-Kaggle-Starter` and `arcprize/ARC-AGI-3-Agents` from
GitHub and `pip install arc-agi arcengine` from PyPI to read the actual
`Agent`, `FrameData`, `GameAction`, `GameState` classes rather than
guessing. `tests/` exercises the adapter and the RHAE scoring math against
mocked data and passes.

What is **not** verified here: this sandbox can't reach
`three.arcprize.org` (not on its network allowlist), so I could not
actually download a game, play it, or run a training loop against a live
environment. The DQN code is syntax-checked but not execution-tested
(this sandbox also ran out of disk space installing PyTorch's CUDA
dependencies — remove the `nvidia-*` packages or install a CPU wheel if
you hit the same thing locally). You'll need to run this for real, on
your machine (via the Kaggle starter's `make setup` / `make play-local`)
or on Kaggle, where you have API access.

## Design decisions (put these in the report — see docstrings for full reasoning)

- **Observation**: last 4 grids, one-hot over 16 colours, stacked on the
  channel axis → `(64, 64, 64)` tensor into a small CNN. Frame-stacking
  gives a feed-forward network a short memory window, the same trick
  classic Atari DQN uses.
- **Action space**: 6 simple actions (RESET, ACTION1–5, ACTION7) + a
  discretised click grid for ACTION6 (default 8×8 = 64 cells, not the
  full 4096 pixels) — an explicit, documented action abstraction, and
  itself a candidate ablation (try 16×16 later).
- **Action masking**: `FrameData.available_actions` → boolean mask,
  applied both when acting (epsilon-greedy only samples legal actions)
  and when bootstrapping the DQN target (illegal next-actions excluded
  from the max).
- **Episode boundary**: GAME_OVER triggers an internal soft RESET
  (matches the starter kit's own reference agent) rather than ending the
  episode; only WIN or the action budget ends it. Worth an ablation.
- **Reward shaping (training only)**: `+1` per level, `+5` on WIN,
  `-0.05` on death, `-0.002`/action. The brief explicitly says not to
  report shaped return as the ARC-AGI-3 score — `evaluation/rhae.py`
  computes the real metric separately.
- **RHAE scoring**: implemented per `docs.arcprize.org/methodology.md`
  (fetched 27 Sep 2026) — per-level score capped at **1.15**× human
  baseline (not 1.0), per-game score is a *level-index-weighted* average
  where unsolved levels still count in the denominator, headline score is
  the unweighted mean over games. Verified against the doc's own worked
  examples in `tests/test_rhae.py`. You still need to source the real
  per-level human-baseline action counts from the competition once you
  have API access — `human_baselines` is a placeholder dict.

## Directory layout

```
environment/arc_gym_env.py   Gymnasium adapter (the "if your RL library
                              requires Gymnasium, implement an adapter"
                              requirement)
rl_agents/dqn/                DQN baseline (course-covered algorithm A):
                              q_network.py, replay_buffer.py, dqn_agent.py
rl_agents/policy_agent.py     Wraps a trained checkpoint back into the
                              official Agent interface for submission
evaluation/rhae.py           Real RHAE scoring + multi-seed harness
tests/                        Wiring tests, no network needed
```

Named `rl_agents/`, not `agents/`, on purpose — the vendored
ARC-AGI-3-Agents framework already owns a top-level `agents` package.

## Six-week roadmap against the deliverable checklist

**Now → ~1 week (foundation)**
- [ ] Register the team on both Kaggle tracks (ARC-AGI-3 + Paper Track),
      same name/membership. *Only you can do this — needs your Kaggle
      accounts.*
- [ ] `git clone` the Kaggle starter, `make setup`, get an `ARC_API_KEY`,
      confirm `make play-local` runs the default random agent end-to-end.
      **This is checklist item 2** ("minimal end-to-end agent").
- [ ] Wire this scaffold in: drop `environment/`, `rl_agents/`,
      `evaluation/` into the starter kit's repo, point `agent/my_agent.py`
      at `rl_agents.policy_agent.make_policy_agent_class` once you have a
      checkpoint.
- [ ] Pick 3–4 local games for a train/dev split and 2–3 held out for a
      generalisation check (per the brief's leakage-prevention clause).

**Weeks 2–3 (baselines)**
- [ ] Get the DQN baseline training against a real game locally; sanity
      check against the random-policy reference.
- [ ] Baseline B (non-course algorithm) — see open decision below.
- [ ] Freeze an evaluation protocol (seeds, action budget, which games)
      before touching "improvements", per the brief.

**Weeks 3–5 (iterative improvements)**
- [ ] ≥2 substantive improvements per algorithm, each: limitation →
      hypothesis → intervention → result. Candidates already flagged in
      the code comments: masked exploration variants, click-grid
      resolution, Double DQN / dueling heads / prioritized replay for the
      DQN side, reward-shaping ablations, curriculum over games.
- [ ] Ablations isolating each improvement's contribution.
- [ ] ≥3 seeds per config, variability reported.

**Week 6 (writeup)**
- [ ] Final comparison, success/failure trajectories, RLC LaTeX report
      (8 pages), condensed <1500-word Kaggle write-up, `Kaggle-team-
      name.zip` with report + notebook + score evidence + contribution
      statement.

## Open decisions I need from you to keep going

1. **Baseline B (the "not covered in the course" algorithm)** — PPO via
   Stable-Baselines3 (fast to get correct, but you still need to explain
   its objective/updates in depth for the rubric's understanding
   criterion) vs. a custom PyTorch PPO (more implementation work, more
   clearly "yours", more marks-safe on the "prompted/reused model alone
   doesn't satisfy the requirement" clause).
2. **Where you're at on Kaggle/API access** — tells me whether the next
   useful thing is a `make play-local` smoke test on your machine, or
   whether I should build further (e.g. the PPO baseline, ablation
   configs) against the same verified interface while you sort that out.
