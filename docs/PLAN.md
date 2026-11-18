# Plan

Work top to bottom. Each task has a **Definition of Done (DoD)** that should be checkable in a few minutes.
Tick boxes as you go. Keep experiment numbers in the results table in the README.

Conventions to settle in M0 and keep everywhere (sim, ONNX, ROS node):
- **Frames:** MuJoCo world = ENU, body = FLU. PX4 uses NED/FRD. Convert only at the ROS↔PX4 boundary.
- **Action space:** decide early. Collective thrust + body rates (CTBR) maps cleanly to PX4 offboard
  `VehicleRatesSetpoint` and transfers better than raw motor commands. It is the recommended choice.
- **Control rate:** e.g. policy at 50 Hz, physics at 500 Hz (10 substeps).

---

## M0: Foundations

- [ ] **0.1 Quadrotor MJCF.** Write `src/drone_nav/assets/quadrotor.xml` (Crazyflie-scale or a generic
  ~1 kg X-quad matching the PX4 SITL `x500` you'll fly later. Pick one and note its mass, inertia and arm length).
  Model thrust/torque as actuators on the body (or 4 rotor sites).
  *DoD:* loads in `mujoco.viewer` on the Mac; with hover thrust = m·g it stays within 1 cm for 2 s in plain MuJoCo.
- [ ] **0.2 Physical parameters doc.** `docs/vehicle.md` with mass, inertia, thrust/torque coefficients,
  motor time constant and max thrust, with sources.
  *DoD:* every number in the MJCF is traceable to this table.
- [ ] **0.3 MJX parity.** Same open-loop action sequence in MuJoCo vs MJX.
  *DoD:* a test asserts the state trajectories match to tolerance over 1 s.
- [ ] **0.4 Throughput benchmark.** `scripts/bench_mjx.py`: steps/sec vs num_envs (256…8192) on the 3070.
  *DoD:* table in `docs/notes.md`; chosen `num_envs` that fits in 8 GB with headroom.

## M1: Hover and waypoint tracking (PPO, MJX)

- [ ] **1.1 Env API.** `envs/hover.py` implementing Brax's `PipelineEnv`/MJX env interface:
  `reset`, `step`, observation (pos error, linear vel, rotation as 6D or quat, angular vel, last action),
  action in CTBR (or motor) space, termination (crash, out of bounds, tilt > limit).
  *DoD:* `jax.jit(env.step)` + `jax.vmap` over 4096 envs runs; random-action rollout has no NaNs (test).
- [ ] **1.2 Reward.** Position error, velocity, action-rate and tilt penalties, plus an alive bonus. Each term logged separately.
  *DoD:* reward terms appear in TensorBoard; a test checks reward is maximal at the goal with zero velocity.
- [ ] **1.3 Training script + config.** `scripts/train.py --config configs/hover.yaml` using `brax.training.agents.ppo`;
  checkpointing to `checkpoints/<run>/`, TensorBoard logging, seed in config.
  *DoD:* one command trains; rerunning with the same seed gives the same curve (±noise from XLA nondeterminism).
- [ ] **1.4 Hover works.** *DoD:* the policy holds hover within 5 cm RMS from random initial states (±1 m, ±30° tilt)
  in ≥ 95% of 1000 eval episodes; training < 15 min on the 3070.
- [ ] **1.5 Waypoint tracking.** Random goals, then waypoint sequences.
  *DoD:* reaches 3D goals within 10 cm in ≥ 95% of eval episodes; eval script prints the metric table.
- [ ] **1.6 Render/eval tooling.** `scripts/eval.py` (metrics on held-out seeds) and `scripts/render.py` (MP4/GIF via mediapy,
  headless with `MUJOCO_GL=egl`).
  *DoD:* first GIF in `media/`, embedded in README.

## M2: Obstacle avoidance and domain randomization

- [ ] **2.1 Procedural layouts.** Pillar forest (density, radius ranges) and gates, generated per-env from a seed
  (fixed max obstacle count, inactive ones parked far away, so shapes stay static for JIT).
  *DoD:* train/test seed splits defined; a render shows 4 distinct layouts.
- [ ] **2.2 Range/depth observations.** Start with N ray casts (`mjx.ray`) in a horizontal fan + a few elevation rows;
  optionally a low-res depth image later.
  *DoD:* ray distances match `mujoco.mj_ray` in plain MuJoCo (test); env step still ≥ X steps/s (record it).
- [ ] **2.3 Navigation reward + curriculum.** Progress-to-goal, collision penalty, smoothness; curriculum on obstacle density.
  *DoD:* ≥ 80% success on unseen layouts at medium density (no DR yet).
- [ ] **2.4 Domain randomization.** Per-episode mass (±20%), inertia, thrust coefficient, motor time constant/delay
  (action buffer of k steps), observation noise + dropout on rays, initial-state ranges. All configurable in YAML.
  *DoD:* each DR term can be toggled; a test verifies sampled params lie in configured ranges.
- [ ] **2.5 Evaluation protocol.** Fixed held-out layout set (e.g. 500 seeds × 3 densities); metrics: success, collision,
  timeout, time-to-goal, path length. Robustness sweep over out-of-range mass/delay.
  *DoD:* `scripts/eval.py` writes a CSV + markdown table; README results rows filled for DR and no-DR (ablation).
- [ ] **2.6 Demo media.** *DoD:* GIF of the policy threading the forest, at the top of the README.

## M3: Deployment (ONNX → C++ ROS 2 → PX4 SITL)

- [ ] **3.1 ONNX export.** `export/onnx.py`: export the deterministic actor (mean action) with obs-normalization baked in
  (jax2onnx, or rebuild the MLP in onnx/torch from the Brax params).
  *DoD:* onnxruntime output vs JAX policy max abs diff < 1e-5 on 10k random observations (test).
- [ ] **3.2 Install PX4 + bridge** (needs sudo/apt for deps; check Gazebo Jetty compatibility first).
  PX4-Autopilot (`make px4_sitl gz_x500`), Micro XRCE-DDS Agent, and `px4_msgs` built into `ros2_ws`
  (matching the PX4 release branch).
  *DoD:* `ros2 topic echo /fmu/out/vehicle_odometry` shows data while SITL runs; versions recorded in `docs/px4_setup.md`.
- [ ] **3.3 C++ policy node.** `drone_nav_deploy`: subscribes to PX4 odometry (+ range sensor topic), builds the
  observation exactly as in training (NED→ENU conversion, same normalization), runs ONNX Runtime C++ API,
  publishes `OffboardControlMode` + `VehicleRatesSetpoint` (CTBR). Fixed-rate timer, no allocations in the loop.
  *DoD:* offline unit test (gtest): same obs vectors → same actions as Python onnxruntime; loop latency p99 < 2 ms logged.
- [ ] **3.4 Safety layer.** Arming/offboard state machine, watchdog (stale odometry → hold/land), action clamps, geofence.
  *DoD:* killing the odometry topic triggers a failsafe in SITL (demonstrated in a log).
- [ ] **3.5 Sim-to-sim: waypoints.** Fly waypoint tracking in Gazebo + PX4 SITL.
  *DoD:* tracking error metric comparable to MJX (report both in README).
- [ ] **3.6 Sim-to-sim: obstacles.** Gazebo world with pillars generated from the same seeds; lidar/range sensor bridged
  to ROS 2, matched to the training ray layout.
  *DoD:* success rate on ≥ 20 SITL layouts in the README table; side-by-side video MJX vs Gazebo.
- [ ] **3.7 Sim-to-sim gap analysis.** Where does it break (delay, thrust model, sensor differences)? Fix via DR, retrain.
  *DoD:* short section in README/`docs/` with before/after numbers.

## M4 (optional): Hardware

- [ ] **4.1 Platform choice.** Crazyflie 2.1 (+ Flow deck / Multi-ranger deck) with Crazyswarm2 (ROS 2), or a small PX4 quad.
  *DoD:* documented decision, re-identified vehicle params, retrained policy with matching dynamics.
- [ ] **4.2 Tethered/hover test.** *DoD:* stable hover video with the learned policy and safety pilot override.
- [ ] **4.3 Obstacle course.** *DoD:* real-flight success rate on a few physical layouts + video.

## Polish (throughout)

- [ ] CI (GitHub Actions): ruff + CPU pytest on push.
- [ ] README: lead GIF, results table, one architecture diagram (training → ONNX → ROS 2 → PX4).
- [ ] Reproducibility: every number in the README produced by a single command listed next to it.
