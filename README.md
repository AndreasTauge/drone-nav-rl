# drone-nav-rl

Reinforcement learning for quadrotor navigation with MuJoCo/MJX and PPO. Work in progress.

## Repository layout

```
src/drone_nav/        Python package
  envs/               MJX environments
  training/           PPO training (Brax)
  export/             Policy export (ONNX)
  assets/             MJCF models
configs/              Experiment configs (YAML)
scripts/              CLI entry points
tests/                pytest
ros2_ws/src/drone_nav_deploy/   C++ ROS 2 package for policy deployment
docs/                 Notes
media/                README media
```

## Setup

Tested with Ubuntu 26.04, an RTX 3070 (8 GB), NVIDIA driver 595 (CUDA 13.2), and Python 3.12. The Python stack is managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (user space, no sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv with Python 3.12 and install everything (incl. dev + logging extras)
uv sync --all-extras

# Verify: JAX sees the GPU, MuJoCo + MJX step, ONNX imports
uv run python -c "import jax; print(jax.devices())"
uv run pytest -v
uv run ruff check . && uv run ruff format --check .
```

On **macOS** the same `uv sync` installs CPU JAX (CUDA wheels are Linux-only), which is enough for viewing models
with `uv run python -m mujoco.viewer`.

**GPU memory.** By default JAX preallocates 75% of VRAM. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` for large runs,
or `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing the GPU.

### ROS 2

Uses system ROS 2 Lyrical Luth and Gazebo Jetty. Build with the system Python, **not** inside `.venv`:

```bash
deactivate 2>/dev/null; cd ros2_ws
source /opt/ros/lyrical/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## License

MIT. See [LICENSE](LICENSE).
