"""Environment smoke tests: imports, MuJoCo step, MJX step on the default JAX device."""

import jax
import mujoco
import numpy as np
import pytest
from mujoco import mjx

FALLING_BOX = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_package_imports():
    import brax  # noqa: F401
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

    import drone_nav

    assert drone_nav.__version__


def test_mujoco_step():
    model = mujoco.MjModel.from_xml_string(FALLING_BOX)
    data = mujoco.MjData(model)
    z0 = data.qpos[2]
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.qpos[2] < z0  # gravity pulls the box down


def test_mjx_step_matches_mujoco():
    model = mujoco.MjModel.from_xml_string(FALLING_BOX)
    mx = mjx.put_model(model)
    dx = mjx.put_data(model, mujoco.MjData(model))
    step = jax.jit(mjx.step)
    for _ in range(10):
        dx = step(mx, dx)

    ref = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, ref)
    np.testing.assert_allclose(np.asarray(dx.qpos), ref.qpos, atol=1e-4)


@pytest.mark.gpu
def test_mjx_runs_on_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("no CUDA GPU visible to JAX (expected on macOS)")
    model = mujoco.MjModel.from_xml_string(FALLING_BOX)
    mx = mjx.put_model(model)
    dx = mjx.put_data(model, mujoco.MjData(model))
    batch = jax.vmap(lambda _: dx)(np.arange(1024))
    batch = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))(mx, batch)
    assert batch.qpos.shape == (1024, 7)
    assert next(iter(batch.qpos.devices())).platform == "gpu"
