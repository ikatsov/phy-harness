import numpy as np

from robot_manipulation_sim import UR5GripperEnv
from robot_manipulation_sim.env import map_normalized_actions


def test_env_steps_without_rgb():
    env = UR5GripperEnv(enable_rgb=False, seed=0)
    obs = env.reset(box_xy_noise=0.0)
    assert obs["images"] == {}
    assert obs["qpos"].size == env.model.nq
    for _ in range(10):
        env.step(env._home)
    assert env.data.time > 0


def test_map_normalized_actions():
    env = UR5GripperEnv(enable_rgb=False)
    env.reset()
    z = np.zeros(env.nu)
    ctrl = map_normalized_actions(z, env.model)
    assert ctrl.shape == (env.nu,)
