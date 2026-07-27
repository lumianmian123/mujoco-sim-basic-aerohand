# Copyright 2026 TetherIA, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Play an Aero Hand PPO params file in MuJoCo Playground."""

import functools

from absl import app
from absl import flags
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as ppo_train
import jax
import mediapy as media
import mujoco

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params


_ENV_NAME = flags.DEFINE_string(
    "env_name", "AeroCubeRotateZAxis", "MuJoCo Playground environment name."
)
_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    None,
    "Path to the Brax params file saved with brax.io.model.save_params.",
)
_OUTPUT = flags.DEFINE_string(
    "output", "aero_policy_rollout.mp4", "Output video path."
)
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 500, "Rollout length.")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")


def _make_inference_fn(env_name: str, env):
  ppo_params = manipulation_params.brax_ppo_config(env_name)
  network_factory_config = ppo_params.get("network_factory", {})
  del ppo_params["network_factory"]

  if "num_timesteps" in ppo_params:
    del ppo_params["num_timesteps"]

  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **network_factory_config
  )

  make_inference_fn, _, _ = ppo_train(
      environment=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      network_factory=network_factory,
      num_timesteps=0,
      seed=_SEED.value,
      **ppo_params,
  )
  return make_inference_fn


def main(argv):
  del argv

  if _MODEL_PATH.value is None:
    raise ValueError("Please pass --model_path /path/to/policy_params")

  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env = registry.load(_ENV_NAME.value, config=env_cfg)
  make_inference_fn = _make_inference_fn(_ENV_NAME.value, env)

  params = model.load_params(_MODEL_PATH.value)
  inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

  rng = jax.random.PRNGKey(_SEED.value)
  rng, reset_rng = jax.random.split(rng)
  reset_fn = jax.jit(env.reset)
  step_fn = jax.jit(env.step)
  state = reset_fn(reset_rng)

  rollout = []
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  for step in range(_EPISODE_LENGTH.value):
    rng, act_rng = jax.random.split(rng)
    action = inference_fn(state.obs, act_rng)[0]
    state = step_fn(state, action)
    rollout.append(state)

    if bool(state.done):
      print(f"Episode terminated at step {step}.")
      break

  fps = 1.0 / env.dt / 2
  frames = env.render(
      rollout[::2],
      height=480,
      width=640,
      scene_option=scene_option,
  )
  media.write_video(_OUTPUT.value, frames, fps=fps)
  print(f"Rollout video saved to {_OUTPUT.value}.")


if __name__ == "__main__":
  app.run(main)
