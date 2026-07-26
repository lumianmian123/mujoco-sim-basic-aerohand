from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from sim_aero_hand import default_scene_path


def main() -> None:
  model = mujoco.MjModel.from_xml_path(str(default_scene_path()))
  data = mujoco.MjData(model)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      mujoco.mj_step(model, data)
      viewer.sync()
      time.sleep(0.01)


if __name__ == "__main__":
  main()

