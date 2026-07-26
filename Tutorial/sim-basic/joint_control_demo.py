from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from sim_aero_hand import SimAeroHand


def main() -> None:
  hand = SimAeroHand()

  poses = [
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 20.0, 25.0, 20.0, 20.0, 25.0, 20.0, 20.0, 25.0, 20.0, 20.0, 25.0, 20.0],
      [0.0, 0.0, 0.0, 0.0, 45.0, 55.0, 45.0, 45.0, 55.0, 45.0, 45.0, 55.0, 45.0, 45.0, 55.0, 45.0],
      [0.0, 0.0, 0.0, 0.0, 70.0, 80.0, 70.0, 70.0, 80.0, 70.0, 70.0, 80.0, 70.0, 70.0, 80.0, 70.0],
  ]

  with mujoco.viewer.launch_passive(hand.model, hand.data) as viewer:
    while viewer.is_running():
      for pose in poses:
        hand.set_joint_positions(pose)
        hand.step(10)
        viewer.sync()
        time.sleep(1)


if __name__ == "__main__":
  main()
