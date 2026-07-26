from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from sim_aero_hand import SimAeroHand


def main() -> None:
  hand = SimAeroHand()

  open_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  close_pose = [0.0, 0.0, 150.0, 210.0, 210.0, 210.0, 210.0]

  with mujoco.viewer.launch_passive(hand.model, hand.data) as viewer:
    while viewer.is_running():
      for t in range(101):
        alpha = t / 100.0
        actuations = [
            open_pose[i] + alpha * (close_pose[i] - open_pose[i])
            for i in range(7)
        ]
        hand.set_actuations(actuations)
        hand.step(5)
        viewer.sync()
        time.sleep(0.5)

      for t in range(101):
        alpha = t / 100.0
        actuations = [
            close_pose[i] + alpha * (open_pose[i] - close_pose[i])
            for i in range(7)
        ]
        hand.set_actuations(actuations)
        hand.step(5)
        viewer.sync()
        time.sleep(0.5)


if __name__ == "__main__":
  main()
