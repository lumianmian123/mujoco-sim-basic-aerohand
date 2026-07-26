"""Small MuJoCo wrapper that mimics the Aero Hand SDK style."""

from __future__ import annotations

from pathlib import Path

import mujoco

from aero_open_sdk.joints_to_actuations import JointsToActuationsModel

from mappings import actuation_array_to_sim_array


def default_scene_path() -> Path:
  return Path(__file__).resolve().parent.parent / "assets" / "scene_right.xml"


class SimAeroHand:
  def __init__(self, xml_path: str | Path | None = None):
    scene_path = Path(xml_path) if xml_path is not None else default_scene_path()
    self.model = mujoco.MjModel.from_xml_path(str(scene_path))
    self.data = mujoco.MjData(self.model)
    self.jta = JointsToActuationsModel()

    self.actuator_names = [
        "right_index_A_tendon",
        "right_middle_A_tendon",
        "right_ring_A_tendon",
        "right_pinky_A_tendon",
        "right_thumb_A_cmc_abd",
        "right_th1_A_tendon",
        "right_th2_A_tendon",
    ]
    self.actuator_ids = [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in self.actuator_names
    ]

  def set_actuations(self, actuations_deg: list[float]) -> None:
    sim_ctrl = actuation_array_to_sim_array(actuations_deg)
    for i, value in enumerate(sim_ctrl):
      self.data.ctrl[self.actuator_ids[i]] = value

  def set_joint_positions(self, joint_positions_deg: list[float]) -> None:
    assert len(joint_positions_deg) == 16
    actuations_deg = self.jta.hand_actuations(joint_positions_deg)
    self.set_actuations(actuations_deg)

  def step(self, n: int = 1) -> None:
    for _ in range(n):
      mujoco.mj_step(self.model, self.data)

  def reset(self) -> None:
    mujoco.mj_resetData(self.model, self.data)

