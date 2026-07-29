"""Local actuator mapping helpers for the AEROHAND MuJoCo scene."""

from __future__ import annotations

from aero_open_sdk.aero_hand_constants import AeroHandConstants
from aero_open_sdk.joints_to_actuations import JointsToActuationsModel

PI = 3.141592653589793
MOTOR_PULLEY_RADIUS = 9.000

SIM_RANGE = [
    (0.0617776, 0.107723),
    (0.0621875, 0.1084),
    (0.0616733, 0.10775),
    (0.0637823, 0.109504),
    (-0.0254462, 1.77858),
    (0.026941, 0.0382787),
    (0.0839985, 0.110133),
]

ACTUATIONS_LOWER_LIMITS = AeroHandConstants.actuation_lower_limits
ACTUATIONS_UPPER_LIMITS = AeroHandConstants.actuation_upper_limits

ACTUATION_RANGE = [
    (ACTUATIONS_LOWER_LIMITS[0], ACTUATIONS_UPPER_LIMITS[0]),
    (ACTUATIONS_LOWER_LIMITS[1], ACTUATIONS_UPPER_LIMITS[1]),
    (ACTUATIONS_LOWER_LIMITS[2], ACTUATIONS_UPPER_LIMITS[2]),
    (ACTUATIONS_LOWER_LIMITS[3], ACTUATIONS_UPPER_LIMITS[3]),
    (ACTUATIONS_LOWER_LIMITS[4], ACTUATIONS_UPPER_LIMITS[4]),
    (ACTUATIONS_LOWER_LIMITS[5], ACTUATIONS_UPPER_LIMITS[5]),
    (ACTUATIONS_LOWER_LIMITS[6], ACTUATIONS_UPPER_LIMITS[6]),
]

THUMB_ABD_ACTUATION = 0
THUMB_FLEX_ACTUATION = 1
THUMB_MCP_ACTUATION = 2
FINGER_IDX_ACTUATION = 3
FINGER_MIDDLE_ACTUATION = 4
FINGER_RING_ACTUATION = 5
FINGER_PINKY_ACTUATION = 6

THUMB_ABD_SIM = 4
THUMB_FLEX_SIM = 5
THUMB_MCP_SIM = 6
FINGER_IDX_SIM = 0
FINGER_MIDDLE_SIM = 1
FINGER_RING_SIM = 2
FINGER_PINKY_SIM = 3


def _clamp(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, value))


def sim_to_actuation_forward(
    x: float, lo: float, hi: float, min_u: float, max_u: float
) -> float:
  x = _clamp(x, lo, hi)
  t = (x - lo) / (hi - lo)
  return min_u + t * (max_u - min_u)


def sim_to_actuation_reverse(
    x: float, lo: float, hi: float, min_u: float, max_u: float
) -> float:
  x = _clamp(x, lo, hi)
  t = (hi - x) / (hi - lo)
  return min_u + t * (max_u - min_u)


def actuation_to_sim_reverse(
        u: float, lo: float, hi: float, min_u: float, max_u: float
) -> float:
    u = _clamp(u, min_u, max_u)
    t = (u - min_u) / (max_u - min_u)
    return hi + t * (lo - hi)


def sim_to_actuation_thumb_mcp(
    sim_abd_joint: float, sim_flex_tendon: float, sim_mcp_tendon: float
) -> tuple[float, float, float]:
  joint_abd = sim_abd_joint
  joint_flex = (
      0.000344 * sim_abd_joint
      - 78.088995 * sim_flex_tendon
      + 0.188440 * sim_mcp_tendon
      + 2.977490
  )
  joint_mcp = (
      0.004162 * sim_abd_joint
      - 11.373921 * sim_flex_tendon
      - 56.722756 * sim_mcp_tendon
      + 6.666491
  )
  joint_ip = (
      0.004528469071365329 * sim_abd_joint
      - 11.422035184164583 * sim_flex_tendon
      - 56.887542891723974 * sim_mcp_tendon
      + 6.687096101625219
  )

  model = JointsToActuationsModel()
  abd, flex, mcp = model.thumb_actuations(
      joint_abd, joint_flex, joint_mcp, joint_ip
  )
  return abd / PI * 180.0, flex / PI * 180.0, mcp / PI * 180.0


def actuation_to_sim_thumb_cmc_flex(
    actuation_cmc_flex: float, actuation_abd: float
) -> float:
  cable = actuation_cmc_flex / 180.0 * PI * MOTOR_PULLEY_RADIUS
  return ((cable - 2.5000 * actuation_abd) - 37.517992) / (-977.220399)


def actuation_to_sim_thumb_tendon(
    actuation_thumb_tendon: float, actuation_abd: float
) -> float:
  cable = actuation_thumb_tendon / 180.0 * PI * MOTOR_PULLEY_RADIUS
  return ((cable - 2.5000 * actuation_abd) - 136.590025) / (-1241.571958)


def actuation_array_to_sim_array(actuation_arr: list[float]) -> list[float]:
  sim_arr = [0.0] * len(actuation_arr)

  sim_arr[THUMB_ABD_SIM] = sim_to_actuation_forward(
      actuation_arr[THUMB_ABD_ACTUATION],
      SIM_RANGE[THUMB_ABD_SIM][0],
      SIM_RANGE[THUMB_ABD_SIM][1],
      ACTUATION_RANGE[THUMB_ABD_ACTUATION][0],
      ACTUATION_RANGE[THUMB_ABD_ACTUATION][1],
  )
  sim_arr[THUMB_FLEX_SIM] = actuation_to_sim_thumb_cmc_flex(
      actuation_arr[THUMB_FLEX_ACTUATION], sim_arr[THUMB_ABD_SIM]
  )
  sim_arr[THUMB_MCP_SIM] = actuation_to_sim_thumb_tendon(
      actuation_arr[THUMB_MCP_ACTUATION], sim_arr[THUMB_ABD_SIM]
  )

  sim_arr[FINGER_IDX_SIM] = actuation_to_sim_reverse(
      actuation_arr[FINGER_IDX_ACTUATION],
      SIM_RANGE[FINGER_IDX_SIM][0],
      SIM_RANGE[FINGER_IDX_SIM][1],
      ACTUATION_RANGE[FINGER_IDX_ACTUATION][0],
      ACTUATION_RANGE[FINGER_IDX_ACTUATION][1],
  )
  sim_arr[FINGER_MIDDLE_SIM] = actuation_to_sim_reverse(
      actuation_arr[FINGER_MIDDLE_ACTUATION],
      SIM_RANGE[FINGER_MIDDLE_SIM][0],
      SIM_RANGE[FINGER_MIDDLE_SIM][1],
      ACTUATION_RANGE[FINGER_MIDDLE_ACTUATION][0],
      ACTUATION_RANGE[FINGER_MIDDLE_ACTUATION][1],
  )
  sim_arr[FINGER_RING_SIM] = actuation_to_sim_reverse(
      actuation_arr[FINGER_RING_ACTUATION],
      SIM_RANGE[FINGER_RING_SIM][0],
      SIM_RANGE[FINGER_RING_SIM][1],
      ACTUATION_RANGE[FINGER_RING_ACTUATION][0],
      ACTUATION_RANGE[FINGER_RING_ACTUATION][1],
  )
  sim_arr[FINGER_PINKY_SIM] = actuation_to_sim_reverse(
      actuation_arr[FINGER_PINKY_ACTUATION],
      SIM_RANGE[FINGER_PINKY_SIM][0],
      SIM_RANGE[FINGER_PINKY_SIM][1],
      ACTUATION_RANGE[FINGER_PINKY_ACTUATION][0],
      ACTUATION_RANGE[FINGER_PINKY_ACTUATION][1],
  )
  return sim_arr
