from typing import List

from config import Config
from models import Node, VehicleState


def initialize_fleet(cfg: Config, depot: Node) -> List[VehicleState]:
    """初始化車隊 — 所有車輛在 t=0 位於 depot 待命"""
    fleet: List[VehicleState] = []
    vid = 0

    for _ in range(cfg.NUM_MCS_SLOW):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='mcs_slow',
            position=depot, available_time=0.0,
            remaining_energy=cfg.MCS_CAPACITY,
        ))
        vid += 1

    for _ in range(cfg.NUM_MCS_FAST):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='mcs_fast',
            position=depot, available_time=0.0,
            remaining_energy=cfg.MCS_CAPACITY,
        ))
        vid += 1

    for _ in range(cfg.NUM_UAV):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='uav',
            position=depot, available_time=0.0,
            remaining_energy=cfg.UAV_DELIVERABLE_ENERGY,
        ))
        vid += 1

    return fleet
