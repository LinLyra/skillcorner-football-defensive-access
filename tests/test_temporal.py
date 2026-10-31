import numpy as np

from defensive_access.temporal import DefensiveStateDecoder


def test_switch_penalty_removes_one_frame_flicker():
    costs = np.array(
        [
            [0.0, 3.0, 3.0, 3.0],
            [0.0, 3.0, 3.0, 3.0],
            [1.0, 0.0, 3.0, 3.0],
            [0.0, 3.0, 3.0, 3.0],
        ]
    )
    labels = DefensiveStateDecoder(switch_penalty=2.0).decode(costs)
    assert labels == ["high_press"] * 4

