import pandas as pd
import pytest

from defensive_access.threat import aggregate_dangerous_options


def test_dangerous_option_concession_is_threat_weighted():
    options = pd.DataFrame(
        {
            "game_id": ["1", "1"],
            "choice_set_id": ["a", "a"],
            "decision_frame": [10, 10],
            "aligned_frame": [9, 9],
            "receiver_id": [2, 3],
            "chosen": [1, 0],
            "threat": [0.8, 0.2],
            "nearest_defender_m": [5.0, 5.0],
            "access_probability": [0.5, 0.25],
            "lead_to_shot": [0, 0],
            "pass_epv_delta_for": [0.01, 0.01],
            "n_opponents_bypassed": [2, 2],
        }
    )
    result = aggregate_dangerous_options(options).iloc[0]
    assert result["access_concession"] == pytest.approx(0.45)
    assert result["selected_threat"] == pytest.approx(0.8)
