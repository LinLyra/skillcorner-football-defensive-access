from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_dangerous_options(options: pd.DataFrame) -> pd.DataFrame:
    table = options.copy()
    capped_distance = table["nearest_defender_m"].clip(upper=15.0)
    table["distance_access"] = 1.0 / (1.0 + np.exp(-(-1.2 + 0.18 * capped_distance)))
    table["distance_danger"] = table["distance_access"] * table["threat"]
    table["access_danger"] = table["access_probability"] * table["threat"]
    table["selected_threat_component"] = table["chosen"] * table["threat"]

    keys = ["game_id", "choice_set_id"]
    result = table.groupby(keys, as_index=False).agg(
        decision_frame=("decision_frame", "first"),
        aligned_frame=("aligned_frame", "first"),
        option_count=("receiver_id", "size"),
        sum_threat=("threat", "sum"),
        max_threat=("threat", "max"),
        selected_threat=("selected_threat_component", "sum"),
        distance_concession=("distance_danger", "sum"),
        access_concession=("access_danger", "sum"),
        mean_nearest_defender_m=("nearest_defender_m", "mean"),
        lead_to_shot=("lead_to_shot", "first"),
        pass_epv_delta_for=("pass_epv_delta_for", "first"),
        n_opponents_bypassed=("n_opponents_bypassed", "first"),
    )
    result["access_increment"] = result["access_concession"] - result["distance_concession"]
    return result
