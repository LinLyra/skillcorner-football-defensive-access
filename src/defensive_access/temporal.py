from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_STATES = ("high_press", "mid_block", "low_block", "transition")


@dataclass
class DefensiveStateDecoder:
    states: tuple[str, ...] = DEFAULT_STATES
    switch_penalty: float = 2.2
    transition_penalty: dict[tuple[str, str], float] = field(default_factory=dict)

    def decode(self, emission_costs: np.ndarray) -> list[str]:
        """Viterbi decoding over negative-log emission costs."""
        costs = np.asarray(emission_costs, dtype=float)
        if costs.ndim != 2 or costs.shape[1] != len(self.states):
            raise ValueError("emission_costs must be [time, n_states]")
        if len(costs) == 0:
            return []
        n_t, n_s = costs.shape
        dp = np.full((n_t, n_s), np.inf)
        back = np.zeros((n_t, n_s), dtype=int)
        dp[0] = costs[0]
        for t in range(1, n_t):
            for current in range(n_s):
                options = []
                for previous in range(n_s):
                    penalty = 0.0 if previous == current else self.switch_penalty
                    penalty += self.transition_penalty.get(
                        (self.states[previous], self.states[current]), 0.0
                    )
                    options.append(dp[t - 1, previous] + penalty)
                best = int(np.argmin(options))
                dp[t, current] = options[best] + costs[t, current]
                back[t, current] = best
        path = [int(np.argmin(dp[-1]))]
        for t in range(n_t - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        path.reverse()
        return [self.states[i] for i in path]


def state_emission_costs(rows: list[dict]) -> np.ndarray:
    """Rule-informed costs; transparent and intentionally unfitted."""
    result = []
    for row in rows:
        pressure = min(float(row["nearest_defender_to_ball_m"]), 20.0) / 20.0
        ball_depth = abs(float(row["ball_x"])) / 52.5
        compact = min(float(row["defensive_length_m"]), 70.0) / 70.0
        access = float(row["team_access"])
        # Lower is better. High press: close pressure and low accessible options.
        high = 1.5 * pressure + 0.8 * access + 0.3 * (1.0 - ball_depth)
        mid = abs(compact - 0.55) + 0.4 * abs(pressure - 0.35) + 0.3 * access
        low = 0.9 * (1.0 - ball_depth) + 0.5 * abs(compact - 0.35) + 0.4 * access
        transition = 0.6 * compact + 0.5 * (1.0 - pressure) + 0.2 * (1.0 - access)
        result.append([high, mid, low, transition])
    return np.asarray(result, dtype=float)

