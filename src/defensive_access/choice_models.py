from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATURE_SETS = {
    "B0": ["nearest_defender_m"],
    "B1": [
        "nearest_defender_m",
        "pass_distance_m",
        "pass_dx_m",
        "pass_dy_m",
        "receiver_x",
        "receiver_y",
    ],
    "B2": [
        "nearest_defender_m",
        "pass_distance_m",
        "pass_dx_m",
        "pass_dy_m",
        "receiver_x",
        "receiver_y",
        "defenders_within_5m",
        "corridor_defenders",
    ],
    "B3": [
        "nearest_defender_m",
        "pass_distance_m",
        "pass_dx_m",
        "pass_dy_m",
        "receiver_x",
        "receiver_y",
        "defenders_within_5m",
        "corridor_defenders",
        "control_margin_s",
    ],
    "M": [
        "nearest_defender_m",
        "pass_distance_m",
        "pass_dx_m",
        "pass_dy_m",
        "receiver_x",
        "receiver_y",
        "defenders_within_5m",
        "corridor_defenders",
        "control_margin_s",
        "lane_block",
    ],
}


def choice_key(table: pd.DataFrame) -> pd.Series:
    return table["game_id"].astype(str) + ":" + table["choice_set_id"].astype(str)


def _groups(keys: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    return [part for part in np.split(order, boundaries) if len(part)]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


@dataclass
class ConditionalSoftmax:
    ridge: float = 1.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, keys: np.ndarray) -> ConditionalSoftmax:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        z = (x - self.mean_) / self.scale_
        groups = _groups(np.asarray(keys))
        if any(y[index].sum() != 1 for index in groups):
            raise ValueError("Every choice set must contain exactly one chosen option")

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            loss = 0.5 * self.ridge * float(beta @ beta)
            gradient = self.ridge * beta
            for index in groups:
                utilities = z[index] @ beta
                probabilities = _softmax(utilities)
                chosen = int(np.flatnonzero(y[index])[0])
                loss -= np.log(max(probabilities[chosen], 1e-15))
                gradient += z[index].T @ (probabilities - y[index])
            return loss / len(groups), gradient / len(groups)

        result = minimize(
            objective,
            np.zeros(x.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        if not result.success:
            raise RuntimeError(f"Conditional softmax did not converge: {result.message}")
        self.coef_ = result.x
        return self

    def predict_proba(self, x: np.ndarray, keys: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("Model must be fit before prediction")
        z = (np.asarray(x, dtype=float) - self.mean_) / self.scale_
        utilities = z @ self.coef_
        probabilities = np.zeros(len(z), dtype=float)
        for index in _groups(np.asarray(keys)):
            probabilities[index] = _softmax(utilities[index])
        return probabilities


def evaluate_choices(table: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    y = table["chosen"].to_numpy(dtype=int)
    keys = choice_key(table).to_numpy()
    probabilities = np.asarray(probabilities, dtype=float)
    losses: list[float] = []
    reciprocal_ranks: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    brier: list[float] = []
    for index in _groups(keys):
        chosen = int(np.flatnonzero(y[index])[0])
        local = probabilities[index]
        rank = int(np.sum(local > local[chosen])) + 1
        losses.append(-np.log(max(local[chosen], 1e-15)))
        reciprocal_ranks.append(1.0 / rank)
        top1.append(float(rank == 1))
        top3.append(float(rank <= 3))
        brier.append(float(np.mean((local - y[index]) ** 2)))
    return {
        "choice_sets": float(len(losses)),
        "conditional_log_loss": float(np.mean(losses)),
        "top1": float(np.mean(top1)),
        "top3": float(np.mean(top3)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "brier": float(np.mean(brier)),
    }
