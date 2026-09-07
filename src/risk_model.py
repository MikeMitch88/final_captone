"""XGBoost risk scoring and evaluation for custody consignments."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_engineering import CustodyFeatureEngineer, FEATURE_COLUMNS


class CustodyRiskModel:
    """Train and serve a normalized XGBoost consignment risk score."""

    def __init__(self, threshold: float = 0.5, random_state: int = 42):
        self.threshold = threshold
        self.random_state = random_state
        self.model: Any = None

    def _new_classifier(self, positive_count: int, negative_count: int) -> Any:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is required for risk scoring. Install requirements.txt first."
            ) from exc

        scale_pos_weight = max(1.0, negative_count / max(positive_count, 1))
        return XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.9,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=4,
        )

    def fit(self, features: pd.DataFrame, target_column: str = "anomaly_label") -> "CustodyRiskModel":
        if target_column not in features:
            raise ValueError(f"Missing target column: {target_column}")
        target = features[target_column].astype(int)
        if target.nunique() < 2:
            raise ValueError("Training target must contain both positive and negative labels")
        self.model = self._new_classifier(
            int(target.sum()), int((target == 0).sum())
        )
        self.model.fit(features[FEATURE_COLUMNS], target)
        return self

    def predict_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        scores = self.model.predict_proba(features[FEATURE_COLUMNS])[:, 1]
        result = features.copy()
        result["Risk_Score"] = np.round(scores * 100, 2)
        result["High_Risk_Alert"] = scores >= self.threshold
        return result

    def evaluate(self, test_features: pd.DataFrame) -> dict[str, float]:
        scored = self.predict_scores(test_features)
        actual = test_features["anomaly_label"].astype(int)
        predicted = scored["High_Risk_Alert"].astype(int)
        true_positive = int(((actual == 1) & (predicted == 1)).sum())
        false_positive = int(((actual == 0) & (predicted == 1)).sum())
        false_negative = int(((actual == 1) & (predicted == 0)).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        metrics: dict[str, float] = {
            "precision": precision,
            "recall": recall,
            "detection_rate": recall,
            "volumetric_discrepancy_detection_rate": float(
                self._recall(test_features["volumetric_anomaly_label"].astype(int), predicted)
            ),
        }

        alert_time = self._first_datetime(test_features, ["timestamp_telemetry", "timestamp"])
        event_time = self._first_datetime(test_features, ["sample_timestamp"])
        lead_hours = (event_time - alert_time).dt.total_seconds() / 3600
        detected_leads = lead_hours[(actual == 1) & (predicted == 1)].dropna()
        metrics["detection_lead_time_hours"] = float(detected_leads.mean()) if not detected_leads.empty else 0.0
        return metrics

    @staticmethod
    def _recall(actual: pd.Series, predicted: pd.Series) -> float:
        true_positive = int(((actual == 1) & (predicted == 1)).sum())
        false_negative = int(((actual == 1) & (predicted == 0)).sum())
        return true_positive / max(true_positive + false_negative, 1)

    @staticmethod
    def _first_datetime(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for column in candidates:
            if column in frame:
                return pd.to_datetime(frame[column], errors="coerce")
        return pd.Series(pd.NaT, index=frame.index)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model")
        with open(path, "wb") as stream:
            pickle.dump(self, stream)

    @staticmethod
    def load(path: str | Path) -> "CustodyRiskModel":
        with open(path, "rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, CustodyRiskModel):
            raise ValueError("File does not contain a CustodyRiskModel")
        return model


def train_and_evaluate(db_path: str, model_path: str) -> dict[str, float]:
    features = CustodyFeatureEngineer().build_from_sqlite(db_path)
    rng = np.random.default_rng(42)
    train_parts = []
    test_parts = []
    for _, group in features.groupby("anomaly_label"):
        indices = rng.permutation(len(group))
        split_at = max(1, int(len(group) * 0.75))
        train_parts.append(group.iloc[indices[:split_at]])
        test_parts.append(group.iloc[indices[split_at:]])
    train = pd.concat(train_parts).sample(frac=1, random_state=42)
    test = pd.concat(test_parts).sample(frac=1, random_state=42)
    model = CustodyRiskModel().fit(train)
    metrics = model.evaluate(test)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custody XGBoost risk model")
    parser.add_argument("--db", default="data/custody_data.db")
    parser.add_argument("--model", default="data/custody_risk_model.pkl")
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(args.db, args.model), indent=2))


if __name__ == "__main__":
    main()