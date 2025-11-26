"""Isolation Forest anomaly detection for cascade patterns."""

import numpy as np
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float  # -1 to 1, lower = more anomalous
    features: dict
    explanation: str


class AnomalyDetector:
    """Detect unusual cascade patterns using Isolation Forest."""

    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.is_fitted = False
        self.feature_names = [
            "rainfall_mm",
            "depth_m",
            "duration_h",
            "alert_count",
            "max_risk_score",
            "cascade_depth",
        ]

    def extract_features(self, scan_result) -> np.ndarray:
        """Extract features from scan result."""
        hazard = scan_result.hazard_state
        alerts = scan_result.alerts

        max_risk = max((a.risk_score for a in alerts), default=0)
        cascade_depth = sum(1 for a in alerts if a.hazard_type == "cascade")

        return np.array([
            hazard.rainfall_24h_mm,
            hazard.depth_m,
            hazard.duration_h,
            len(alerts),
            max_risk,
            cascade_depth,
        ])

    def fit(self, scan_results: List) -> None:
        """Fit model on historical scan results."""
        if len(scan_results) < 10:
            logger.warning("Need ≥10 samples for reliable anomaly detection")
            return

        X = np.array([self.extract_features(r) for r in scan_results])
        self.model.fit(X)
        self.is_fitted = True
        logger.info(f"Fitted anomaly detector on {len(scan_results)} samples")

    def detect(self, scan_result) -> AnomalyResult:
        """Check if scan result is anomalous."""
        features = self.extract_features(scan_result)

        if not self.is_fitted:
            # Use heuristic thresholds
            return self._heuristic_check(features)

        X = features.reshape(1, -1)
        score = self.model.decision_function(X)[0]
        is_anomaly = self.model.predict(X)[0] == -1

        feature_dict = dict(zip(self.feature_names, features))

        if is_anomaly:
            explanation = self._explain_anomaly(feature_dict)
        else:
            explanation = "Pattern within normal range"

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=float(score),
            features=feature_dict,
            explanation=explanation,
        )

    def _heuristic_check(self, features: np.ndarray) -> AnomalyResult:
        """Fallback heuristic when model not fitted."""
        rainfall, depth, duration, alerts, max_risk, cascades = features

        feature_dict = dict(zip(self.feature_names, features))
        anomalies = []

        # Extreme thresholds
        if rainfall > 200:
            anomalies.append(f"Extreme rainfall: {rainfall:.0f}mm")
        if depth > 1.0:
            anomalies.append(f"Severe flooding: {depth:.2f}m")
        if alerts > 15:
            anomalies.append(f"High alert count: {int(alerts)}")
        if max_risk > 90:
            anomalies.append(f"Critical risk: {max_risk:.0f}%")
        if cascades > 5:
            anomalies.append(f"Multiple cascades: {int(cascades)}")

        is_anomaly = len(anomalies) > 0
        score = -0.5 if is_anomaly else 0.5

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            features=feature_dict,
            explanation="; ".join(anomalies) if anomalies else "Normal conditions",
        )

    def _explain_anomaly(self, features: dict) -> str:
        """Generate human-readable explanation."""
        parts = []

        if features["rainfall_mm"] > 150:
            parts.append(f"Heavy rain ({features['rainfall_mm']:.0f}mm)")
        if features["depth_m"] > 0.5:
            parts.append(f"Deep flooding ({features['depth_m']:.2f}m)")
        if features["alert_count"] > 10:
            parts.append(f"Many alerts ({int(features['alert_count'])})")
        if features["cascade_depth"] > 3:
            parts.append(f"Deep cascades ({int(features['cascade_depth'])})")

        return "; ".join(parts) if parts else "Unusual pattern detected"


anomaly_detector = AnomalyDetector()
