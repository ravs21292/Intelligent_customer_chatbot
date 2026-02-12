"""Comprehensive drift detection implementation."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from config.pipeline_config import pipeline_config
from src.utils.logger import logger
from src.utils.metrics import metrics_collector


class DriftDetector:
    """Detects data drift, concept drift, and model drift."""
    
    def __init__(self):
        self.drift_threshold = pipeline_config.DRIFT_DETECTION_THRESHOLD
        self.performance_threshold = pipeline_config.PERFORMANCE_DEGRADATION_THRESHOLD
    
    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            reference_data: Baseline/reference dataset
            current_data: Current dataset to compare
            
        Returns:
            Drift detection results
        """
        drift_results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "feature_drift": {},
            "overall_drift": {}
        }
        
        drifted_features = []
        
        # Check each feature
        for column in reference_data.columns:
            if column in current_data.columns:
                ref_values = reference_data[column].dropna()
                curr_values = current_data[column].dropna()
                
                if len(ref_values) == 0 or len(curr_values) == 0:
                    continue
                
                # Continuous features: KS test
                if pd.api.types.is_numeric_dtype(ref_values):
                    ks_stat, p_value = ks_2samp(ref_values, curr_values)
                    drift_detected = p_value < 0.05
                    
                    # Calculate PSI (Population Stability Index)
                    psi = self._calculate_psi(ref_values, curr_values)
                    
                    drift_results["feature_drift"][column] = {
                        "type": "continuous",
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "psi": float(psi),
                        "drift_detected": drift_detected or psi > 0.2
                    }
                
                # Categorical features: Chi-square test
                else:
                    # Create contingency table
                    ref_counts = ref_values.value_counts()
                    curr_counts = curr_values.value_counts()
                    
                    # Align indices
                    all_categories = set(ref_counts.index) | set(curr_counts.index)
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                    curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                        contingency = np.array([ref_aligned, curr_aligned])
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        drift_detected = p_value < 0.05
                        
                        drift_results["feature_drift"][column] = {
                            "type": "categorical",
                            "chi2_statistic": float(chi2),
                            "p_value": float(p_value),
                            "drift_detected": drift_detected
                        }
                
                # Track drifted features
                if drift_results["feature_drift"][column].get("drift_detected", False):
                    drifted_features.append(column)
        
        # Overall drift score
        total_features = len(drift_results["feature_drift"])
        if total_features > 0:
            drift_score = len(drifted_features) / total_features
            drift_results["drift_score"] = drift_score
            drift_results["drift_detected"] = drift_score > self.drift_threshold
        
        drift_results["overall_drift"] = {
            "drifted_features": drifted_features,
            "total_features": total_features,
            "drift_percentage": drift_score * 100
        }
        
        # Log and track metrics
        if drift_results["drift_detected"]:
            logger.warning(f"Data drift detected: {drift_score:.2%} features drifted")
            metrics_collector.put_metric(
                "data_drift_detected",
                1,
                dimensions={"drift_score": str(drift_score)}
            )
        
        return drift_results
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Population Stability Index (PSI)."""
        # Create bins from reference data
        _, bin_edges = np.histogram(reference, bins=10)
        
        # Calculate distributions
        ref_dist, _ = np.histogram(reference, bins=bin_edges)
        curr_dist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize
        ref_dist = ref_dist / len(reference) if len(reference) > 0 else ref_dist
        curr_dist = curr_dist / len(current) if len(current) > 0 else curr_dist
        
        # Avoid division by zero
        ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
        curr_dist = np.where(curr_dist == 0, 0.0001, curr_dist)
        
        # Calculate PSI
        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        
        return float(psi)
    
    def detect_concept_drift(
        self,
        model_performance_history: List[float],
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Detect concept drift from performance degradation.
        
        Args:
            model_performance_history: List of performance metrics over time
            window_size: Window size for comparison
            
        Returns:
            Concept drift detection results
        """
        if len(model_performance_history) < window_size:
            return {
                "drift_detected": False,
                "reason": "Insufficient data"
            }
        
        # Baseline performance (first window)
        baseline_window = model_performance_history[:window_size]
        baseline_avg = np.mean(baseline_window)
        baseline_std = np.std(baseline_window)
        
        # Recent performance (last window)
        recent_window = model_performance_history[-window_size:]
        recent_avg = np.mean(recent_window)
        recent_std = np.std(recent_window)
        
        # Calculate degradation
        degradation = baseline_avg - recent_avg
        degradation_pct = (degradation / baseline_avg) * 100 if baseline_avg > 0 else 0
        
        # Statistical test (t-test would be better, but simplified here)
        drift_detected = degradation > self.performance_threshold
        
        result = {
            "drift_detected": drift_detected,
            "degradation": float(degradation),
            "degradation_percentage": float(degradation_pct),
            "baseline": {
                "mean": float(baseline_avg),
                "std": float(baseline_std)
            },
            "recent": {
                "mean": float(recent_avg),
                "std": float(recent_std)
            },
            "threshold": self.performance_threshold
        }
        
        if drift_detected:
            logger.warning(f"Concept drift detected: {degradation_pct:.2f}% degradation")
            metrics_collector.put_metric(
                "concept_drift_detected",
                1,
                dimensions={"degradation": str(degradation_pct)}
            )
        
        return result
    
    def detect_model_drift(
        self,
        prediction_distribution: Dict[str, float],
        reference_distribution: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detect model drift from prediction distribution changes.
        
        Args:
            prediction_distribution: Current prediction distribution
            reference_distribution: Reference/baseline distribution
            
        Returns:
            Model drift detection results
        """
        # Calculate KL divergence or simple difference
        all_classes = set(prediction_distribution.keys()) | set(reference_distribution.keys())
        
        total_drift = 0.0
        for class_name in all_classes:
            pred_prob = prediction_distribution.get(class_name, 0.0)
            ref_prob = reference_distribution.get(class_name, 0.0)
            drift = abs(pred_prob - ref_prob)
            total_drift += drift
        
        avg_drift = total_drift / len(all_classes) if all_classes else 0.0
        drift_detected = avg_drift > 0.1  # 10% average change
        
        result = {
            "drift_detected": drift_detected,
            "drift_score": float(avg_drift),
            "class_drift": {
                class_name: abs(
                    prediction_distribution.get(class_name, 0.0) -
                    reference_distribution.get(class_name, 0.0)
                )
                for class_name in all_classes
            }
        }
        
        if drift_detected:
            logger.warning(f"Model drift detected: {avg_drift:.2%} average change")
        
        return result


# Global drift detector instance
drift_detector = DriftDetector()

