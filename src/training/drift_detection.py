"""Data and concept drift detection - wrapper for comprehensive detector."""

from typing import Dict, Any, List
import pandas as pd
from src.monitoring.drift_detector import drift_detector
from src.utils.logger import logger


class DriftDetector:
    """Detects data and concept drift - uses comprehensive drift detector."""
    
    def detect_data_drift(
        self,
        reference_data: List[Dict[str, Any]],
        current_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Drift detection results
        """
        # Convert to DataFrame for drift detection
        try:
            ref_df = pd.DataFrame(reference_data)
            curr_df = pd.DataFrame(current_data)
            
            return drift_detector.detect_data_drift(ref_df, curr_df)
        except Exception as e:
            logger.error(f"Error in data drift detection: {e}")
            return {
                "drift_detected": False,
                "error": str(e)
            }
    
    def detect_concept_drift(
        self,
        model_performance_history: List[float]
    ) -> Dict[str, Any]:
        """
        Detect concept drift from performance degradation.
        
        Args:
            model_performance_history: Historical performance metrics
            
        Returns:
            Concept drift detection results
        """
        return drift_detector.detect_concept_drift(model_performance_history)


# Global drift detector instance
drift_detector_wrapper = DriftDetector()

