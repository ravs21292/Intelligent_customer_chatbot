"""Data versioning using DVC (Data Version Control)."""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import subprocess
from config.pipeline_config import pipeline_config
from src.data_collection.s3_storage import s3_storage
from src.utils.logger import logger


class DataVersioning:
    """Manages data versioning using DVC."""
    
    def __init__(self):
        self.dvc_dir = ".dvc"
        self.remote_name = "s3-remote"
        self.remote_url = f"s3://{pipeline_config.S3_BUCKET_DATA}/dvc-storage"
    
    def initialize_dvc(self) -> bool:
        """Initialize DVC repository."""
        try:
            if not os.path.exists(self.dvc_dir):
                subprocess.run(["dvc", "init"], check=True, capture_output=True)
                logger.info("DVC initialized")
            
            # Set up remote if not exists
            result = subprocess.run(
                ["dvc", "remote", "list"],
                capture_output=True,
                text=True
            )
            
            if self.remote_name not in result.stdout:
                subprocess.run(
                    ["dvc", "remote", "add", "-d", self.remote_name, self.remote_url],
                    check=True,
                    capture_output=True
                )
                logger.info(f"DVC remote configured: {self.remote_url}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error initializing DVC: {e}")
            return False
    
    def track_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Track a dataset with DVC.
        
        Args:
            dataset_path: Path to dataset file/directory
            dataset_name: Name for the dataset
            metadata: Optional metadata to store
            
        Returns:
            True if successful
        """
        try:
            # Add dataset to DVC
            subprocess.run(
                ["dvc", "add", dataset_path],
                check=True,
                capture_output=True
            )
            
            # Create metadata file
            if metadata:
                metadata_path = f"{dataset_path}.meta"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            # Commit to DVC
            subprocess.run(
                ["dvc", "commit", "-f", dataset_path + ".dvc"],
                check=True,
                capture_output=True
            )
            
            logger.info(f"Dataset {dataset_name} tracked with DVC")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error tracking dataset: {e}")
            return False
    
    def push_dataset(self, dataset_path: str) -> bool:
        """Push dataset to remote storage."""
        try:
            subprocess.run(
                ["dvc", "push", dataset_path + ".dvc"],
                check=True,
                capture_output=True
            )
            logger.info(f"Dataset pushed to remote: {dataset_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error pushing dataset: {e}")
            return False
    
    def pull_dataset(self, dataset_path: str) -> bool:
        """Pull dataset from remote storage."""
        try:
            subprocess.run(
                ["dvc", "pull", dataset_path + ".dvc"],
                check=True,
                capture_output=True
            )
            logger.info(f"Dataset pulled from remote: {dataset_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error pulling dataset: {e}")
            return False
    
    def get_dataset_info(self, dataset_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a tracked dataset."""
        try:
            dvc_file = dataset_path + ".dvc"
            if not os.path.exists(dvc_file):
                return None
            
            with open(dvc_file, "r") as f:
                dvc_data = f.read()
            
            # Parse DVC file (YAML format)
            import yaml
            dvc_info = yaml.safe_load(dvc_data)
            
            return {
                "path": dataset_path,
                "md5": dvc_info.get("outs", [{}])[0].get("md5", ""),
                "size": dvc_info.get("outs", [{}])[0].get("size", 0),
                "dvc_info": dvc_info
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return None
    
    def create_data_snapshot(
        self,
        datasets: Dict[str, str],
        snapshot_name: str,
        description: str = ""
    ) -> bool:
        """
        Create a snapshot of multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to paths
            snapshot_name: Name for the snapshot
            description: Optional description
            
        Returns:
            True if successful
        """
        try:
            snapshot_metadata = {
                "snapshot_name": snapshot_name,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "datasets": datasets
            }
            
            # Save snapshot metadata
            snapshot_file = f"data/snapshots/{snapshot_name}.json"
            os.makedirs(os.path.dirname(snapshot_file), exist_ok=True)
            
            with open(snapshot_file, "w") as f:
                json.dump(snapshot_metadata, f, indent=2)
            
            # Track snapshot file
            self.track_dataset(snapshot_file, snapshot_name)
            
            logger.info(f"Data snapshot created: {snapshot_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return False


# Global data versioning instance
data_versioning = DataVersioning()

