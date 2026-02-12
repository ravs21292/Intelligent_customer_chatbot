"""Data preparation for fine-tuning."""

import json
from typing import Dict, Any, List
from config.pipeline_config import pipeline_config
from src.data_collection.s3_storage import s3_storage
from src.utils.logger import logger


class FineTuningDataPreparation:
    """Prepares data for fine-tuning."""
    
    def extract_qa_pairs(
        self,
        support_tickets: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract Q&A pairs from support tickets.
        
        Args:
            support_tickets: List of support ticket records
            
        Returns:
            List of Q&A pairs
        """
        qa_pairs = []
        
        for ticket in support_tickets:
            question = ticket.get("customer_message", ticket.get("question", ""))
            answer = ticket.get("agent_response", ticket.get("answer", ""))
            
            if question and answer:
                qa_pairs.append({
                    "instruction": question,
                    "response": answer,
                    "domain": ticket.get("category", "general")
                })
        
        logger.info(f"Extracted {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def augment_data(
        self,
        qa_pairs: List[Dict[str, str]],
        augmentation_factor: int = 2
    ) -> List[Dict[str, str]]:
        """
        Augment training data.
        
        Args:
            qa_pairs: Original Q&A pairs
            augmentation_factor: How many times to augment
            
        Returns:
            Augmented dataset
        """
        augmented = qa_pairs.copy()
        
        # Simple augmentation: rephrase questions
        for _ in range(augmentation_factor - 1):
            for pair in qa_pairs:
                # Add variations (in production, use more sophisticated methods)
                augmented.append({
                    "instruction": f"Can you help me with: {pair['instruction']}",
                    "response": pair["response"],
                    "domain": pair.get("domain", "general")
                })
        
        logger.info(f"Augmented dataset size: {len(augmented)}")
        return augmented
    
    def prepare_domain_specific_data(
        self,
        domain: str,
        data_source: str
    ) -> List[Dict[str, str]]:
        """
        Prepare domain-specific training data.
        
        Args:
            domain: Domain name (e.g., 'billing', 'technical_support')
            data_source: Path to data source (local or S3)
            
        Returns:
            Prepared training data
        """
        # Load data
        if data_source.startswith("s3://"):
            data = s3_storage.download_data(data_source.replace("s3://", "").split("/", 1)[1])
        else:
            with open(data_source, "r") as f:
                data = json.load(f)
        
        # Filter by domain
        domain_data = [
            item for item in data
            if item.get("domain", item.get("category", "")) == domain
        ]
        
        # Extract Q&A pairs
        qa_pairs = self.extract_qa_pairs(domain_data)
        
        # Augment
        augmented = self.augment_data(qa_pairs)
        
        return augmented
    
    def format_for_training(
        self,
        qa_pairs: List[Dict[str, str]],
        output_path: str
    ):
        """Format and save data for training."""
        with open(output_path, "w") as f:
            json.dump(qa_pairs, f, indent=2)
        
        logger.info(f"Training data saved to {output_path}")


# Global data preparation instance
data_preparation = FineTuningDataPreparation()

