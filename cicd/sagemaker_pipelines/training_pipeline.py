"""SageMaker Pipeline for model training."""

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.huggingface import HuggingFace
from config.pipeline_config import pipeline_config
from config.model_config import model_config


def create_training_pipeline() -> Pipeline:
    """Create SageMaker training pipeline."""
    
    # Pipeline parameters
    model_name = ParameterString(
        name="ModelName", 
        default_value="intent-classifier"  # Pipeline job name, not the ML model
    )
    training_data_uri = ParameterString(
        name="TrainingDataUri",
        default_value=f"{pipeline_config.S3_TRAINING_PATH}/intent_train"
    )
    
    # Training step
    # The actual ML model (BERT-base-uncased) is specified in hyperparameters
    # and will be used in the training script (train.py)
    estimator = HuggingFace(
        entry_point="train.py",
        source_dir="src/intent_classification",
        instance_type="ml.g4dn.xlarge",
        role=pipeline_config.SAGEMAKER_ROLE_ARN,
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39",
        hyperparameters={
            "model_name": model_config.INTENT_MODEL_NAME,  # "bert-base-uncased"
            "num_labels": len(model_config.INTENT_CLASSES),  # 8
            "epochs": 3,
            "batch_size": model_config.BATCH_SIZE,  # 32
            "max_length": model_config.MAX_SEQUENCE_LENGTH  # 512
        }
    )
    
    training_step = TrainingStep(
        name="TrainIntentClassifier",
        estimator=estimator,
        inputs={
            "training": training_data_uri
        }
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name="CustomerChatbotTrainingPipeline",
        parameters=[model_name, training_data_uri],
        steps=[training_step]
    )
    
    return pipeline


if __name__ == "__main__":
    pipeline = create_training_pipeline()
    pipeline.upsert(role_arn=pipeline_config.SAGEMAKER_ROLE_ARN)
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")

