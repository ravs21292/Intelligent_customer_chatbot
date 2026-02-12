"""Real-time data ingestion from Kinesis streams."""

import json
import time
from typing import Dict, Any, List, Optional
import boto3
from botocore.exceptions import ClientError
from config.aws_config import aws_config
from config.pipeline_config import data_collection_config
from src.utils.logger import logger
from src.utils.metrics import metrics_collector


class KinesisIngestion:
    """Handles real-time data ingestion from Kinesis."""
    
    def __init__(self):
        self.kinesis = aws_config.get_kinesis_client()
        self.stream_name = data_collection_config.KINESIS_STREAM_NAME
        self.s3_client = aws_config.get_s3_client()
        self.bucket = data_collection_config.S3_BUCKET_DATA
        self.batch_buffer = []
        self.last_flush = time.time()
        self.flush_interval = data_collection_config.FLUSH_INTERVAL
        
    def create_stream_if_not_exists(self, shard_count: int = 2):
        """Create Kinesis stream if it doesn't exist."""
        try:
            self.kinesis.describe_stream(StreamName=self.stream_name)
            logger.info(f"Stream {self.stream_name} already exists")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(f"Creating stream {self.stream_name}")
                self.kinesis.create_stream(
                    StreamName=self.stream_name,
                    ShardCount=shard_count
                )
                # Wait for stream to be active
                waiter = self.kinesis.get_waiter("stream_exists")
                waiter.wait(StreamName=self.stream_name)
                logger.info(f"Stream {self.stream_name} created successfully")
            else:
                raise
    
    def ingest_chat_message(
        self,
        user_id: str,
        message: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Ingest a chat message to Kinesis.
        
        Args:
            user_id: User identifier
            message: Chat message content
            session_id: Conversation session ID
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            record = {
                "user_id": user_id,
                "message": message,
                "session_id": session_id,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            response = self.kinesis.put_record(
                StreamName=self.stream_name,
                Data=json.dumps(record),
                PartitionKey=user_id
            )
            
            metrics_collector.put_metric(
                "messages_ingested",
                1,
                dimensions={"stream": self.stream_name}
            )
            
            logger.debug(f"Message ingested: {response['SequenceNumber']}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting message: {e}")
            metrics_collector.put_metric(
                "ingestion_errors",
                1,
                dimensions={"error_type": type(e).__name__}
            )
            return False
    
    def batch_ingest(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch ingest multiple records.
        
        Args:
            records: List of record dictionaries
            
        Returns:
            Response with success/failure counts
        """
        try:
            kinesis_records = [
                {
                    "Data": json.dumps(record),
                    "PartitionKey": record.get("user_id", "default")
                }
                for record in records
            ]
            
            response = self.kinesis.put_records(
                Records=kinesis_records,
                StreamName=self.stream_name
            )
            
            success_count = sum(1 for r in response["Records"] if "ErrorCode" not in r)
            failed_count = len(records) - success_count
            
            metrics_collector.put_metric(
                "batch_ingestion_success",
                success_count
            )
            if failed_count > 0:
                metrics_collector.put_metric(
                    "batch_ingestion_failures",
                    failed_count
                )
            
            return {
                "success": success_count,
                "failed": failed_count,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error in batch ingestion: {e}")
            return {"success": 0, "failed": len(records), "error": str(e)}
    
    def consume_stream(
        self,
        shard_id: str,
        callback: callable,
        max_records: int = 100
    ):
        """
        Consume records from a Kinesis stream shard.
        
        Args:
            shard_id: Shard identifier
            callback: Function to process each record
            max_records: Maximum records to retrieve per call
        """
        shard_iterator = self.kinesis.get_shard_iterator(
            StreamName=self.stream_name,
            ShardId=shard_id,
            ShardIteratorType="LATEST"
        )["ShardIterator"]
        
        while True:
            try:
                response = self.kinesis.get_records(
                    ShardIterator=shard_iterator,
                    Limit=max_records
                )
                
                for record in response["Records"]:
                    data = json.loads(record["Data"])
                    callback(data)
                
                shard_iterator = response.get("NextShardIterator")
                
                if not shard_iterator:
                    break
                    
                time.sleep(1)  # Avoid throttling
                
            except Exception as e:
                logger.error(f"Error consuming stream: {e}")
                break
    
    def archive_to_s3(self, records: List[Dict[str, Any]], date_prefix: str):
        """
        Archive records to S3 for long-term storage.
        
        Args:
            records: List of records to archive
            date_prefix: Date prefix for S3 path (e.g., "2024/01/15")
        """
        if not records:
            return
        
        try:
            key = f"{data_collection_config.RAW_DATA_PREFIX}/{date_prefix}/batch_{int(time.time())}.json"
            
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(records, indent=2),
                ContentType="application/json"
            )
            
            logger.info(f"Archived {len(records)} records to s3://{self.bucket}/{key}")
            metrics_collector.put_metric(
                "records_archived",
                len(records),
                dimensions={"bucket": self.bucket}
            )
            
        except Exception as e:
            logger.error(f"Error archiving to S3: {e}")


# Global ingestion instance
kinesis_ingestion = KinesisIngestion()

