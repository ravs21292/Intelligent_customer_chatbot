"""Chat API endpoints."""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.models.model_router import multi_model_router
from src.data_collection.kinesis_ingestion import kinesis_ingestion
from src.training.incremental_learning import incremental_learning
from src.utils.helpers import generate_conversation_id, validate_message
from src.utils.logger import logger
from src.utils.metrics import metrics_collector


router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    intent: str
    confidence: float
    strategy: str
    sources: Optional[List[Dict[str, Any]]] = None
    escalate: bool = False
    conversation_id: str
    timestamp: str


class FeedbackRequest(BaseModel):
    """Feedback request model."""
    conversation_id: str
    feedback: str = Field(..., regex="^(thumbs_up|thumbs_down)$")
    correct_intent: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for customer support.
    
    Args:
        request: Chat request with message and user info
        
    Returns:
        Chat response with generated answer
    """
    try:
        # Validate message
        if not validate_message(request.message):
            raise HTTPException(status_code=400, detail="Invalid message")
        
        # Generate conversation ID if not provided
        conversation_id = request.session_id or generate_conversation_id(request.user_id)
        
        # Generate response using multi-model router
        response = multi_model_router.generate_response(
            message=request.message,
            conversation_history=request.conversation_history,
            user_id=request.user_id
        )
        
        # Ingest to Kinesis for logging
        kinesis_ingestion.ingest_chat_message(
            user_id=request.user_id,
            message=request.message,
            session_id=conversation_id,
            metadata={
                "intent": response.get("intent"),
                "strategy": response.get("strategy")
            }
        )
        
        # Track metrics
        metrics_collector.put_metric(
            "chat_requests",
            1,
            dimensions={
                "intent": response.get("intent", "unknown"),
                "strategy": response.get("strategy", "unknown")
            }
        )
        
        return ChatResponse(
            response=response.get("response", ""),
            intent=response.get("intent", "general_inquiry"),
            confidence=response.get("routing", {}).get("confidence", 0.0),
            strategy=response.get("strategy", "pre_trained"),
            sources=response.get("sources"),
            escalate=response.get("escalate", False),
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for model improvement.
    
    Args:
        request: Feedback request
        
    Returns:
        Success confirmation
    """
    try:
        success = incremental_learning.collect_feedback_data(
            conversation_id=request.conversation_id,
            user_feedback=request.feedback,
            correct_intent=request.correct_intent
        )
        
        if success:
            return {"message": "Feedback submitted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to submit feedback")
            
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    
    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()
    conversation_id = None
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            user_id = data.get("user_id", "")
            
            if not conversation_id:
                conversation_id = generate_conversation_id(user_id)
            
            # Generate response
            response = multi_model_router.generate_response(
                message=message,
                user_id=user_id
            )
            
            # Send response
            await websocket.send_json({
                "response": response.get("response", ""),
                "intent": response.get("intent"),
                "strategy": response.get("strategy"),
                "conversation_id": conversation_id
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket chat: {e}")
        await websocket.close(code=1011, reason=str(e))

