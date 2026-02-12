"""AWS Bedrock client for pre-trained LLM access."""

import json
from typing import Dict, Any, Optional, List
from config.aws_config import aws_config
from config.model_config import model_config
from src.utils.logger import logger
from src.utils.metrics import metrics_collector


class BedrockClient:
    """Client for interacting with AWS Bedrock models."""
    
    def __init__(self):
        self.bedrock = aws_config.get_bedrock_client()
        self.model_id = model_config.BEDROCK_MODEL_ID
        self.max_tokens = model_config.BEDROCK_MAX_TOKENS
        self.temperature = model_config.BEDROCK_TEMPERATURE
    
    @metrics_collector.track_latency("bedrock_inference")
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate response using Bedrock model.
        
        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response with metadata
        """
        try:
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            # Format prompt for Claude
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
            
            # Prepare request body
            body = {
                "prompt": formatted_prompt,
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stop_sequences": ["\n\nHuman:"]
            }
            
            # Invoke model
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            generated_text = response_body.get("completion", "")
            
            result = {
                "response": generated_text.strip(),
                "model": self.model_id,
                "strategy": "pre_trained",
                "tokens_used": response_body.get("usage", {}).get("total_tokens", 0)
            }
            
            metrics_collector.put_metric(
                "bedrock_requests",
                1,
                dimensions={"model": self.model_id}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from Bedrock: {e}")
            metrics_collector.put_metric(
                "bedrock_errors",
                1,
                dimensions={"error_type": type(e).__name__}
            )
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
                "error": str(e),
                "strategy": "pre_trained"
            }
    
    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response in chat format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            context: Optional context information
            
        Returns:
            Generated response
        """
        # Build conversation prompt
        conversation = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                conversation.append(f"Human: {content}")
            elif role == "assistant":
                conversation.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(conversation)
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        return self.generate_response(prompt)
    
    def generate_customer_support_response(
        self,
        user_message: str,
        intent: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate customer support response with context.
        
        Args:
            user_message: User's message
            intent: Classified intent
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response
        """
        system_prompt = f"""You are a helpful customer support assistant. 
The user's intent has been classified as: {intent}.

Provide a clear, helpful, and professional response. Be concise but thorough.
If you cannot help with the request, politely suggest escalating to a human agent."""

        if conversation_history:
            messages = conversation_history + [{"role": "user", "content": user_message}]
            return self.generate_chat_response(messages, context=system_prompt)
        else:
            return self.generate_response(user_message, system_prompt=system_prompt)


# Global Bedrock client instance
bedrock_client = BedrockClient()

