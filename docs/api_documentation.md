# API Documentation

## Base URL

```
http://localhost:8000  # Local development
https://api.yourdomain.com  # Production
```

## Authentication

Currently, the API does not require authentication for MVP. In production, implement:
- API keys
- OAuth 2.0
- JWT tokens

## Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### Chat

```http
POST /api/v1/chat
```

**Request Body:**
```json
{
  "message": "I need help with my billing",
  "user_id": "user-123",
  "session_id": "optional-session-id",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you?"
    }
  ]
}
```

**Response:**
```json
{
  "response": "I can help you with billing. What specific issue are you experiencing?",
  "intent": "billing",
  "confidence": 0.92,
  "strategy": "fine_tuned",
  "sources": [],
  "escalate": false,
  "conversation_id": "conv-abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Submit Feedback

```http
POST /api/v1/feedback
```

**Request Body:**
```json
{
  "conversation_id": "conv-abc123",
  "feedback": "thumbs_up",
  "correct_intent": "billing"
}
```

**Response:**
```json
{
  "message": "Feedback submitted successfully"
}
```

### WebSocket Chat

```http
WS /api/v1/ws/chat
```

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/chat');

// Send message
ws.send(JSON.stringify({
  message: "Hello",
  user_id: "user-123"
}));

// Receive response
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log(response);
};
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid message"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "detail": "Error message"
}
```

## Rate Limiting

- **Free Tier**: 100 requests/minute
- **Paid Tier**: 1000 requests/minute

## Response Times

- **Pre-trained (Bedrock)**: ~1-2 seconds
- **Fine-tuned**: ~2-3 seconds
- **RAG**: ~3-5 seconds

## Examples

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "I need help",
        "user_id": "user-123"
    }
)

print(response.json())
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need help",
    "user_id": "user-123"
  }'
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/v1/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'I need help',
    user_id: 'user-123'
  })
});

const data = await response.json();
console.log(data);
```

