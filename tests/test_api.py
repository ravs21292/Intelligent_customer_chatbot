"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint():
    """Test chat endpoint."""
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hello, I need help",
            "user_id": "test-user-123"
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert "intent" in response.json()


def test_chat_endpoint_invalid_message():
    """Test chat endpoint with invalid message."""
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "",
            "user_id": "test-user-123"
        }
    )
    assert response.status_code == 400

