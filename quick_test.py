"""Quick test of the complete flow via API."""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ Health check: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is it running?")
        print("  Start server with: uvicorn src.api.main:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_chat(message: str):
    """Test chat endpoint."""
    payload = {
        "message": message,
        "user_id": "test-user"
    }
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"\n📨 Message: {message}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Intent: {data.get('intent')}")
            print(f"   ✓ Confidence: {data.get('confidence', 0):.3f}")
            print(f"   ✓ Strategy: {data.get('strategy')}")
            print(f"   ✓ Escalate: {data.get('escalate', False)}")
            response_text = data.get('response', '')
            preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
            print(f"   ✓ Response: {preview}")
            return True
        else:
            print(f"   ✗ Error: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"   ✗ Request timed out")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("QUICK API TEST")
    print("=" * 60)
    print("\nTesting API endpoints...")
    print("Make sure the server is running: uvicorn src.api.main:app --reload --port 8000\n")
    
    # Test health
    print("[1] Testing health endpoint...")
    if not test_health():
        print("\n❌ Health check failed. Please start the server first.")
        sys.exit(1)
    
    # Test different messages
    print("\n[2] Testing chat endpoint with different messages...")
    print("-" * 60)
    
    messages = [
        "Hello, I need help",
        "What are your business hours?",
        "I have a billing question",
        "How do I reset my password?",
        "I want a refund"
    ]
    
    results = []
    for msg in messages:
        result = test_chat(msg)
        results.append(result)
        print("-" * 60)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        sys.exit(1)

