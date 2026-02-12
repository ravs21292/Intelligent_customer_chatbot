"""Debug the complete flow step by step."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_flow(message: str):
    """Debug the complete flow."""
    print("=" * 60)
    print(f"DEBUGGING FLOW FOR MESSAGE: {message}")
    print("=" * 60)
    
    # Step 1: Intent Classification
    print("\n[STEP 1] Intent Classification")
    print("-" * 60)
    try:
        from src.intent_classification.intent_classifier import intent_classifier
        intent_result = intent_classifier.classify(message)
        print(f"✓ Intent: {intent_result['intent']}")
        print(f"✓ Confidence: {intent_result['confidence']:.3f}")
        if 'top_intents' in intent_result:
            print(f"✓ Top 3 intents:")
            for top_intent in intent_result['top_intents']:
                print(f"    - {top_intent['intent']}: {top_intent['confidence']:.3f}")
    except Exception as e:
        print(f"✗ Error in intent classification: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Routing Decision
    print("\n[STEP 2] Routing Decision")
    print("-" * 60)
    try:
        from src.intent_classification.router import model_router
        routing_decision = model_router.route(message, intent_result)
        print(f"✓ Strategy: {routing_decision['strategy']}")
        print(f"✓ Model Name: {routing_decision.get('model_name', 'N/A')}")
        print(f"✓ Reasoning: {routing_decision.get('reasoning', 'N/A')}")
        print(f"✓ Use RAG: {routing_decision.get('use_rag', False)}")
        print(f"✓ Confidence: {routing_decision.get('confidence', 0):.3f}")
    except Exception as e:
        print(f"✗ Error in routing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Response Generation
    print("\n[STEP 3] Response Generation")
    print("-" * 60)
    try:
        from src.models.model_router import multi_model_router
        response = multi_model_router.generate_response(message)
        print(f"✓ Response generated")
        print(f"✓ Strategy used: {response.get('strategy', 'N/A')}")
        print(f"✓ Response length: {len(response.get('response', ''))}")
        print(f"✓ Escalate: {response.get('escalate', False)}")
        if response.get('sources'):
            print(f"✓ Sources: {len(response['sources'])} documents")
        print(f"\nResponse preview:")
        response_text = response.get('response', '')
        print(f"  {response_text[:200]}..." if len(response_text) > 200 else f"  {response_text}")
    except Exception as e:
        print(f"✗ Error in response generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✓ FLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    # Test with different messages
    test_messages = [
        "Hello, I need help with my billing",
        "What are your business hours?",
        "How do I reset my password?",
        "I want a refund for my subscription"
    ]
    
    print("\n" + "=" * 60)
    print("DEBUGGING COMPLETE FLOW")
    print("=" * 60)
    print("\nThis script will test the flow for multiple messages.")
    print("Make sure the API server is NOT running (this tests the code directly).\n")
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_messages)}")
        print(f"{'='*60}")
        debug_flow(msg)
        if i < len(test_messages):
            print("\n" + "-" * 60 + "\n")

