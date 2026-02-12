"""Test if project structure and imports work."""

print("Testing project structure...")
print("=" * 60)

try:
    print("\n[1] Testing config imports...")
    from config.model_config import model_config, domain_config
    print(f"  ✓ Intent classes: {len(model_config.INTENT_CLASSES)}")
    print(f"     Classes: {model_config.INTENT_CLASSES}")
    print(f"  ✓ Routing thresholds: {model_config.ROUTING_THRESHOLDS}")
    print(f"  ✓ Domains configured: {len(domain_config.DOMAINS)}")
    
    print("\n[2] Testing API imports...")
    from src.api.main import app
    print("  ✓ FastAPI app created")
    print(f"  ✓ App title: {app.title}")
    
    print("\n[3] Testing intent classification imports...")
    from src.intent_classification.router import model_router
    print("  ✓ Model router imported")
    
    print("\n[4] Testing model router imports...")
    from src.models.model_router import multi_model_router
    print("  ✓ Multi-model router imported")
    
    print("\n[5] Testing utility imports...")
    from src.utils.helpers import validate_message, generate_conversation_id
    print("  ✓ Helper functions imported")
    
    print("\n[6] Testing logger...")
    from src.utils.logger import logger
    print("  ✓ Logger imported")
    
    print("\n" + "=" * 60)
    print("✅ ALL STRUCTURE CHECKS PASSED!")
    print("=" * 60)
    print("\nProject structure is correct. You can proceed to full setup.")
    print("\nNext steps:")
    print("1. Install all dependencies: pip install -r requirements.txt")
    print("2. Create .env file with configuration")
    print("3. Run: uvicorn src.api.main:app --reload --port 8000")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nCheck that:")
    print("1. You're in the project root directory")
    print("2. PYTHONPATH is set correctly")
    print("3. All required packages are installed")
    print("\nTo set PYTHONPATH:")
    print("  Windows: set PYTHONPATH=%CD%")
    print("  Linux/Mac: export PYTHONPATH=\"${PYTHONPATH}:$(pwd)\"")
    import sys
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

