"""
test_core.py - Quick test to verify all core components work
Run: python test_core.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("Math Mentor AI - Core Component Tests")
print("=" * 60)

# Test 1: Config
print("\n[1] Testing Config...")
try:
    from config import config
    print(f"   Config loaded. Provider: {config.LLM_PROVIDER}, Model: {config.LLM_MODEL}")
except Exception as e:
    print(f"   Config failed: {e}")

# Test 2: Math Tools
print("\n[2] Testing Math Tools (SymPy)...")
try:
    from utils.math_tools import solve_equation, differentiate, compute_limit, safe_eval_math
    result = solve_equation("x**2 - 5*x + 6", "x")
    print(f"   Solve x²-5x+6=0: {result}")
    result2 = differentiate("x**3 + 2*x", "x")
    print(f"   d/dx(x³+2x): {result2}")
    result3 = compute_limit("sin(x)/x", "x", "0")
    print(f"   lim(x→0) sin(x)/x: {result3}")
except Exception as e:
    print(f"   Math tools failed: {e}")

# Test 3: Memory
print("\n[3] Testing Memory Store...")
try:
    from memory.memory_store import MemoryStore
    mem = MemoryStore("./data/test_memory.json")
    sid = mem.save_session({
        "input_type": "text",
        "raw_input": "Test problem",
        "topic": "algebra",
        "solution": "x = 3",
    })
    mem.update_feedback(sid, "correct")
    stats = mem.get_stats()
    print(f"   Memory works. Session {sid} saved. Stats: {stats}")
    import os
    os.remove("./data/test_memory.json")
except Exception as e:
    print(f"   Memory failed: {e}")

# Test 4: LLM Client
print("\n[4] Testing LLM Client...")
try:
    from utils.llm_client import get_llm_response, parse_json_response
    if config.ANTHROPIC_API_KEY or config.OPENAI_API_KEY:
        response = get_llm_response("What is 2+2? Reply with just the number.", max_tokens=10)
        print(f"   LLM response: {response[:50]}")
    else:
        print("   No API key set - skipping LLM test")
except Exception as e:
    print(f"   LLM failed: {e}")

# Test 5: RAG Pipeline (without model)
print("\n[5] Testing RAG Fallback...")
try:
    from rag.rag_pipeline import RAGPipeline, KNOWLEDGE_BASE
    rag = RAGPipeline()
    # Test fallback search
    results = rag._fallback_retrieve("quadratic equation solve", "algebra", 3)
    print(f"   Fallback retrieval: {len(results)} chunks, top: {results[0]['title'] if results else 'none'}")
    print(f"   Knowledge base has {len(KNOWLEDGE_BASE)} documents")
except Exception as e:
    print(f"   RAG failed: {e}")

# Test 6: Agent imports
print("\n[6] Testing Agent Imports...")
try:
    from agents.agents import (
        ParserAgent, IntentRouterAgent, SolverAgent,
        VerifierAgent, ExplainerAgent, GuardrailAgent, AgentOrchestrator
    )
    print("   All 6 agents imported successfully")
    orch = AgentOrchestrator()
    print("   Orchestrator created")
except Exception as e:
    print(f"   Agent import failed: {e}")

print("\n" + "=" * 60)
print("Core tests complete!")
print("Run 'streamlit run app.py' to start the application.")
print("=" * 60)
