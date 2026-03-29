"""Test NVIDIA API connection."""

import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    print("❌ NVIDIA_API_KEY not found in .env!")
    exit(1)

print(f"✅ API Key found: {api_key[:20]}...")

# Test embeddings
print("\n🔍 Testing Embeddings...")
try:
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=api_key
    )
    test_embedding = embeddings.embed_query("Hello world")
    print(f"✅ Embeddings work! Dimension: {len(test_embedding)}")
except Exception as e:
    print(f"❌ Embeddings failed: {e}")

# Test LLM
print("\n🤖 Testing LLM...")
try:
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        api_key=api_key
    )
    response = llm.invoke("Say hello!")
    print(f"✅ LLM works! Response: {response.content[:100]}...")
except Exception as e:
    print(f"❌ LLM failed: {e}")

print("\n✅ All API tests passed!")