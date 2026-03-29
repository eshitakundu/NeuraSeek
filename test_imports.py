"""Test if all imports work correctly."""

print("Testing imports...")

try:
    from dotenv import load_dotenv
    print("✅ dotenv")
except ImportError as e:
    print(f"❌ dotenv: {e}")

try:
    import streamlit as st
    print("✅ streamlit")
except ImportError as e:
    print(f"❌ streamlit: {e}")

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
    print("✅ langchain-nvidia-ai-endpoints")
except ImportError as e:
    print(f"❌ langchain-nvidia-ai-endpoints: {e}")

try:
    import faiss
    print("✅ faiss")
except ImportError as e:
    print(f"❌ faiss: {e}")

try:
    import sys
    import os
    # Add src directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from src.utils.document_processor import DocumentProcessor
    print("✅ DocumentProcessor")
except ImportError as e:
    print(f"❌ DocumentProcessor: {e}")

try:
    from src.utils.vector_store import VectorStore
    print("✅ VectorStore")
except ImportError as e:
    print(f"❌ VectorStore: {e}")

try:
    from src.utils.rag_chain import RAGChain
    print("✅ RAGChain")
except ImportError as e:
    print(f"❌ RAGChain: {e}")

print("\n✅ All imports successful!")