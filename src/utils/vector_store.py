import os
import pickle
from typing import List, Tuple
import faiss
import numpy as np
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


class VectorStore:    
    def __init__(self, api_key: str, model: str = "nvidia/nv-embedqa-e5-v5"):
        self.embeddings = NVIDIAEmbeddings(
            model=model,
            api_key=api_key,
            truncate="END"
        )
        self.dimension = 1024  # NV-Embed dimension
        self.index = None
        self.chunks = []
        self.metadata = []
    
    def create_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.metadata = []
    
    def add_documents(self, chunks: List[str], doc_name: str):
        if self.index is None:
            self.create_index()
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(chunks)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        self.metadata.extend([{"source": doc_name, "chunk_id": i} 
                             for i in range(len(chunks))])
    
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, dict]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_array, k)
        
        # Return results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], self.metadata[idx]))
        
        return results
    
    def save(self, path: str = "data/vector_store"):
        os.makedirs(path, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, f"{path}/index.faiss")
        
        with open(f"{path}/chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path: str = "data/vector_store"):
        if os.path.exists(f"{path}/index.faiss"):
            self.index = faiss.read_index(f"{path}/index.faiss")
        
        if os.path.exists(f"{path}/chunks.pkl"):
            with open(f"{path}/chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
        
        if os.path.exists(f"{path}/metadata.pkl"):
            with open(f"{path}/metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
    
    def clear(self):
        self.create_index()