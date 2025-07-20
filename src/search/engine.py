import torch
import time
from ..config import TOP_K, device
from ..embeddings import EmbeddingGenerator


class SearchEngine:
    """Handles semantic search operations"""
    
    def __init__(self):
        self.embeddings = None
        self.document_chunks = []
        self.document_name = None
    
    def load_embeddings(self, embeddings: torch.Tensor, chunks: list, document_name: str):
        """Load embeddings and chunks for search"""
        self.embeddings = embeddings
        self.document_chunks = chunks
        self.document_name = document_name
    
    def find_relevant_chunks(self, query: str, top_k: int = TOP_K):
        """Find relevant document chunks using semantic similarity"""
        if self.embeddings is None or not self.document_chunks:
            raise RuntimeError("Embeddings not initialized. Load embeddings first.")
        
        # Get query embedding
        q_vec = EmbeddingGenerator.get_embedding(query)
        
        # Calculate similarities
        similarities = self.embeddings @ q_vec  # (num_chunks,) on device
        
        # Get top results
        scores, indices = torch.topk(similarities, k=min(top_k, len(self.document_chunks)))
        
        results = []
        for score, i in zip(scores, indices):
            results.append({
                "chunk_index": i.item(),
                "text": self.document_chunks[i.item()],
                "score": score.item()
            })
        
        return results
    
    def search_and_display(self, query: str, top_k: int = TOP_K):
        """Search and display results with timing"""
        start = time.time()
        try:
            results = self.find_relevant_chunks(query, top_k)
            elapsed = time.time() - start
            
            print(f"\nFound {len(results)} relevant chunks (took {elapsed:.3f}s):\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['score']:.4f}")
                print(f"   Text: {result['text']}")
                print(f"   [Chunk {result['chunk_index']}]")
                print()
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return [] 