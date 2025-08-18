import os
import faiss
import torch
import numpy as np
import asyncio
from ..config import EMBEDDINGS_DIR, DIM
from ..embeddings.generator_async import AsyncEmbeddingGenerator


class AsyncDatabaseManager:
    """Async database manager for embeddings storage and retrieval"""
    
    @staticmethod
    def embeddings_exist(document_name: str, chunk_count: int) -> bool:
        """Check if embeddings already exist for a document"""
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f"document_embeddings_{document_name}.faiss")
        
        if not os.path.exists(embeddings_path):
            return False
        
        # Load index to verify it has the correct number of vectors
        try:
            index = faiss.read_index(embeddings_path)
            return index.ntotal == chunk_count
        except Exception:
            return False
    
    @staticmethod
    async def store_embeddings_async(chunks: list, document_name: str):
        """
        Async generate and store embeddings for document chunks using FAISS
        
        Args:
            chunks: List of document text chunks
            document_name: Name of the document for storage
        """
        print(f"🔄 Generating embeddings for {len(chunks)} chunks (async)...")
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        # Generate embeddings asynchronously
        async with AsyncEmbeddingGenerator() as generator:
            embeddings_tensor = await generator.get_embeddings_for_chunks_async(chunks)
        
        # Convert to numpy for FAISS
        embeddings_np = embeddings_tensor.cpu().numpy().astype('float32')
        
        print(f"📊 Embeddings shape: {embeddings_np.shape}")
        
        # Create FAISS index
        index = faiss.IndexFlatIP(DIM)  # Inner product similarity
        index.add(embeddings_np)
        
        # Save the index
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f"document_embeddings_{document_name}.faiss")
        faiss.write_index(index, embeddings_path)
        
        print(f"✅ Stored {len(chunks)} embeddings in {embeddings_path}")
        return embeddings_path
    
    @staticmethod
    def load_faiss_index(document_name: str):
        """Load FAISS index from disk"""
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f"document_embeddings_{document_name}.faiss")
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        try:
            index = faiss.read_index(embeddings_path)
            print(f"📚 Loaded FAISS index with {index.ntotal} vectors from {embeddings_path}")
            return index
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")
    
    @staticmethod
    def delete_embeddings(document_name: str) -> bool:
        """Delete embeddings file for a document"""
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f"document_embeddings_{document_name}.faiss")
        
        try:
            if os.path.exists(embeddings_path):
                os.remove(embeddings_path)
                print(f"🗑️ Deleted embeddings: {embeddings_path}")
                return True
            return False
        except Exception as e:
            print(f"❌ Error deleting embeddings: {e}")
            return False
    
    @staticmethod
    def list_stored_documents() -> list:
        """List all documents that have stored embeddings"""
        if not os.path.exists(EMBEDDINGS_DIR):
            return []
        
        documents = []
        for filename in os.listdir(EMBEDDINGS_DIR):
            if filename.startswith("document_embeddings_") and filename.endswith(".faiss"):
                # Extract document name from filename
                doc_name = filename[20:-6]  # Remove "document_embeddings_" and ".faiss"
                documents.append(doc_name)
        
        return documents
    
    @staticmethod
    def get_index_info(document_name: str) -> dict:
        """Get information about a stored FAISS index"""
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f"document_embeddings_{document_name}.faiss")
        
        if not os.path.exists(embeddings_path):
            return {"exists": False}
        
        try:
            index = faiss.read_index(embeddings_path)
            file_size = os.path.getsize(embeddings_path)
            
            return {
                "exists": True,
                "vector_count": index.ntotal,
                "dimension": index.d,
                "file_size_mb": file_size / (1024 * 1024),
                "file_path": embeddings_path
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}


# Async wrapper functions
async def store_embeddings_async(chunks: list, document_name: str):
    """Async wrapper for embedding storage"""
    return await AsyncDatabaseManager.store_embeddings_async(chunks, document_name)

# Sync wrapper for compatibility
def store_embeddings_sync(chunks: list, document_name: str):
    """Sync wrapper around async embedding storage"""
    return asyncio.run(store_embeddings_async(chunks, document_name))


# Example usage and testing
async def test_async_database():
    """Test async database operations"""
    print("🧪 Testing async database operations...")
    
    # Test data
    test_chunks = [
        "This is the first test chunk for embedding generation.",
        "This is the second test chunk with different content.",
        "The third chunk contains information about testing procedures.",
        "Final test chunk to complete the embedding test dataset."
    ]
    
    test_document_name = "test_async_doc"
    
    try:
        # Test embedding storage
        print("📝 Testing async embedding storage...")
        await AsyncDatabaseManager.store_embeddings_async(test_chunks, test_document_name)
        
        # Test loading
        print("📚 Testing index loading...")
        index = AsyncDatabaseManager.load_faiss_index(test_document_name)
        print(f"✅ Loaded index with {index.ntotal} vectors")
        
        # Test info retrieval
        print("📊 Testing info retrieval...")
        info = AsyncDatabaseManager.get_index_info(test_document_name)
        print(f"📋 Index info: {info}")
        
        # Test listing documents
        print("📋 Testing document listing...")
        documents = AsyncDatabaseManager.list_stored_documents()
        print(f"📚 Stored documents: {documents}")
        
        # Cleanup
        print("🗑️ Cleaning up test data...")
        AsyncDatabaseManager.delete_embeddings(test_document_name)
        
        print("✅ All async database tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        # Try to cleanup even if test failed
        AsyncDatabaseManager.delete_embeddings(test_document_name)


if __name__ == "__main__":
    asyncio.run(test_async_database())
