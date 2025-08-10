import os
import faiss
import torch
import glob
import numpy as np
from ..config import DIM, device, EMBEDDINGS_DIR
from ..embeddings import EmbeddingGenerator


class DatabaseManager:
    """Handles FAISS index operations for embeddings with parallel batch processing."""
    
    @staticmethod
    def get_faiss_index_path(document_name: str) -> str:
        """Get FAISS index path for a document"""
        # Ensure embeddings directory exists
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        return os.path.join(EMBEDDINGS_DIR, f"document_embeddings_{document_name}.faiss")
    
    @staticmethod
    def store_embeddings(chunks: list, document_name: str, batch_size: int = 50, max_workers: int = 10):
        """Generate and store embeddings for all document chunks in a FAISS index using parallel batching."""
        import time
        
        total_start = time.time()
        
        faiss_index_path = DatabaseManager.get_faiss_index_path(document_name)
        
        # Generate embeddings in parallel with batching
        embedding_start = time.time()
        print(f"      🔄 Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = EmbeddingGenerator.get_embeddings_in_parallel(chunks, batch_size=batch_size, max_workers=max_workers)
        
        embedding_time = time.time() - embedding_start
        print(f"      ✅ Embedding generation: {embedding_time:.2f}s")
        
        # Create and store FAISS index
        index_start = time.time()
        print(f"      💾 Creating and storing FAISS index...")
        
        index = faiss.IndexFlatL2(DIM)
        index.add(embeddings.cpu().numpy())
        
        faiss.write_index(index, faiss_index_path)
        
        index_time = time.time() - index_start
        print(f"      ✅ FAISS index creation and storage: {index_time:.2f}s")
        
        total_time = time.time() - total_start
        print(f"      🏁 Total embedding pipeline: {total_time:.2f}s")
        print(f"✅ All embeddings stored successfully in {faiss_index_path}!")
    
    @staticmethod
    def load_faiss_index(document_name: str):
        """Load a FAISS index from disk"""
        faiss_index_path = DatabaseManager.get_faiss_index_path(document_name)
        # Backward-compatibility: also check legacy path in root if not found
        if not os.path.exists(faiss_index_path):
            legacy_path = f"document_embeddings_{document_name}.faiss"
            if os.path.exists(legacy_path):
                print(f"⚠️ Found legacy FAISS index at root. Migrating to {EMBEDDINGS_DIR}...")
                os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
                new_path = faiss_index_path
                try:
                    os.replace(legacy_path, new_path)
                    faiss_index_path = new_path
                    print(f"✅ Migrated {legacy_path} → {new_path}")
                except Exception as e:
                    print(f"⚠️ Migration failed: {e}. Will attempt to read from legacy path.")
                    faiss_index_path = legacy_path
            else:
                print(f"❌ FAISS index not found: {faiss_index_path}")
                return None
        
        index = faiss.read_index(faiss_index_path)
        print(f"✅ Loaded FAISS index from {faiss_index_path}")
        return index
    
    @staticmethod
    def list_documents():
        """List all available document FAISS indexes"""
        # Look inside the embeddings directory and also legacy root for any leftover files
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        faiss_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "document_embeddings_*.faiss"))
        legacy_files = glob.glob("document_embeddings_*.faiss")
        faiss_files.extend([f for f in legacy_files if f not in faiss_files])
        if not faiss_files:
            print("No document FAISS indexes found.")
            return []
        
        documents = []
        for faiss_file in faiss_files:
            base_name = os.path.basename(faiss_file)
            doc_name = base_name.replace("document_embeddings_", "").replace(".faiss", "")
            index = faiss.read_index(faiss_file)
            documents.append((doc_name, index.ntotal, faiss_file))
        
        print("Available document FAISS indexes:")
        for i, (doc_name, count, faiss_file) in enumerate(documents, 1):
            print(f"  {i}. {doc_name} ({count} chunks) - {faiss_file}")
        
        return documents
    
    @staticmethod
    def delete_document(document_name: str):
        """Delete a document FAISS index"""
        faiss_index_path = DatabaseManager.get_faiss_index_path(document_name)
        legacy_path = f"document_embeddings_{document_name}.faiss"
        deleted_any = False
        if os.path.exists(faiss_index_path):
            os.remove(faiss_index_path)
            print(f"✅ Deleted document FAISS index: {faiss_index_path}")
            deleted_any = True
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
            print(f"✅ Deleted legacy document FAISS index: {legacy_path}")
            deleted_any = True
        if not deleted_any:
            print(f"❌ Document FAISS index not found: {faiss_index_path}")
        else:
            print(f"❌ Document FAISS index not found: {faiss_index_path}")
    
    @staticmethod
    def embeddings_exist(document_name: str, expected_count: int) -> bool:
        """Check if a FAISS index exists and matches the expected count"""
        faiss_index_path = DatabaseManager.get_faiss_index_path(document_name)
        # Prefer new path; fallback to legacy if needed
        candidate_paths = [faiss_index_path, f"document_embeddings_{document_name}.faiss"]
        for path in candidate_paths:
            if os.path.exists(path):
                index = faiss.read_index(path)
                return index.ntotal == expected_count
        return False 