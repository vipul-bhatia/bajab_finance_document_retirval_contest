import os
import sqlite3
import torch
import glob
from ..config import DIM, device
from ..embeddings import EmbeddingGenerator


class DatabaseManager:
    """Handles SQLite database operations for embeddings"""
    
    @staticmethod
    def get_db_path(document_name: str) -> str:
        """Get SQLite database path for a document"""
        return f"document_embeddings_{document_name}.db"
    
    @staticmethod
    def ensure_table_exists(document_name: str):
        """Creates the document_embeddings table if it doesn't exist."""
        db_path = DatabaseManager.get_db_path(document_name)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
              chunk_index INTEGER PRIMARY KEY,
              chunk_text TEXT,
              embedding BLOB
            );
        """)
        conn.commit()
        conn.close()
        print(f"✅ Database table ready: {db_path}")
    
    @staticmethod
    def store_embeddings(chunks: list, document_name: str):
        """Generate and store embeddings for all document chunks in parallel (embedding generation only)"""
        import concurrent.futures
        import time
        
        total_start = time.time()
        
        DatabaseManager.ensure_table_exists(document_name)
        db_path = DatabaseManager.get_db_path(document_name)
        
        def generate_embedding(idx_chunk):
            idx, chunk = idx_chunk
            vec = EmbeddingGenerator.get_embedding(chunk)
            blob = vec.cpu().numpy().tobytes()
            return (idx, chunk, blob)
        
        # Parallel embedding generation
        embedding_start = time.time()
        print(f"      🔄 Generating embeddings for {len(chunks)} chunks...")
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for result in executor.map(generate_embedding, enumerate(chunks)):
                results.append(result)
                idx = result[0]
                if (idx + 1) % 10 == 0 or (idx + 1) == len(chunks):
                    print(f"      Processed embeddings for {idx + 1}/{len(chunks)} chunks")
        
        embedding_time = time.time() - embedding_start
        print(f"      ✅ Embedding generation: {embedding_time:.2f}s")
        
        # Sort results by idx to maintain order
        results.sort(key=lambda x: x[0])
        
        # Store in database (single thread)
        db_start = time.time()
        print(f"      💾 Storing embeddings in database...")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM document_embeddings;")
        sql = "INSERT INTO document_embeddings(chunk_index, chunk_text, embedding) VALUES(?,?,?);"
        for idx, chunk, blob in results:
            cur.execute(sql, (idx, chunk, blob))
        conn.commit()
        conn.close()
        db_time = time.time() - db_start
        print(f"      ✅ Database storage: {db_time:.2f}s")
        
        total_time = time.time() - total_start
        print(f"      🏁 Total embedding pipeline: {total_time:.2f}s")
        print(f"✅ All embeddings stored successfully in {db_path}!")
    
    @staticmethod
    def load_embeddings(M: int, document_name: str) -> tuple[torch.Tensor, list]:
        """Load precomputed embeddings into Torch tensors on device"""
        
        db_path = DatabaseManager.get_db_path(document_name)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get count of embeddings
        cursor.execute("SELECT COUNT(*) FROM document_embeddings")
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.close()
            print(f"❌ No embeddings found in {db_path}")
            return torch.zeros((M, DIM), dtype=torch.float32, device=device), []
        
        # Load embeddings and texts
        E = torch.zeros((count, DIM), dtype=torch.float32, device=device)
        chunks = []
        
        cursor.execute("SELECT chunk_index, chunk_text, embedding FROM document_embeddings ORDER BY chunk_index ASC")
        
        for idx, chunk_text, blob in cursor.fetchall():
            vec = torch.frombuffer(blob, dtype=torch.float32).to(device)
            E[idx] = vec
            chunks.append(chunk_text)
        
        conn.close()
        print(f"✅ Loaded {count} embeddings from {db_path}")
        return E, chunks
    
    @staticmethod
    def list_documents():
        """List all available document databases"""
        db_files = glob.glob("document_embeddings_*.db")
        if not db_files:
            print("No document databases found.")
            return []
        
        documents = []
        for db_file in db_files:
            # Extract document name from filename
            doc_name = db_file.replace("document_embeddings_", "").replace(".db", "")
            
            # Get chunk count
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_embeddings")
            count = cur.fetchone()[0]
            conn.close()
            
            documents.append((doc_name, count, db_file))
        
        print("Available document databases:")
        for i, (doc_name, count, db_file) in enumerate(documents, 1):
            print(f"  {i}. {doc_name} ({count} chunks) - {db_file}")
        
        return documents
    
    @staticmethod
    def delete_document(document_name: str):
        """Delete a document database"""
        db_path = DatabaseManager.get_db_path(document_name)
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"✅ Deleted document database: {db_path}")
        else:
            print(f"❌ Document database not found: {db_path}")
    
    @staticmethod
    def embeddings_exist(document_name: str, expected_count: int) -> bool:
        """Check if embeddings exist and match expected count"""
        db_path = DatabaseManager.get_db_path(document_name)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='document_embeddings'")
        table_exists = cur.fetchone()[0] > 0
        
        if table_exists:
            cur.execute("SELECT COUNT(*) FROM document_embeddings")
            count = cur.fetchone()[0]
            conn.close()
            return count == expected_count
        
        conn.close()
        return False 